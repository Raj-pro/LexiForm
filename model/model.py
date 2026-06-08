import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.config import ModelConfig
from model.blocks import EncoderBlock, DecoderBlock, RMSNorm


class CopyGate(nn.Module):
    """
    Pointer-generator gate.
    Returns a raw logit (pre-sigmoid). Caller uses F.logsigmoid(+logit) /
    F.logsigmoid(-logit) for an exact, allocation-free log-space mixture.

    Input features: [context ; decoder_state ; decoder_input_emb]. The
    decoder-input embedding (post-lookup, pre-stack) is concatenated alongside
    the decoder hidden state — required for compatibility with the trained
    checkpoint, which was produced with this 3-way input.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(d_model * 3, 1, bias=True)

    def forward(
        self,
        context: torch.Tensor,        # (B, T, d) weighted sum of encoder states
        decoder_state: torch.Tensor,  # (B, T, d) decoder hidden state
        decoder_input: torch.Tensor,  # (B, T, d) raw decoder-input embedding
    ) -> torch.Tensor:
        combined = torch.cat([context, decoder_state, decoder_input], dim=-1)
        return self.linear(combined)  # (B, T, 1) — raw logit


def _build_src_pad_mask(
    src_ids: torch.Tensor, pad_id: int, dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Returns additive float mask of shape (B, 1, 1, T_src) with -inf at padding."""
    pad = (src_ids == pad_id)  # (B, T_src)
    mask = torch.zeros(pad.shape, dtype=dtype, device=pad.device)
    mask = mask.masked_fill(pad, float("-inf"))
    return mask.unsqueeze(1).unsqueeze(1)


class ParaphraseModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config    = config
        # Embedding and output projection are sized to include the
        # `num_sentinels` ids appended after the BPE vocab (T5 span corruption).
        # At num_sentinels=0 this is identical to the original architecture.
        V = config.effective_vocab_size
        self.embedding = nn.Embedding(V, config.d_model, padding_idx=config.pad_id)
        self.enc_norm  = RMSNorm(config.d_model)
        self.dec_norm  = RMSNorm(config.d_model)
        self.encoder   = nn.ModuleList([EncoderBlock(config) for _ in range(config.num_encoder_layers)])
        self.decoder   = nn.ModuleList([DecoderBlock(config) for _ in range(config.num_decoder_layers)])
        self.output_proj = nn.Linear(config.d_model, V, bias=False)
        self.output_proj.weight = self.embedding.weight  # tie weights
        self.drop = nn.Dropout(config.dropout)

        if config.use_copy:
            self.copy_gate = CopyGate(config.d_model)

        self._init_weights()

    def _init_weights(self):
        d = self.config.d_model
        for name, p in self.named_parameters():
            if "copy_gate" in name:
                continue  # handled explicitly below
            if name == "embedding.weight":
                # Tied with the output projection — Xavier on (V, d) gives a
                # per-element scale ~1/sqrt(V+d), starving the logit gradient.
                # Normal(0, d^-0.5) matches T5/LLaMA-style weight-tied init.
                nn.init.normal_(p, mean=0.0, std=d ** -0.5)
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # Zero-init the copy gate so logit==0 -> p_gen==0.5 at step 0,
        # giving an exact 50/50 mix between vocab and copy distributions.
        if self.config.use_copy:
            nn.init.zeros_(self.copy_gate.linear.weight)
            nn.init.zeros_(self.copy_gate.linear.bias)

    # ---------- encoding ----------

    def encode(self, src_ids: torch.Tensor) -> torch.Tensor:
        src_attn_mask = _build_src_pad_mask(
            src_ids, self.config.pad_id, dtype=self.embedding.weight.dtype,
        )
        # sqrt(d_model) scaling: tied embeddings double as the output
        # projection, so the unscaled lookup leaves both the residual stream
        # and the output logits under-magnitude.
        scale = math.sqrt(self.config.d_model)
        x = self.drop(self.embedding(src_ids) * scale)
        for layer in self.encoder:
            x = layer(x, attn_mask=src_attn_mask)
        return self.enc_norm(x)

    # ---------- internals: a single decoder pass, parallel or incremental ----------

    def _decoder_pass(
        self,
        tgt_ids: torch.Tensor,
        encoder_out: torch.Tensor,
        src_attn_mask: torch.Tensor,
        layer_caches: list[dict] | None,
        start_pos: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        scale = math.sqrt(self.config.d_model)
        x_emb = self.drop(self.embedding(tgt_ids) * scale)
        x = x_emb

        last_idx = len(self.decoder) - 1
        attn_weights = None
        for i, layer in enumerate(self.decoder):
            layer_cache = layer_caches[i] if layer_caches is not None else None
            # Only the last layer's cross-attn weights feed the copy mechanism,
            # so the earlier layers can use SDPA (Flash kernels) and skip the
            # manual softmax + dropout-on-probs path.
            need_w = self.config.use_copy and (i == last_idx)
            x, attn_weights = layer(
                x, encoder_out,
                src_attn_mask=src_attn_mask,
                self_attn_mask=None,  # rely on SDPA causal flag / cache semantics
                layer_cache=layer_cache,
                start_pos=start_pos,
                need_weights=need_w,
            )

        x = self.dec_norm(x)
        return x, attn_weights, x_emb  # attn_weights: (B, H, T_tgt, T_src) from last layer

    def _project_and_copy(
        self,
        dec_state: torch.Tensor,           # (B, T_tgt, d)
        encoder_out: torch.Tensor,         # (B, T_src, d)
        attn_weights: torch.Tensor,        # (B, H, T_tgt, T_src)
        src_ids: torch.Tensor | None,
        dec_input_emb: torch.Tensor,       # (B, T_tgt, d) raw decoder-input embedding
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Returns (scores, attn_avg).

        scores   — log-probs when use_copy=True, raw logits otherwise.
        attn_avg — (B, T_tgt, T_src) mean-over-heads cross-attention, or None
                   when copy is disabled / src_ids is None.  Exposed so the
                   inference loop can read the attended source position for the
                   phrase-table bias without a second forward pass.
        """
        vocab_logits = self.output_proj(dec_state)  # (B, T, V)

        if not self.config.use_copy or src_ids is None:
            return vocab_logits, None  # caller treats as raw logits

        B, T_tgt, V = vocab_logits.shape

        # average per-head attention to a single distribution for copy
        attn_avg = attn_weights.mean(dim=1)  # (B, T_tgt, T_src)

        # log-domain mixture: avoids the log(p + eps) numerical issue
        log_vocab_probs = F.log_softmax(vocab_logits, dim=-1)  # (B, T, V)

        src_expanded = src_ids.unsqueeze(1).expand(B, T_tgt, -1)  # (B, T_tgt, T_src) — view
        # accumulate copy mass in prob space then log; safe because per-token copy mass is >= 0
        copy_probs = torch.zeros_like(log_vocab_probs)
        copy_probs.scatter_add_(2, src_expanded, attn_avg)
        log_copy_probs = torch.log(copy_probs.clamp_min(1e-20))

        context     = torch.bmm(attn_avg, encoder_out)             # (B, T_tgt, d)
        p_gen_logit = self.copy_gate(context, dec_state, dec_input_emb)  # (B, T_tgt, 1) — raw

        # F.logsigmoid is numerically exact for both branches — no clamp needed.
        log_p_gen  = F.logsigmoid( p_gen_logit)
        log_p_copy = F.logsigmoid(-p_gen_logit)
        final_log_probs = torch.logaddexp(
            log_p_gen  + log_vocab_probs,
            log_p_copy + log_copy_probs,
        )
        return final_log_probs, attn_avg  # caller treats scores as log-probs

    # ---------- training-time parallel decode ----------

    def decode(
        self,
        tgt_ids: torch.Tensor,
        encoder_out: torch.Tensor,
        src_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if src_ids is not None:
            src_attn_mask = _build_src_pad_mask(
                src_ids, self.config.pad_id, dtype=encoder_out.dtype,
            )
        else:
            src_attn_mask = None
        dec_state, attn_weights, dec_input_emb = self._decoder_pass(
            tgt_ids, encoder_out, src_attn_mask, layer_caches=None, start_pos=0,
        )
        scores, _ = self._project_and_copy(
            dec_state, encoder_out, attn_weights, src_ids, dec_input_emb,
        )
        return scores

    # ---------- inference-time incremental decode ----------

    def init_caches(self, num_layers: int | None = None) -> list[dict]:
        n = num_layers if num_layers is not None else self.config.num_decoder_layers
        return [{"self": {}, "cross": {}} for _ in range(n)]

    @torch.no_grad()
    def decode_step(
        self,
        token: torch.Tensor,           # (B, 1)
        encoder_out: torch.Tensor,     # (B, T_src, d)
        src_attn_mask: torch.Tensor,   # (B, 1, 1, T_src), additive float
        src_ids: torch.Tensor | None,
        layer_caches: list[dict],
        step: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Returns (scores, attn_avg).

        attn_avg is (B, 1, T_src) when use_copy=True, else None.
        The inference loop reads attn_avg to identify the attended source
        position for the phrase-table bias without a second forward pass.
        """
        dec_state, attn_weights, dec_input_emb = self._decoder_pass(
            token, encoder_out, src_attn_mask, layer_caches=layer_caches, start_pos=step,
        )
        return self._project_and_copy(
            dec_state, encoder_out, attn_weights, src_ids, dec_input_emb,
        )

    # ---------- top-level forward (training) ----------

    def forward(
        self,
        src_ids: torch.Tensor,
        dec_input: torch.Tensor,
    ) -> torch.Tensor:
        encoder_out = self.encode(src_ids)
        return self.decode(dec_input, encoder_out, src_ids=src_ids)

    # ---------- simple greedy generate (KV-cached) — kept for quick smoke tests ----------

    @torch.no_grad()
    def generate(
        self,
        src_ids: torch.Tensor,
        max_new_tokens: int = 128,
    ) -> torch.Tensor:
        self.eval()
        device = src_ids.device
        encoder_out = self.encode(src_ids)
        src_attn_mask = _build_src_pad_mask(
            src_ids, self.config.pad_id, dtype=encoder_out.dtype,
        )
        B = src_ids.size(0)
        caches = self.init_caches()

        cur = torch.full((B, 1), self.config.bos_id, dtype=torch.long, device=device)
        out_tokens = []
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for step in range(max_new_tokens):
            logits_or_lp, _ = self.decode_step(cur, encoder_out, src_attn_mask, src_ids, caches, step)
            next_token = logits_or_lp[:, -1, :].argmax(dim=-1, keepdim=True)
            out_tokens.append(next_token)
            finished = finished | (next_token.squeeze(-1) == self.config.eos_id)
            if finished.all():
                break
            cur = next_token

        return torch.cat(out_tokens, dim=1)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
