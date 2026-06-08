import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def precompute_rope(head_dim: int, max_seq_len: int, base: int = 10000) -> tuple[torch.Tensor, torch.Tensor]:
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    pos = torch.arange(max_seq_len).float()
    freqs = torch.outer(pos, theta)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return cos, sin  # (max_seq_len, head_dim//2)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
    # x: (B, heads, T, head_dim) — rotates positions [start_pos, start_pos+T)
    T = x.size(-2)
    cos = cos[start_pos:start_pos + T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, D//2)
    sin = sin[start_pos:start_pos + T].unsqueeze(0).unsqueeze(0)
    x1, x2 = x[..., ::2], x[..., 1::2]
    x_even = x1 * cos - x2 * sin
    x_odd  = x2 * cos + x1 * sin
    return torch.stack([x_even, x_odd], dim=-1).flatten(-2)


class MultiHeadAttention(nn.Module):
    """
    Self-attention with RoPE and optional KV cache.

    kv_cache: dict, mutated in-place. Pass {} on the first incremental step;
    on subsequent steps pass the same dict to grow the cache.
    """
    def __init__(self, config, causal: bool = False):
        super().__init__()
        assert config.d_model % config.num_heads == 0
        self.num_heads = config.num_heads
        self.head_dim  = config.d_model // config.num_heads
        self.causal    = causal
        self.dropout_p = config.dropout

        self.q   = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k   = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v   = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out = nn.Linear(config.d_model, config.d_model, bias=False)

        cos, sin = precompute_rope(self.head_dim, config.max_seq_len)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

    def _split_heads(self, t: torch.Tensor) -> torch.Tensor:
        B, T, _ = t.shape
        return t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        kv_cache: dict | None = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        B, T_new, C = x.shape

        q     = apply_rope(self._split_heads(self.q(x)), self.rope_cos, self.rope_sin, start_pos)
        k_new = apply_rope(self._split_heads(self.k(x)), self.rope_cos, self.rope_sin, start_pos)
        v_new = self._split_heads(self.v(x))

        if kv_cache is not None and kv_cache.get("k") is not None:
            k = torch.cat([kv_cache["k"], k_new], dim=2)
            v = torch.cat([kv_cache["v"], v_new], dim=2)
        else:
            k = k_new
            v = v_new

        if kv_cache is not None:
            kv_cache["k"] = k
            kv_cache["v"] = v

        # Causal masking only applies to the initial parallel pass.
        # During incremental decoding, the new query attends to all cached K/V.
        incremental = kv_cache is not None and k.size(2) > T_new
        is_causal   = self.causal and not incremental and attn_mask is None
        dropout_p   = self.dropout_p if self.training else 0.0

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            is_causal=is_causal,
            dropout_p=dropout_p,
        )
        out = out.transpose(1, 2).contiguous().view(B, T_new, C)
        return self.out(out)


class CrossAttention(nn.Module):
    """
    Encoder-decoder cross-attention. When `need_weights=True`, runs the manual
    softmax path and returns full per-head attention weights (B, H, T_tgt, T_src)
    for the pointer-generator copy mechanism. When False, dispatches to
    `F.scaled_dot_product_attention` (Flash kernels) and returns weights=None.
    Encoder K/V are cached on first call.
    """
    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.num_heads == 0
        self.num_heads = config.num_heads
        self.head_dim  = config.d_model // config.num_heads
        self.dropout_p = config.dropout

        self.q   = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k   = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v   = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out = nn.Linear(config.d_model, config.d_model, bias=False)
        self.drop = nn.Dropout(config.dropout)

    def _split(self, t: torch.Tensor) -> torch.Tensor:
        B, T, _ = t.shape
        return t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        enc: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        kv_cache: dict | None = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T_new, C = x.shape

        q = self._split(self.q(x))

        if kv_cache is not None and kv_cache.get("k") is not None:
            k = kv_cache["k"]
            v = kv_cache["v"]
        else:
            k = self._split(self.k(enc))
            v = self._split(self.v(enc))
            if kv_cache is not None:
                kv_cache["k"] = k
                kv_cache["v"] = v

        if not need_weights:
            dropout_p = self.dropout_p if self.training else 0.0
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=False,
            )
            out = out.transpose(1, 2).contiguous().view(B, T_new, C)
            return self.out(out), None

        # Manual path — needed when the caller wants the softmaxed weights
        # (last decoder layer, feeding the copy distribution).
        scale  = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, H, T_new, S)

        if attn_mask is not None:
            scores = scores + attn_mask  # additive float mask, -inf at padded positions

        attn      = F.softmax(scores, dim=-1)                  # (B, H, T_new, S)
        attn_drop = self.drop(attn)
        out       = torch.matmul(attn_drop, v)
        out       = out.transpose(1, 2).contiguous().view(B, T_new, C)
        return self.out(out), attn
