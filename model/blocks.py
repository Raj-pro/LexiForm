import torch
import torch.nn as nn
from model.attention import MultiHeadAttention, CrossAttention
from model.config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * x / rms


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up   = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual-side dropout is applied by the enclosing block; do not
        # double-drop here (the previous version produced an effective rate
        # of 1 - (1-p)**2, ~0.19 at p=0.1).
        return self.down(nn.functional.silu(self.gate(x)) * self.up(x))


class EncoderBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1     = RMSNorm(config.d_model)
        self.self_attn = MultiHeadAttention(config, causal=False)
        self.norm2     = RMSNorm(config.d_model)
        self.ff        = SwiGLU(config)
        self.drop      = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.drop(self.self_attn(self.norm1(x), attn_mask=attn_mask))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1      = RMSNorm(config.d_model)
        self.self_attn  = MultiHeadAttention(config, causal=True)
        self.norm2      = RMSNorm(config.d_model)
        self.cross_attn = CrossAttention(config)
        self.norm3      = RMSNorm(config.d_model)
        self.ff         = SwiGLU(config)
        self.drop       = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor,
        src_attn_mask: torch.Tensor | None = None,
        self_attn_mask: torch.Tensor | None = None,
        layer_cache: dict | None = None,
        start_pos: int = 0,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # layer_cache layout: {"self": {"k","v"}, "cross": {"k","v"}}
        self_cache  = layer_cache["self"]  if layer_cache is not None else None
        cross_cache = layer_cache["cross"] if layer_cache is not None else None

        x = x + self.drop(self.self_attn(
            self.norm1(x), attn_mask=self_attn_mask,
            kv_cache=self_cache, start_pos=start_pos,
        ))
        cross_out, attn_weights = self.cross_attn(
            self.norm2(x), encoder_out,
            attn_mask=src_attn_mask, kv_cache=cross_cache, need_weights=need_weights,
        )
        x = x + self.drop(cross_out)
        x = x + self.drop(self.ff(self.norm3(x)))
        return x, attn_weights
