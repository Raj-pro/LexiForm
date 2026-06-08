"""
Exponential moving average of model parameters.

Keeps a CPU- or GPU-resident shadow copy of every floating-point parameter and
updates it after each optimizer step. At eval / checkpoint time the EMA weights
are swapped into the live model with `apply_to(model)` and restored with
`restore(model)`. This is the standard trick that buys a small but reliable
val-loss reduction on small datasets without extra compute at inference.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        self._backup: dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if p.requires_grad and p.dtype.is_floating_point:
                self.shadow[name] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for name, p in model.named_parameters():
            if name in self.shadow:
                # in-place: shadow = d*shadow + (1-d)*p
                self.shadow[name].mul_(d).add_(p.detach(), alpha=1.0 - d)

    @torch.no_grad()
    def apply_to(self, model: nn.Module) -> None:
        """Swap live weights with EMA weights; remember live weights for restore()."""
        assert not self._backup, "apply_to() called twice without restore()"
        for name, p in model.named_parameters():
            if name in self.shadow:
                self._backup[name] = p.detach().clone()
                p.data.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        for name, p in model.named_parameters():
            if name in self._backup:
                p.data.copy_(self._backup[name])
        self._backup.clear()

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        for k, v in state.items():
            if k in self.shadow:
                self.shadow[k].copy_(v)
