import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingLoss(nn.Module):
    """
    Label-smoothed cross-entropy with explicit input mode.

    Caller must pass `input_is_log_probs=True` when the model returns log-probs
    (copy/pointer-generator mode) and `False` for raw logits. We never auto-detect
    because the previous heuristic both forced a GPU->CPU sync (.item()) and
    misfired on early-training logits that happened to be all <= 0.
    """
    def __init__(self, vocab_size: int, smoothing: float = 0.1, ignore_index: int = 0):
        super().__init__()
        self.smoothing    = smoothing
        self.vocab_size   = vocab_size
        self.ignore_index = ignore_index

    def forward(
        self,
        scores: torch.Tensor,   # (B, T, V)
        targets: torch.Tensor,  # (B, T)
        input_is_log_probs: bool,
    ) -> torch.Tensor:
        B, T, V = scores.shape
        x       = scores.view(-1, V)
        targets = targets.view(-1)

        log_probs = x if input_is_log_probs else F.log_softmax(x, dim=-1)

        nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)

        # Uniform smoothing target excludes ignore_index (pad). The previous
        # version put smoothing mass on every vocab id including <pad>, pushing
        # the model toward emitting padding and inflating reported NLL.
        smooth_target_size = self.vocab_size - 1
        log_probs_sum      = log_probs.sum(dim=-1) - log_probs[:, self.ignore_index]
        smooth_loss        = -log_probs_sum / smooth_target_size

        loss = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        mask = targets.ne(self.ignore_index)
        return loss[mask].mean()
