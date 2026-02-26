"""Custom losses for reducing shadow false positives on background regions."""

from __future__ import annotations

import torch
import torch.nn as nn

from mmseg.registry import MODELS


@MODELS.register_module()
class ShadowFalsePositiveLoss(nn.Module):
    """Penalize high shadow probability on non-shadow pixels.

    This complements region overlap losses by explicitly suppressing
    background pixels that are predicted as shadow with high confidence.
    """

    def __init__(
        self,
        shadow_class: int = 1,
        ignore_index: int = 255,
        gamma: float = 2.0,
        reduction: str = "mean",
        loss_weight: float = 1.0,
        loss_name: str = "loss_shadow_fp",
    ) -> None:
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be one of: none, mean, sum")
        self.shadow_class = shadow_class
        self.ignore_index = ignore_index
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    def forward(
        self,
        cls_score: torch.Tensor,
        label: torch.Tensor,
        weight: torch.Tensor | None = None,
        avg_factor: float | None = None,
        reduction_override: str | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute the false-positive penalty from segmentation logits."""
        reduction = reduction_override if reduction_override else self.reduction
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be one of: none, mean, sum")
        if cls_score.dim() != 4:
            raise ValueError("cls_score must have shape (N, C, H, W)")
        if label.dim() != 3:
            raise ValueError("label must have shape (N, H, W)")
        if self.shadow_class >= cls_score.size(1):
            raise ValueError(
                f"shadow_class={self.shadow_class} is out of range for "
                f"{cls_score.size(1)} classes"
            )

        probs = cls_score.softmax(dim=1)
        shadow_prob = probs[:, self.shadow_class, :, :]

        valid_mask = label != self.ignore_index
        bg_mask = (label != self.shadow_class) & valid_mask

        loss_map = shadow_prob.new_zeros(label.shape, dtype=shadow_prob.dtype)
        if bg_mask.any():
            penalty = shadow_prob[bg_mask].pow(self.gamma)
            if weight is not None and weight.shape == label.shape:
                penalty = penalty * weight[bg_mask]
            loss_map[bg_mask] = penalty

        if reduction == "none":
            return self.loss_weight * loss_map

        fp_values = loss_map[bg_mask]
        if fp_values.numel() == 0:
            zero = cls_score.sum() * 0.0
            return self.loss_weight * zero

        if reduction == "sum":
            loss = fp_values.sum()
        else:
            denom = avg_factor if avg_factor is not None else fp_values.numel()
            loss = fp_values.sum() / max(float(denom), 1.0)

        return self.loss_weight * loss

    @property
    def loss_name(self) -> str:
        """Name used by mmseg to aggregate losses."""
        return self._loss_name
