"""Cross-entropy loss variant that safely handles ignore labels with class weights."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.models.losses.utils import get_class_weight, weight_reduce_loss
from mmseg.registry import MODELS


@MODELS.register_module()
class SafeCrossEntropyLoss(nn.Module):
    """CrossEntropyLoss drop-in replacement with safe ignore handling.

    mmseg's built-in CrossEntropyLoss can index out of bounds when both:
    - `class_weight` is set
    - labels include `ignore_index` (for example from padded masks)
    """

    def __init__(
        self,
        use_sigmoid: bool = False,
        use_mask: bool = False,
        reduction: str = "mean",
        class_weight=None,
        loss_weight: float = 1.0,
        loss_name: str = "loss_ce",
        avg_non_ignore: bool = False,
    ) -> None:
        super().__init__()
        if use_sigmoid or use_mask:
            raise ValueError("SafeCrossEntropyLoss currently supports only softmax CE.")
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be one of: none, mean, sum")
        self.reduction = reduction
        self.class_weight = get_class_weight(class_weight)
        self.loss_weight = loss_weight
        self.avg_non_ignore = avg_non_ignore
        self._loss_name = loss_name

    def forward(
        self,
        cls_score: torch.Tensor,
        label: torch.Tensor,
        weight: torch.Tensor | None = None,
        avg_factor: float | None = None,
        reduction_override: str | None = None,
        ignore_index: int = 255,
        **kwargs,
    ) -> torch.Tensor:
        """Compute weighted cross entropy with safe ignore-index masking."""
        reduction = reduction_override if reduction_override else self.reduction
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be one of: none, mean, sum")

        class_weight = None
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)

        loss = F.cross_entropy(
            cls_score,
            label,
            weight=class_weight,
            reduction="none",
            ignore_index=ignore_index,
        )

        if avg_factor is None and reduction == "mean":
            valid_mask = label != ignore_index
            if class_weight is None:
                if self.avg_non_ignore:
                    avg_factor = valid_mask.sum().item()
                else:
                    avg_factor = label.numel()
            else:
                safe_label = label.clone()
                safe_label[~valid_mask] = 0
                label_weights = class_weight[safe_label]
                if self.avg_non_ignore:
                    label_weights = label_weights * valid_mask
                avg_factor = label_weights.sum().item()

        if weight is not None:
            weight = weight.float()

        loss = weight_reduce_loss(
            loss,
            weight=weight,
            reduction=reduction,
            avg_factor=avg_factor,
        )
        return self.loss_weight * loss

    @property
    def loss_name(self) -> str:
        """Name used by mmseg to aggregate this loss."""
        return self._loss_name
