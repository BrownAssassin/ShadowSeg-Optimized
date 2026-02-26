"""Utilities for running mmseg with mmcv-lite when compiled ops are missing."""

from __future__ import annotations

import os
import sys
import types

import torch
import torch.nn.functional as F

_SHIM_WARNING = (
    "[WARN] mmcv compiled ops are unavailable. Falling back to a lightweight "
    "mmcv.ops shim for point_sample/sigmoid_focal_loss."
)


def ensure_runtime_env() -> None:
    """Set environment defaults required by current MMEngine checkpoint loading."""
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")


def _looks_like_mmcv_ext_failure(exc: Exception) -> bool:
    """Return True if the exception indicates missing compiled mmcv ops."""
    msg = str(exc)
    return "mmcv._ext" in msg or "_ext" in msg or "mmcv.ops" in msg


def _build_mmcv_ops_shim() -> types.ModuleType:
    """Create a minimal replacement for ``mmcv.ops`` used by this project."""
    ops = types.ModuleType("mmcv.ops")
    ops.__ROVERSHADOW_SHIM__ = True

    def point_sample(input, points, align_corners=False, **kwargs):
        if points.dim() == 3:
            points = points.unsqueeze(2)
        grid = points * 2 - 1
        sampled = F.grid_sample(
            input,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=align_corners,
        )
        return sampled.squeeze(3)

    def sigmoid_focal_loss(
        pred,
        target,
        gamma=2.0,
        alpha=0.25,
        weight=None,
        reduction="none",
    ):
        prob = pred.sigmoid()
        ce = F.binary_cross_entropy_with_logits(
            pred,
            target.type_as(pred),
            reduction="none",
        )
        p_t = prob * target + (1 - prob) * (1 - target)
        alpha_t = alpha * target + (1 - alpha) * (1 - target)
        loss = ce * alpha_t * ((1 - p_t) ** gamma)
        if weight is not None:
            loss = loss * weight
        if reduction == "sum":
            return loss.sum()
        if reduction == "mean":
            return loss.mean()
        return loss

    class _UnavailableOp:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "This operator requires compiled mmcv ops, which are unavailable."
            )

    ops.point_sample = point_sample
    ops.sigmoid_focal_loss = sigmoid_focal_loss
    ops.CrissCrossAttention = _UnavailableOp
    ops.PSAMask = _UnavailableOp
    ops.DeformConv2d = _UnavailableOp
    ops.ModulatedDeformConv2d = _UnavailableOp
    return ops


def install_mmcv_ops_shim_if_needed(verbose: bool = True) -> bool:
    """Install a shim for ``mmcv.ops`` if compiled ops are unavailable.

    Returns:
        bool: True if the shim is active after this call, otherwise False.
    """
    existing = sys.modules.get("mmcv.ops")
    if existing is not None and getattr(existing, "__ROVERSHADOW_SHIM__", False):
        return True

    try:
        import mmcv.ops  # noqa: F401
        return False
    except Exception as exc:
        if not _looks_like_mmcv_ext_failure(exc):
            raise

    sys.modules.pop("mmcv.ops", None)
    sys.modules["mmcv.ops"] = _build_mmcv_ops_shim()
    if verbose:
        print(_SHIM_WARNING)
    return True
