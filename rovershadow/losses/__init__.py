"""Custom loss modules registered for RoverShadow experiments."""

from .safe_cross_entropy_loss import SafeCrossEntropyLoss
from .shadow_false_positive_loss import ShadowFalsePositiveLoss

__all__ = ["SafeCrossEntropyLoss", "ShadowFalsePositiveLoss"]
