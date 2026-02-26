"""Utilities for external-only pseudo-label generation and dataset integration."""

from .external_infer import ExternalShadowInferencer, calibrate_shadow_threshold
from .external_model_registry import resolve_external_checkpoint
from .fallback_external_trainer import train_fallback_external_model
from .render_integrator import integrate_render_dataset, plan_render_integration_dry_run

__all__ = [
    "ExternalShadowInferencer",
    "calibrate_shadow_threshold",
    "resolve_external_checkpoint",
    "train_fallback_external_model",
    "integrate_render_dataset",
    "plan_render_integration_dry_run",
]
