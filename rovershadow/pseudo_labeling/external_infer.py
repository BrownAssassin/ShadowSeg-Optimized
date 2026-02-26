"""Inference helpers for external-only pseudo-label generation."""

from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.config import Config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rovershadow.runtime import ensure_runtime_env, install_mmcv_ops_shim_if_needed

ensure_runtime_env()
install_mmcv_ops_shim_if_needed()

from mmseg.apis import inference_model, init_model


def extract_num(name: str) -> int | None:
    """Extract first integer from a string for stable sorting."""
    match = re.search(r"(\d+)", name)
    return int(match.group(1)) if match else None


def sorted_paths(paths: Iterable[Path]) -> list[Path]:
    """Sort paths by numeric token and then filename."""
    return sorted(
        paths,
        key=lambda p: (extract_num(p.stem) if extract_num(p.stem) is not None else 10**9, p.name),
    )


def collect_public_pairs(public_root: Path) -> list[tuple[Path, Path]]:
    """Collect public validation image/mask pairs."""
    image_root = public_root / "ShadowImages" / "val"
    mask_root = public_root / "ShadowMasks" / "val"
    pairs: list[tuple[Path, Path]] = []
    for image_path in sorted_paths(image_root.glob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        mask_path = mask_root / f"{image_path.stem}.png"
        if mask_path.is_file():
            pairs.append((image_path, mask_path))
    return pairs


def _logits_from_image(model, image_bgr: np.ndarray) -> torch.Tensor:
    """Run single inference and return logits in CxHxW."""
    result = inference_model(model, image_bgr)
    logits = result.seg_logits.data
    if logits.dim() == 4:
        logits = logits[0]
    return logits.detach().float().cpu()


def _logits_to_shadow_prob(logits: torch.Tensor) -> torch.Tensor:
    """Convert logits to shadow probability map."""
    if logits.dim() != 3:
        raise ValueError("Expected logits in CxHxW format.")
    channels = logits.shape[0]
    if channels == 1:
        return torch.sigmoid(logits[0])
    if channels >= 2:
        probs = torch.softmax(logits, dim=0)
        return probs[1]
    raise ValueError(f"Unexpected number of logits channels: {channels}")


@dataclass
class ThresholdCalibrationResult:
    """Results from F0.5 threshold sweep."""

    best_threshold: float
    best_f05: float
    rows: list[dict]


class ExternalShadowInferencer:
    """Wrapper for external shadow model inference with TTA support."""

    def __init__(
        self,
        config_path: Path,
        checkpoint_path: Path,
        device: str,
    ) -> None:
        if not config_path.is_file():
            raise FileNotFoundError(f"Config not found: {config_path}")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        cfg = Config.fromfile(str(config_path))
        if not hasattr(cfg, "test_pipeline"):
            if "test_dataloader" in cfg and "dataset" in cfg.test_dataloader and "pipeline" in cfg.test_dataloader.dataset:
                cfg.test_pipeline = cfg.test_dataloader.dataset.pipeline
            elif "val_dataloader" in cfg and "dataset" in cfg.val_dataloader and "pipeline" in cfg.val_dataloader.dataset:
                cfg.test_pipeline = cfg.val_dataloader.dataset.pipeline
            else:
                raise RuntimeError("Config does not provide a test/val pipeline.")
        self.model = init_model(cfg, str(checkpoint_path), device=device)

    def predict_shadow_probability(self, image_bgr: np.ndarray, tta: str = "flip-ms") -> np.ndarray:
        """Predict shadow probability map in [0,1], shape HxW."""
        if tta not in {"none", "flip", "flip-ms"}:
            raise ValueError("tta must be one of: none, flip, flip-ms")
        height, width = image_bgr.shape[:2]
        if tta == "flip-ms":
            scales = [0.75, 1.0, 1.25]
            use_flip = True
        elif tta == "flip":
            scales = [1.0]
            use_flip = True
        else:
            scales = [1.0]
            use_flip = False

        total_prob: torch.Tensor | None = None
        count = 0

        for scale in scales:
            if math.isclose(scale, 1.0):
                scaled = image_bgr
            else:
                scaled = cv2.resize(
                    image_bgr,
                    (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
                    interpolation=cv2.INTER_LINEAR,
                )

            logits = _logits_from_image(self.model, scaled)
            shadow_prob = _logits_to_shadow_prob(logits)
            shadow_prob = F.interpolate(
                shadow_prob.unsqueeze(0).unsqueeze(0),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)
            total_prob = shadow_prob if total_prob is None else total_prob + shadow_prob
            count += 1

            if use_flip:
                flipped = cv2.flip(scaled, 1)
                logits_flip = _logits_from_image(self.model, flipped)
                prob_flip = _logits_to_shadow_prob(logits_flip)
                prob_flip = torch.flip(prob_flip, dims=[1])
                prob_flip = F.interpolate(
                    prob_flip.unsqueeze(0).unsqueeze(0),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).squeeze(0)
                total_prob = total_prob + prob_flip
                count += 1

        if total_prob is None or count == 0:
            raise RuntimeError("No probabilities produced by external inferencer.")
        avg_prob = total_prob / float(count)
        return avg_prob.cpu().numpy().astype(np.float32)


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _binary_confusion(pred_bin: np.ndarray, gt_bin: np.ndarray) -> tuple[int, int, int]:
    """Return TP/FP/FN for class=shadow."""
    tp = int(np.logical_and(pred_bin == 1, gt_bin == 1).sum())
    fp = int(np.logical_and(pred_bin == 1, gt_bin == 0).sum())
    fn = int(np.logical_and(pred_bin == 0, gt_bin == 1).sum())
    return tp, fp, fn


def calibrate_shadow_threshold(
    inferencer: ExternalShadowInferencer,
    public_pairs: list[tuple[Path, Path]],
    thresholds: list[float],
    tta: str = "flip-ms",
    progress_interval: int = 50,
) -> ThresholdCalibrationResult:
    """Calibrate threshold on public val by maximizing F0.5 for shadow."""
    if not public_pairs:
        raise ValueError("No public validation pairs available for calibration.")
    agg = {thr: dict(tp=0, fp=0, fn=0) for thr in thresholds}

    for idx, (image_path, mask_path) in enumerate(public_pairs, start=1):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if gt is None:
            raise FileNotFoundError(f"Could not read mask: {mask_path}")
        gt_bin = (gt > 0).astype(np.uint8)
        prob = inferencer.predict_shadow_probability(image, tta=tta)
        for thr in thresholds:
            pred_bin = (prob >= thr).astype(np.uint8)
            tp, fp, fn = _binary_confusion(pred_bin, gt_bin)
            agg[thr]["tp"] += tp
            agg[thr]["fp"] += fp
            agg[thr]["fn"] += fn
        if progress_interval > 0 and (idx % progress_interval == 0 or idx == len(public_pairs)):
            print(f"[INFO] Threshold calibration progress {idx}/{len(public_pairs)}")

    rows: list[dict] = []
    best_thr = thresholds[0]
    best_f05 = -1.0
    beta_sq = 0.5 * 0.5
    for thr in thresholds:
        tp = agg[thr]["tp"]
        fp = agg[thr]["fp"]
        fn = agg[thr]["fn"]
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        numerator = (1 + beta_sq) * precision * recall
        denominator = beta_sq * precision + recall
        f05 = _safe_divide(numerator, denominator)
        row = {
            "threshold": thr,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision_shadow": precision,
            "recall_shadow": recall,
            "f0_5_shadow": f05,
        }
        rows.append(row)
        if f05 > best_f05:
            best_f05 = f05
            best_thr = thr

    return ThresholdCalibrationResult(best_threshold=best_thr, best_f05=best_f05, rows=rows)
