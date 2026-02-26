"""Evaluate RoverShadow checkpoints on public validation or private holdout."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.config import Config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rovershadow.runtime import ensure_runtime_env, install_mmcv_ops_shim_if_needed

ensure_runtime_env()
install_mmcv_ops_shim_if_needed()

import rovershadow.losses  # noqa: F401
from mmseg.apis import inference_model, init_model

CLASS_NAMES = ("background", "shadow")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a RoverShadow model.")
    parser.add_argument(
        "--config",
        default="configs/shadow_deeplabv3plus_r50.py",
        help="Model config path.",
    )
    parser.add_argument(
        "--ckpt",
        required=True,
        help="Checkpoint path.",
    )
    parser.add_argument(
        "--split",
        default="public-val",
        choices=["public-val", "private"],
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Execution device. auto selects cuda when available.",
    )
    parser.add_argument(
        "--tta",
        default="none",
        choices=["none", "flip", "flip-ms"],
        help="Inference-time augmentation mode.",
    )
    parser.add_argument(
        "--shadow-threshold",
        type=float,
        default=None,
        help="Optional probability threshold for class=shadow (binary mode).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Evaluate only the first N images after sorting.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=50,
        help="Print progress every N images. Use 0 to disable per-image progress logs.",
    )
    parser.add_argument(
        "--save-json",
        default=None,
        help="Optional output path for metrics JSON.",
    )
    parser.add_argument(
        "--public-img-dir",
        default="data/public/Rover_Shadow_Public_Dataset/ShadowImages/val",
        help="Public validation image directory.",
    )
    parser.add_argument(
        "--public-mask-dir",
        default="data/public/Rover_Shadow_Public_Dataset/ShadowMasks/val",
        help="Public validation mask directory.",
    )
    parser.add_argument(
        "--private-img-dir",
        default="data/private/LunarShadowDataset/ShadowImages",
        help="Private holdout image directory.",
    )
    parser.add_argument(
        "--private-mask-dir",
        default="data/private/LunarShadowDataset/ShadowMasks",
        help="Private holdout mask directory.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    """Resolve explicit/auto device request."""
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no GPU is visible to PyTorch.")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def extract_num(name: str) -> int | None:
    """Extract first integer token for stable numeric sorting."""
    match = re.search(r"(\d+)", name)
    return int(match.group(1)) if match else None


def _sorted_files(paths: Iterable[Path]) -> list[Path]:
    """Sort files by first integer token, then filename."""
    return sorted(
        paths,
        key=lambda p: (extract_num(p.stem) if extract_num(p.stem) is not None else 10**9, p.name),
    )


def collect_public_pairs(image_dir: Path, mask_dir: Path) -> list[tuple[Path, Path]]:
    """Build image/mask pairs for public validation split."""
    image_paths = _sorted_files(image_dir.glob("*"))
    pairs: list[tuple[Path, Path]] = []
    for image_path in image_paths:
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        mask_path = mask_dir / f"{image_path.stem}.png"
        if mask_path.is_file():
            pairs.append((image_path, mask_path))
        else:
            print(f"[WARN] Missing mask for {image_path.name}: expected {mask_path.name}")
    return pairs


def collect_private_pairs(image_dir: Path, mask_dir: Path) -> list[tuple[Path, Path]]:
    """Build image/mask pairs for private holdout split."""
    image_paths = _sorted_files(image_dir.glob("*"))
    pairs: list[tuple[Path, Path]] = []
    for image_path in image_paths:
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        idx = extract_num(image_path.stem)
        candidates = []
        if idx is not None:
            candidates.extend(
                [
                    mask_dir / f"Mask-{idx}.png",
                    mask_dir / f"Mask-{idx:02d}.png",
                ]
            )
        candidates.extend(
            [
                mask_dir / f"{image_path.stem}.png",
                mask_dir / f"{image_path.stem.replace('Image', 'Mask')}.png",
            ]
        )
        mask_path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if mask_path is None:
            print(f"[WARN] Missing mask for {image_path.name}")
            continue
        pairs.append((image_path, mask_path))
    return pairs


def load_mask(mask_path: Path) -> np.ndarray:
    """Load mask and normalize to {0,1}."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")
    return (mask > 0).astype(np.uint8)


def logits_from_image(model, image_bgr: np.ndarray) -> torch.Tensor:
    """Run single-pass inference and return raw logits in CxHxW format."""
    result = inference_model(model, image_bgr)
    logits = result.seg_logits.data
    if logits.dim() == 4:
        logits = logits[0]
    return logits.detach().float().cpu()


def aggregate_tta_logits(model, image_bgr: np.ndarray, tta: str) -> torch.Tensor:
    """Aggregate logits across TTA views and resize back to original size."""
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

    total_logits: torch.Tensor | None = None
    count = 0

    for scale in scales:
        if scale == 1.0:
            scaled = image_bgr
        else:
            scaled = cv2.resize(
                image_bgr,
                (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
                interpolation=cv2.INTER_LINEAR,
            )

        logits = logits_from_image(model, scaled)
        logits = F.interpolate(
            logits.unsqueeze(0),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        total_logits = logits if total_logits is None else total_logits + logits
        count += 1

        if use_flip:
            flipped = cv2.flip(scaled, 1)
            logits_flip = logits_from_image(model, flipped)
            logits_flip = torch.flip(logits_flip, dims=[2])
            logits_flip = F.interpolate(
                logits_flip.unsqueeze(0),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            total_logits = total_logits + logits_flip
            count += 1

    if total_logits is None or count == 0:
        raise RuntimeError("No logits were produced during evaluation.")
    return total_logits / float(count)


def logits_to_prediction(logits: torch.Tensor, shadow_threshold: float | None) -> np.ndarray:
    """Convert logits to hard labels with optional class-1 threshold."""
    probs = torch.softmax(logits, dim=0)
    if shadow_threshold is None:
        pred = probs.argmax(dim=0)
    else:
        pred = (probs[1] >= shadow_threshold).to(torch.uint8)
    return pred.cpu().numpy().astype(np.uint8)


def safe_divide(numerator: float, denominator: float) -> float:
    """Division with zero-denominator guard."""
    if denominator <= 0:
        return float("nan")
    return numerator / denominator


def harmonic_mean(x: float, y: float) -> float:
    """Harmonic mean for two positive scalars."""
    if not np.isfinite(x) or not np.isfinite(y) or x <= 0 or y <= 0:
        return 0.0
    return (2.0 * x * y) / (x + y)


def evaluate_pairs(
    model,
    pairs: list[tuple[Path, Path]],
    tta: str,
    shadow_threshold: float | None,
    progress_interval: int,
) -> dict:
    """Run evaluation over pairs and return aggregate metrics."""
    intersections = np.zeros(2, dtype=np.float64)
    unions = np.zeros(2, dtype=np.float64)
    targets = np.zeros(2, dtype=np.float64)
    total_correct = 0.0
    total_labeled = 0.0

    for idx, (image_path, mask_path) in enumerate(pairs, start=1):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        gt = load_mask(mask_path)
        logits = aggregate_tta_logits(model, image, tta)
        pred = logits_to_prediction(logits, shadow_threshold)

        valid = np.ones_like(gt, dtype=bool)
        total_correct += float((pred[valid] == gt[valid]).sum())
        total_labeled += float(valid.sum())

        for class_id in range(2):
            pred_c = (pred == class_id) & valid
            gt_c = (gt == class_id) & valid
            intersections[class_id] += float(np.logical_and(pred_c, gt_c).sum())
            unions[class_id] += float(np.logical_or(pred_c, gt_c).sum())
            targets[class_id] += float(gt_c.sum())

        should_log = (
            progress_interval > 0
            and (idx % progress_interval == 0 or idx == len(pairs))
        )
        if should_log:
            print(f"[INFO] Evaluated {idx}/{len(pairs)}: {image_path.name}")

    iou = [safe_divide(intersections[i], unions[i]) for i in range(2)]
    acc = [safe_divide(intersections[i], targets[i]) for i in range(2)]
    miou = float(np.nanmean(np.array(iou, dtype=np.float64)))
    macc = float(np.nanmean(np.array(acc, dtype=np.float64)))
    aacc = safe_divide(total_correct, total_labeled)
    public_proxy_score = 0.6 * miou + 0.4 * harmonic_mean(iou[0], iou[1])

    return dict(
        num_images=len(pairs),
        IoU_background=iou[0],
        IoU_shadow=iou[1],
        mIoU=miou,
        Acc_background=acc[0],
        Acc_shadow=acc[1],
        mAcc=macc,
        aAcc=aacc,
        public_proxy_score=public_proxy_score,
    )


def main() -> None:
    """Entry point for evaluation."""
    args = parse_args()
    device = resolve_device(args.device)

    if args.shadow_threshold is not None and not (0.0 <= args.shadow_threshold <= 1.0):
        raise ValueError("--shadow-threshold must be within [0, 1].")

    cfg_path = Path(args.config)
    ckpt_path = Path(args.ckpt)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    cfg = Config.fromfile(str(cfg_path))
    if not hasattr(cfg, "test_pipeline"):
        if "test_dataloader" in cfg and "dataset" in cfg.test_dataloader and "pipeline" in cfg.test_dataloader.dataset:
            cfg.test_pipeline = cfg.test_dataloader.dataset.pipeline
        elif "val_dataloader" in cfg and "dataset" in cfg.val_dataloader and "pipeline" in cfg.val_dataloader.dataset:
            cfg.test_pipeline = cfg.val_dataloader.dataset.pipeline
        else:
            raise RuntimeError("Config is missing test_pipeline and dataloader pipeline.")

    model = init_model(cfg, str(ckpt_path), device=device)

    if args.split == "public-val":
        pairs = collect_public_pairs(Path(args.public_img_dir), Path(args.public_mask_dir))
    else:
        pairs = collect_private_pairs(Path(args.private_img_dir), Path(args.private_mask_dir))

    if args.max_images is not None:
        pairs = pairs[: args.max_images]
    if not pairs:
        raise RuntimeError("No valid image/mask pairs found for evaluation.")

    metrics = evaluate_pairs(
        model=model,
        pairs=pairs,
        tta=args.tta,
        shadow_threshold=args.shadow_threshold,
        progress_interval=args.progress_interval,
    )
    metrics["split"] = args.split
    metrics["tta"] = args.tta
    metrics["shadow_threshold"] = args.shadow_threshold
    metrics["config"] = str(cfg_path)
    metrics["checkpoint"] = str(ckpt_path)

    print("")
    print("[RESULTS]")
    print(f"split: {metrics['split']}")
    print(f"num_images: {metrics['num_images']}")
    print(f"IoU_background: {metrics['IoU_background']:.4f}")
    print(f"IoU_shadow: {metrics['IoU_shadow']:.4f}")
    print(f"mIoU: {metrics['mIoU']:.4f}")
    print(f"Acc_background: {metrics['Acc_background']:.4f}")
    print(f"Acc_shadow: {metrics['Acc_shadow']:.4f}")
    print(f"mAcc: {metrics['mAcc']:.4f}")
    print(f"aAcc: {metrics['aAcc']:.4f}")
    print(f"public_proxy_score: {metrics['public_proxy_score']:.4f}")

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"[OK] Saved metrics JSON: {out_path}")


if __name__ == "__main__":
    main()
