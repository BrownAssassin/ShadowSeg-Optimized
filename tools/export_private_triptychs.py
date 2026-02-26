"""Export side-by-side private set diagnostics (image / GT / prediction)."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.config import Config
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rovershadow.runtime import ensure_runtime_env, install_mmcv_ops_shim_if_needed

ensure_runtime_env()
install_mmcv_ops_shim_if_needed()

import rovershadow.losses  # noqa: F401
from mmseg.apis import inference_model, init_model


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Export private-set triptych diagnostics.")
    parser.add_argument(
        "--img-dir",
        default="data/private/LunarShadowDataset/ShadowImages",
        help="Directory with private-set input images.",
    )
    parser.add_argument(
        "--mask-dir",
        default="data/private/LunarShadowDataset/ShadowMasks",
        help="Directory with private-set masks.",
    )
    parser.add_argument(
        "--cfg",
        default="configs/shadow_deeplabv3plus_r50.py",
        help="Model config path.",
    )
    parser.add_argument(
        "--ckpt",
        required=True,
        help="Checkpoint path.",
    )
    parser.add_argument(
        "--out-dir",
        default="work_dirs/private_triptychs",
        help="Output directory for image/GT/pred triptychs.",
    )
    parser.add_argument(
        "--out-overlay-dir",
        default="work_dirs/private_triptychs_overlay",
        help="Output directory for overlay triptychs.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Execution device. auto selects cuda when available.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Export only the first N samples after sorting.",
    )
    parser.add_argument(
        "--tta",
        default="none",
        choices=["none", "flip", "flip-ms"],
        help="Inference-time augmentation mode used for predictions.",
    )
    parser.add_argument(
        "--shadow-threshold",
        type=float,
        default=None,
        help="Optional probability threshold for class=shadow (binary mode).",
    )
    parser.add_argument(
        "--opacity",
        type=float,
        default=0.45,
        help="Overlay opacity in [0, 1].",
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


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """Map binary mask to black/white RGB for easy comparison."""
    out = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    out[mask > 0] = (255, 255, 255)
    return out


def blend_overlay(img_rgb: Image.Image, mask_bin: np.ndarray, color: tuple[int, int, int], alpha: float) -> Image.Image:
    """Blend a binary mask overlay on top of an RGB image."""
    base = np.array(img_rgb).astype(np.float32)
    overlay = base.copy()
    mask = mask_bin > 0
    overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * np.array(color, dtype=np.float32)
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB")


def add_title(img: Image.Image, title: str) -> Image.Image:
    """Add a compact title header above an image panel."""
    pad = 34
    canvas = Image.new("RGB", (img.width, img.height + pad), (30, 30, 30))
    canvas.paste(img, (0, pad))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), title, fill=(240, 240, 240))
    return canvas


def collect_image_paths(img_dir: Path) -> list[Path]:
    """Collect and sort image files by numeric id then filename."""
    paths = [p for p in img_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    return sorted(
        paths,
        key=lambda p: (extract_num(p.stem) if extract_num(p.stem) is not None else 10**9, p.name),
    )


def logits_from_image(model, image_bgr: np.ndarray) -> torch.Tensor:
    """Run single-pass inference and return logits in CxHxW format."""
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
        raise RuntimeError("No logits were produced during triptych export.")
    return total_logits / float(count)


def logits_to_prediction(logits: torch.Tensor, shadow_threshold: float | None) -> np.ndarray:
    """Convert logits to hard labels with optional class-1 threshold."""
    probs = torch.softmax(logits, dim=0)
    if shadow_threshold is None:
        pred = probs.argmax(dim=0)
    else:
        pred = (probs[1] >= shadow_threshold).to(torch.uint8)
    return pred.cpu().numpy().astype(np.uint8)


def resolve_mask_path(mask_dir: Path, image_path: Path) -> Path | None:
    """Resolve private mask path from image name using known conventions."""
    idx = extract_num(image_path.stem)
    candidates = []
    if idx is not None:
        candidates.extend([
            mask_dir / f"Mask-{idx}.png",
            mask_dir / f"Mask-{idx:02d}.png",
        ])
    candidates.extend([
        mask_dir / f"{image_path.stem}.png",
        mask_dir / f"{image_path.stem.replace('Image', 'Mask')}.png",
    ])
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def main() -> None:
    """Entry point for triptych export."""
    args = parse_args()
    if not (0.0 <= args.opacity <= 1.0):
        raise ValueError("--opacity must be in [0, 1].")
    if args.shadow_threshold is not None and not (0.0 <= args.shadow_threshold <= 1.0):
        raise ValueError("--shadow-threshold must be in [0, 1].")

    img_dir = Path(args.img_dir)
    mask_dir = Path(args.mask_dir)
    cfg_path = Path(args.cfg)
    ckpt_path = Path(args.ckpt)
    out_dir = Path(args.out_dir)
    out_overlay_dir = Path(args.out_overlay_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_overlay_dir.mkdir(parents=True, exist_ok=True)

    if not img_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")
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
            raise RuntimeError("No test pipeline found in config.")

    device = resolve_device(args.device)
    model = init_model(cfg, str(ckpt_path), device=device)

    image_paths = collect_image_paths(img_dir)
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]

    exported = 0
    for image_path in image_paths:
        idx = extract_num(image_path.stem)
        mask_path = resolve_mask_path(mask_dir, image_path)
        if mask_path is None:
            print(f"[WARN] Missing GT for {image_path.name}")
            continue

        image = Image.open(image_path).convert("RGB")
        gt = np.array(Image.open(mask_path).convert("L"))
        gt_bin = (gt > 0).astype(np.uint8)

        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        logits = aggregate_tta_logits(model, image_bgr, args.tta)
        pred = logits_to_prediction(logits, args.shadow_threshold)
        pred_bin = (pred == 1).astype(np.uint8)

        gt_rgb = Image.fromarray(mask_to_rgb(gt_bin), mode="RGB")
        pred_rgb = Image.fromarray(mask_to_rgb(pred_bin), mode="RGB")

        panel_image = add_title(image, f"Image {idx if idx is not None else image_path.stem}")
        panel_gt = add_title(gt_rgb, "GT Mask")
        panel_pred = add_title(pred_rgb, "Prediction")

        width, height = panel_image.size
        canvas = Image.new("RGB", (width * 3, height), (0, 0, 0))
        canvas.paste(panel_image, (0, 0))
        canvas.paste(panel_gt, (width, 0))
        canvas.paste(panel_pred, (width * 2, 0))

        serial = idx if idx is not None else exported + 1
        out_path = out_dir / f"triptych_{serial:02d}.png"
        canvas.save(out_path)
        exported += 1
        print(f"[OK] {out_path}")

        gt_overlay = blend_overlay(image, gt_bin, color=(0, 255, 0), alpha=args.opacity)
        pred_overlay = blend_overlay(image, pred_bin, color=(255, 0, 0), alpha=args.opacity)

        panel_overlay_gt = add_title(gt_overlay, "GT Overlay (green=shadow)")
        panel_overlay_pred = add_title(pred_overlay, "Pred Overlay (red=shadow)")

        overlay_canvas = Image.new("RGB", (width * 3, height), (0, 0, 0))
        overlay_canvas.paste(panel_image, (0, 0))
        overlay_canvas.paste(panel_overlay_gt, (width, 0))
        overlay_canvas.paste(panel_overlay_pred, (width * 2, 0))

        out_overlay_path = out_overlay_dir / f"triptych_overlay_{serial:02d}.png"
        overlay_canvas.save(out_overlay_path)
        print(f"[OK] {out_overlay_path}")

    print(f"[DONE] Exported {exported} triptychs to {out_dir}")
    print(f"[DONE] Exported {exported} overlay triptychs to {out_overlay_dir}")


if __name__ == "__main__":
    main()
