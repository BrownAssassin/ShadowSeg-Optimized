"""Verify public dataset integrity after render pseudo-label integration."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    """Parse verification CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Validate naming, pairing, masks, and split summaries."
    )
    parser.add_argument(
        "--public-root",
        default="data/public/Rover_Shadow_Public_Dataset",
        help="Public dataset root directory.",
    )
    parser.add_argument(
        "--workspace",
        default="data/_staging_render",
        help="Workspace with integration manifests.",
    )
    parser.add_argument(
        "--sample-masks",
        type=int,
        default=500,
        help="Number of random masks to sample for value checks.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for mask sampling.",
    )
    parser.add_argument(
        "--save-json",
        default=None,
        help="Optional path to save verification report JSON.",
    )
    return parser.parse_args()


def _collect_stems(folder: Path, image_exts: set[str]) -> set[str]:
    stems: set[str] = set()
    if not folder.is_dir():
        return stems
    for path in folder.iterdir():
        if path.is_file() and path.suffix.lower() in image_exts:
            stems.add(path.stem)
    return stems


def _collect_mask_paths(public_root: Path) -> list[Path]:
    paths: list[Path] = []
    for split in ("train", "val", "quarantine_render"):
        folder = public_root / "ShadowMasks" / split
        if folder.is_dir():
            paths.extend([p for p in folder.iterdir() if p.suffix.lower() == ".png"])
    return paths


def _mask_value_check(mask_paths: list[Path], sample_count: int, seed: int) -> dict:
    rng = random.Random(seed)
    sample = mask_paths.copy()
    rng.shuffle(sample)
    sample = sample[: min(sample_count, len(sample))]

    invalid: list[str] = []
    for path in sample:
        arr = np.array(Image.open(path).convert("L"))
        unique = np.unique(arr)
        if not set(unique.tolist()).issubset({0, 1}):
            invalid.append(path.as_posix())
    return {
        "sample_size": len(sample),
        "invalid_count": len(invalid),
        "invalid_examples": invalid[:20],
    }


def main() -> None:
    """Entry point for dataset integrity verification."""
    args = parse_args()
    public_root = Path(args.public_root)
    workspace = Path(args.workspace)
    if not public_root.is_dir():
        raise FileNotFoundError(f"Public root not found: {public_root}")

    images_root = public_root / "ShadowImages"
    masks_root = public_root / "ShadowMasks"

    report: dict = {"errors": []}
    required_dirs = [
        images_root / "train",
        images_root / "val",
        masks_root / "train",
        masks_root / "val",
    ]
    for req in required_dirs:
        if not req.is_dir():
            report["errors"].append(f"Missing required folder: {req}")

    train_img_stems = _collect_stems(images_root / "train", {".jpg", ".jpeg", ".png"})
    val_img_stems = _collect_stems(images_root / "val", {".jpg", ".jpeg", ".png"})
    q_img_stems = _collect_stems(images_root / "quarantine_render", {".jpg", ".jpeg", ".png"})
    train_mask_stems = _collect_stems(masks_root / "train", {".png"})
    val_mask_stems = _collect_stems(masks_root / "val", {".png"})
    q_mask_stems = _collect_stems(masks_root / "quarantine_render", {".png"})

    report["counts"] = {
        "train_images": len(train_img_stems),
        "train_masks": len(train_mask_stems),
        "val_images": len(val_img_stems),
        "val_masks": len(val_mask_stems),
        "quarantine_images": len(q_img_stems),
        "quarantine_masks": len(q_mask_stems),
    }

    pair_gaps = {
        "train_images_without_masks": sorted(train_img_stems - train_mask_stems)[:20],
        "train_masks_without_images": sorted(train_mask_stems - train_img_stems)[:20],
        "val_images_without_masks": sorted(val_img_stems - val_mask_stems)[:20],
        "val_masks_without_images": sorted(val_mask_stems - val_img_stems)[:20],
        "quarantine_images_without_masks": sorted(q_img_stems - q_mask_stems)[:20],
        "quarantine_masks_without_images": sorted(q_mask_stems - q_img_stems)[:20],
    }
    report["pair_gaps"] = pair_gaps

    split_overlap = {
        "train_val_overlap": sorted((train_img_stems & val_img_stems))[:20],
        "train_quarantine_overlap": sorted((train_img_stems & q_img_stems))[:20],
        "val_quarantine_overlap": sorted((val_img_stems & q_img_stems))[:20],
    }
    report["split_overlap"] = split_overlap

    all_stems = train_img_stems | val_img_stems | q_img_stems
    non_lssd = sorted([stem for stem in all_stems if not stem.lower().startswith("lssd")])[:50]
    report["naming"] = {
        "total_stems_checked": len(all_stems),
        "non_lssd_count": len([stem for stem in all_stems if not stem.lower().startswith("lssd")]),
        "non_lssd_examples": non_lssd,
    }

    mask_paths = _collect_mask_paths(public_root)
    report["mask_value_check"] = _mask_value_check(mask_paths, sample_count=args.sample_masks, seed=args.seed)

    manifest_path = workspace / "manifests" / "split_summary.json"
    if manifest_path.is_file():
        try:
            report["latest_split_summary"] = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            report["errors"].append(f"Failed to parse split_summary.json: {exc}")
    else:
        report["latest_split_summary"] = None

    if any(pair_gaps[key] for key in pair_gaps):
        report["errors"].append("Detected image/mask pairing gaps.")
    if any(split_overlap[key] for key in split_overlap):
        report["errors"].append("Detected duplicate stems across splits.")
    if report["mask_value_check"]["invalid_count"] > 0:
        report["errors"].append("Detected non-binary mask values outside {0,1}.")
    if report["naming"]["non_lssd_count"] > 0:
        report["errors"].append("Detected file stems not following lssd naming convention.")

    report["ok"] = len(report["errors"]) == 0

    print(json.dumps(report, indent=2))
    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[OK] Saved verification report: {out_path}")

    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
