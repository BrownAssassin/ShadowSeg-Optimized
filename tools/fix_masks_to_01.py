"""Normalize segmentation masks to binary labels {0,1} for train/val splits."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    """Parse CLI options for mask normalization."""
    parser = argparse.ArgumentParser(description="Normalize masks in split folders to {0,1}.")
    parser.add_argument(
        "--public-mask-root",
        default="data/public/Rover_Shadow_Public_Dataset/ShadowMasks",
        help="Public dataset mask root containing train/val subfolders.",
    )
    parser.add_argument(
        "--private-mask-root",
        default="data/private/LunarShadowDataset/ShadowMasks",
        help="Private dataset mask root containing train/val subfolders (if present).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Split subfolders to process.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count files only, do not write changes.",
    )
    return parser.parse_args()


def normalize_mask_file(mask_path: Path, dry_run: bool) -> bool:
    """Normalize one mask file to {0,1}; return True if it changed."""
    mask = Image.open(mask_path).convert("L")
    original = np.array(mask)
    fixed = (original > 0).astype(np.uint8)
    changed = not np.array_equal(original, fixed)
    if changed and not dry_run:
        Image.fromarray(fixed).save(mask_path)
    return changed


def normalize_mask_root(mask_root: Path, splits: list[str], dry_run: bool) -> tuple[int, int]:
    """Normalize all masks under split subfolders and return (seen, changed)."""
    seen = 0
    changed = 0

    for split in splits:
        split_dir = mask_root / split
        if not split_dir.is_dir():
            print(f"[WARN] Skipping missing folder: {split_dir}")
            continue

        split_seen = 0
        split_changed = 0
        for mask_path in split_dir.iterdir():
            if mask_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            split_seen += 1
            if normalize_mask_file(mask_path, dry_run=dry_run):
                split_changed += 1

        seen += split_seen
        changed += split_changed
        print(f"[INFO] {split_dir}: processed={split_seen}, changed={split_changed}")

    return seen, changed


def main() -> None:
    """Entry point for split-based mask normalization."""
    args = parse_args()

    public_root = Path(args.public_mask_root)
    private_root = Path(args.private_mask_root)

    print("[INFO] Normalizing PUBLIC dataset masks...")
    public_seen, public_changed = normalize_mask_root(public_root, args.splits, args.dry_run)

    print("[INFO] Normalizing PRIVATE dataset masks...")
    private_seen, private_changed = normalize_mask_root(private_root, args.splits, args.dry_run)

    print("")
    print("[DONE] Mask normalization finished.")
    print(f"[DONE] public: processed={public_seen}, changed={public_changed}")
    print(f"[DONE] private: processed={private_seen}, changed={private_changed}")
    if args.dry_run:
        print("[DONE] dry-run mode was enabled; no files were modified.")


if __name__ == "__main__":
    main()
