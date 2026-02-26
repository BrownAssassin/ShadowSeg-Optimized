"""Normalize flat private-set masks to binary labels {0,1}."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for private mask normalization."""
    parser = argparse.ArgumentParser(description="Normalize private masks to {0,1}.")
    parser.add_argument(
        "--mask-root",
        default="data/private/LunarShadowDataset/ShadowMasks",
        help="Private mask directory (flat, no split subfolders).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count files only, do not write changes.",
    )
    return parser.parse_args()


def normalize_mask(mask_path: Path, dry_run: bool) -> bool:
    """Normalize one private mask and return True when file content changes."""
    original = np.array(Image.open(mask_path).convert("L"))
    fixed = (original > 0).astype(np.uint8)
    changed = not np.array_equal(original, fixed)
    if changed and not dry_run:
        Image.fromarray(fixed).save(mask_path)
    return changed


def main() -> None:
    """Entry point for private mask normalization."""
    args = parse_args()
    root = Path(args.mask_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Mask folder not found: {root}")

    processed = 0
    changed = 0

    for mask_path in root.iterdir():
        if mask_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        processed += 1
        if normalize_mask(mask_path, dry_run=args.dry_run):
            changed += 1

    print(f"[DONE] processed={processed}, changed={changed}, root={root}")
    if args.dry_run:
        print("[DONE] dry-run mode was enabled; no files were modified.")


if __name__ == "__main__":
    main()
