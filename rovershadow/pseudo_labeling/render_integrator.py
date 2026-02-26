"""Integration utilities for external-only render pseudo-label datasets."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .external_infer import ExternalShadowInferencer


@dataclass
class RenderSampleRecord:
    """Metadata for one render sample after pseudo-label generation."""

    source_name: str
    source_path: str
    source_id: int
    shadow_frac: float
    boundary_ratio: float
    entropy_mean: float
    quarantine_reason: str | None
    assigned_split: str | None = None
    assigned_stem: str | None = None
    assigned_image_path: str | None = None
    assigned_mask_path: str | None = None


@dataclass
class IntegrationSummary:
    """Final summary for render pseudo-label integration."""

    timestamp: str
    total_render_images: int
    accepted_count: int
    quarantined_count: int
    train_count: int
    val_count: int
    quarantine_written_count: int
    split_ratio: float
    starting_lssd_id: int
    ending_lssd_id: int
    threshold: float
    dry_run: bool


def _timestamp_tag() -> str:
    """Return compact timestamp tag for folders and manifest names."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _extract_num(name: str) -> int | None:
    """Extract first integer token from string."""
    match = re.search(r"(\d+)", name)
    return int(match.group(1)) if match else None


def _sorted_paths(paths: list[Path]) -> list[Path]:
    """Sort file paths by numeric token and then by filename."""
    return sorted(
        paths,
        key=lambda p: (_extract_num(p.stem) if _extract_num(p.stem) is not None else 10**9, p.name),
    )


def _collect_render_images(render_root: Path) -> list[Path]:
    """Collect sorted render images."""
    files = [
        p for p in render_root.iterdir()
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ]
    return _sorted_paths(files)


def _collect_existing_lssd_ids(public_root: Path) -> set[int]:
    """Collect existing lssd numeric IDs from ShadowImages tree."""
    image_root = public_root / "ShadowImages"
    if not image_root.is_dir():
        return set()
    ids: set[int] = set()
    for path in image_root.rglob("*"):
        if not path.is_file():
            continue
        idx = _extract_num(path.stem)
        if idx is not None and path.stem.lower().startswith("lssd"):
            ids.add(idx)
    return ids


def _entropy_binary(prob: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Compute per-pixel binary entropy for probability map."""
    p = np.clip(prob, eps, 1.0 - eps)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def _postprocess_mask(mask: np.ndarray) -> np.ndarray:
    """Apply morphology and component cleanup to binary mask."""
    kernel_open = np.ones((3, 3), dtype=np.uint8)
    kernel_close = np.ones((5, 5), dtype=np.uint8)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    keep = np.zeros_like(cleaned, dtype=np.uint8)
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area >= 64:
            keep[labels == label_id] = 1
    return keep


def _quarantine_reason(
    boundary_ratio: float,
    shadow_frac: float,
    entropy_mean: float,
) -> str | None:
    """Return quarantine reason if a sample violates quality thresholds."""
    reasons: list[str] = []
    if boundary_ratio > 0.45:
        reasons.append("boundary_ratio>0.45")
    if entropy_mean > 0.62:
        reasons.append("entropy_mean>0.62")
    if shadow_frac < 0.002:
        reasons.append("shadow_frac<0.002")
    if shadow_frac > 0.98:
        reasons.append("shadow_frac>0.98")
    return ";".join(reasons) if reasons else None


def _shadow_bin(shadow_frac: float) -> int:
    """Map shadow fraction into one of five stratification bins."""
    if shadow_frac < 0.01:
        return 0
    if shadow_frac < 0.05:
        return 1
    if shadow_frac < 0.15:
        return 2
    if shadow_frac < 0.35:
        return 3
    return 4


def _stratified_split(
    accepted: list[RenderSampleRecord],
    train_ratio: float,
    seed: int,
) -> tuple[list[RenderSampleRecord], list[RenderSampleRecord]]:
    """Split accepted records into train/val with shadow-coverage stratification."""
    bins: dict[int, list[RenderSampleRecord]] = {idx: [] for idx in range(5)}
    for record in accepted:
        bins[_shadow_bin(record.shadow_frac)].append(record)

    rng = random.Random(seed)
    train_records: list[RenderSampleRecord] = []
    val_records: list[RenderSampleRecord] = []
    for bin_idx in range(5):
        bucket = bins[bin_idx]
        rng.shuffle(bucket)
        if not bucket:
            continue
        train_count = int(round(len(bucket) * train_ratio))
        if len(bucket) > 1:
            train_count = max(1, min(len(bucket) - 1, train_count))
        else:
            train_count = 1
        train_records.extend(bucket[:train_count])
        val_records.extend(bucket[train_count:])

    train_records = sorted(train_records, key=lambda r: (r.source_id, r.source_name))
    val_records = sorted(val_records, key=lambda r: (r.source_id, r.source_name))
    return train_records, val_records


def _write_csv(records: list[dict], out_path: Path) -> None:
    """Write list-of-dicts CSV with deterministic field ordering."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        out_path.write_text("", encoding="utf-8")
        return
    fieldnames = list(records[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _compute_tree_digest(root: Path) -> dict:
    """Compute deterministic digest from relative file paths and sizes."""
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    if not root.is_dir():
        return {"root": str(root), "file_count": 0, "total_bytes": 0, "sha256": digest.hexdigest()}
    for path in sorted([p for p in root.rglob("*") if p.is_file()]):
        rel = path.relative_to(root).as_posix()
        size = path.stat().st_size
        file_count += 1
        total_bytes += size
        digest.update(rel.encode("utf-8"))
        digest.update(b"|")
        digest.update(str(size).encode("utf-8"))
        digest.update(b"\n")
    return {
        "root": str(root),
        "file_count": file_count,
        "total_bytes": total_bytes,
        "sha256": digest.hexdigest(),
    }


def _snapshot_public_folders(public_root: Path, snapshot_root: Path) -> None:
    """Create full snapshot backup of ShadowImages and ShadowMasks folders."""
    snapshot_root.mkdir(parents=True, exist_ok=False)
    for name in ("ShadowImages", "ShadowMasks"):
        src = public_root / name
        dst = snapshot_root / name
        if src.is_dir():
            shutil.copytree(src, dst)


def _save_qa_panels(
    records: list[RenderSampleRecord],
    qa_dir: Path,
    sample_count: int,
    seed: int,
    threshold: float,
) -> None:
    """Save compact QA panels for random subset of generated pseudo labels."""
    if not records:
        return
    qa_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    selected = records.copy()
    rng.shuffle(selected)
    selected = selected[: min(sample_count, len(selected))]
    for idx, record in enumerate(selected, start=1):
        image = Image.open(record.source_path).convert("RGB")
        mask_path = qa_dir.parent / "staging_masks" / f"{Path(record.source_name).stem}.png"
        if not mask_path.is_file():
            continue
        mask = np.array(Image.open(mask_path).convert("L")) > 0
        overlay = np.array(image).astype(np.float32)
        overlay[mask] = (0.55 * overlay[mask]) + (0.45 * np.array([255, 0, 0], dtype=np.float32))
        overlay_img = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB")

        mask_vis = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        mask_vis[mask] = (255, 255, 255)
        mask_img = Image.fromarray(mask_vis, mode="RGB")

        w, h = image.size
        title_h = 42
        canvas = Image.new("RGB", (w * 3, h + title_h), (20, 20, 20))
        canvas.paste(image, (0, title_h))
        canvas.paste(mask_img, (w, title_h))
        canvas.paste(overlay_img, (w * 2, title_h))
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 10), f"{record.source_name} | threshold={threshold:.2f}", fill=(240, 240, 240))
        draw.text((w + 10, 10), "Pseudo Mask", fill=(240, 240, 240))
        draw.text((2 * w + 10, 10), f"Overlay | q={record.quarantine_reason or 'accepted'}", fill=(240, 240, 240))
        out_path = qa_dir / f"qa_{idx:04d}_{Path(record.source_name).stem}.png"
        canvas.save(out_path)


def plan_render_integration_dry_run(
    render_root: Path,
    public_root: Path,
    workspace: Path,
    split_ratio: float,
) -> dict:
    """Create dry-run planning manifests without mutating datasets."""
    workspace.mkdir(parents=True, exist_ok=True)
    manifests = workspace / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    render_images = _collect_render_images(render_root)
    existing_ids = _collect_existing_lssd_ids(public_root)
    next_id = (max(existing_ids) + 1) if existing_ids else 1
    planned_train = int(round(len(render_images) * split_ratio))
    planned_val = len(render_images) - planned_train

    preview = [
        {
            "source_name": p.name,
            "source_id": _extract_num(p.stem) or -1,
            "planned_new_stem": f"lssd{next_id + idx}",
        }
        for idx, p in enumerate(render_images[: min(300, len(render_images))])
    ]
    _write_csv(preview, manifests / "dry_run_preview.csv")
    summary = {
        "dry_run": True,
        "total_render_images": len(render_images),
        "planned_train_count": planned_train,
        "planned_val_count": planned_val,
        "planned_starting_lssd_id": next_id,
        "render_root": str(render_root),
        "public_root": str(public_root),
    }
    (manifests / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[OK] Dry-run manifests written to: {manifests}")
    return summary


def integrate_render_dataset(
    render_root: Path,
    public_root: Path,
    workspace: Path,
    archive_root: Path,
    inferencer: ExternalShadowInferencer,
    threshold: float,
    split_ratio: float = 0.9,
    seed: int = 42,
    qa_samples: int = 200,
    max_render_images: int | None = None,
    dry_run: bool = False,
    cleanup_render: bool = True,
) -> IntegrationSummary:
    """Generate pseudo labels and merge render data into public train/val."""
    if dry_run:
        summary = plan_render_integration_dry_run(
            render_root=render_root,
            public_root=public_root,
            workspace=workspace,
            split_ratio=split_ratio,
        )
        return IntegrationSummary(
            timestamp=_timestamp_tag(),
            total_render_images=summary["total_render_images"],
            accepted_count=0,
            quarantined_count=0,
            train_count=summary["planned_train_count"],
            val_count=summary["planned_val_count"],
            quarantine_written_count=0,
            split_ratio=split_ratio,
            starting_lssd_id=summary["planned_starting_lssd_id"],
            ending_lssd_id=summary["planned_starting_lssd_id"] + summary["total_render_images"] - 1,
            threshold=threshold,
            dry_run=True,
        )

    if not render_root.is_dir():
        raise FileNotFoundError(f"Render root not found: {render_root}")
    if not public_root.is_dir():
        raise FileNotFoundError(f"Public root not found: {public_root}")

    workspace.mkdir(parents=True, exist_ok=True)
    manifests_dir = workspace / "manifests"
    qa_dir = workspace / "qa"
    staging_masks_dir = workspace / "staging_masks"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)
    staging_masks_dir.mkdir(parents=True, exist_ok=True)

    render_images = _collect_render_images(render_root)
    if max_render_images is not None:
        render_images = render_images[: max_render_images]
    if not render_images:
        raise RuntimeError(f"No render images found in {render_root}")

    existing_ids = _collect_existing_lssd_ids(public_root)
    next_id = (max(existing_ids) + 1) if existing_ids else 1
    timestamp = _timestamp_tag()

    pre_checksums = {
        "shadow_images": _compute_tree_digest(public_root / "ShadowImages"),
        "shadow_masks": _compute_tree_digest(public_root / "ShadowMasks"),
    }
    (manifests_dir / "checksums_premerge.json").write_text(
        json.dumps(pre_checksums, indent=2),
        encoding="utf-8",
    )

    records: list[RenderSampleRecord] = []
    for idx, image_path in enumerate(render_images, start=1):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed reading render image: {image_path}")
        prob = inferencer.predict_shadow_probability(image, tta="flip-ms")
        raw_mask = (prob >= threshold).astype(np.uint8)
        mask = _postprocess_mask(raw_mask)
        boundary_ratio = float(np.mean(np.abs(prob - threshold) < 0.08))
        shadow_frac = float(np.mean(mask == 1))
        entropy_mean = float(np.mean(_entropy_binary(prob)))
        reason = _quarantine_reason(
            boundary_ratio=boundary_ratio,
            shadow_frac=shadow_frac,
            entropy_mean=entropy_mean,
        )
        record = RenderSampleRecord(
            source_name=image_path.name,
            source_path=str(image_path),
            source_id=_extract_num(image_path.stem) or idx,
            shadow_frac=shadow_frac,
            boundary_ratio=boundary_ratio,
            entropy_mean=entropy_mean,
            quarantine_reason=reason,
        )
        records.append(record)

        mask_path = staging_masks_dir / f"{image_path.stem}.png"
        Image.fromarray(mask.astype(np.uint8), mode="L").save(mask_path)
        if idx % 100 == 0 or idx == len(render_images):
            print(f"[INFO] Generated pseudo mask {idx}/{len(render_images)}")

    accepted = [r for r in records if r.quarantine_reason is None]
    quarantined = [r for r in records if r.quarantine_reason is not None]

    train_records, val_records = _stratified_split(accepted, train_ratio=split_ratio, seed=seed)

    used_ids = set(existing_ids)
    assign_counter = next_id

    def _assign_record(record: RenderSampleRecord, split: str, image_dir: Path, mask_dir: Path) -> None:
        nonlocal assign_counter
        while assign_counter in used_ids:
            assign_counter += 1
        stem = f"lssd{assign_counter}"
        used_ids.add(assign_counter)
        assign_counter += 1
        record.assigned_split = split
        record.assigned_stem = stem
        record.assigned_image_path = str(image_dir / f"{stem}.jpg")
        record.assigned_mask_path = str(mask_dir / f"{stem}.png")

    train_img_dir = public_root / "ShadowImages" / "train"
    val_img_dir = public_root / "ShadowImages" / "val"
    train_mask_dir = public_root / "ShadowMasks" / "train"
    val_mask_dir = public_root / "ShadowMasks" / "val"
    quarantine_img_dir = public_root / "ShadowImages" / "quarantine_render"
    quarantine_mask_dir = public_root / "ShadowMasks" / "quarantine_render"

    for path in [train_img_dir, val_img_dir, train_mask_dir, val_mask_dir, quarantine_img_dir, quarantine_mask_dir]:
        path.mkdir(parents=True, exist_ok=True)

    for record in train_records:
        _assign_record(record, "train", train_img_dir, train_mask_dir)
    for record in val_records:
        _assign_record(record, "val", val_img_dir, val_mask_dir)
    for record in quarantined:
        _assign_record(record, "quarantine_render", quarantine_img_dir, quarantine_mask_dir)

    qa_records = records.copy()
    _save_qa_panels(
        records=qa_records,
        qa_dir=qa_dir,
        sample_count=qa_samples,
        seed=seed,
        threshold=threshold,
    )

    archive_root.mkdir(parents=True, exist_ok=True)
    snapshot_dir = archive_root / f"public_snapshot_{timestamp}"
    _snapshot_public_folders(public_root=public_root, snapshot_root=snapshot_dir)

    def _write_assigned(record: RenderSampleRecord) -> None:
        assert record.assigned_image_path is not None
        assert record.assigned_mask_path is not None
        source = Path(record.source_path)
        image_bgr = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read source image for write: {source}")
        ok = cv2.imwrite(record.assigned_image_path, image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ok:
            raise RuntimeError(f"Failed writing JPEG image: {record.assigned_image_path}")
        staged_mask_path = staging_masks_dir / f"{source.stem}.png"
        mask_arr = np.array(Image.open(staged_mask_path).convert("L"))
        mask_arr = (mask_arr > 0).astype(np.uint8)
        Image.fromarray(mask_arr, mode="L").save(record.assigned_mask_path)

    for rec in train_records + val_records + quarantined:
        _write_assigned(rec)

    if cleanup_render:
        raw_archive = archive_root / f"render_raw_{timestamp}"
        raw_archive.mkdir(parents=True, exist_ok=False)
        for path in render_images:
            shutil.move(str(path), str(raw_archive / path.name))

    accepted_rows = [asdict(r) for r in (train_records + val_records)]
    quarantined_rows = [asdict(r) for r in quarantined]
    rename_rows = [
        {
            "source_name": r.source_name,
            "source_path": r.source_path,
            "assigned_split": r.assigned_split,
            "assigned_stem": r.assigned_stem,
            "assigned_image_path": r.assigned_image_path,
            "assigned_mask_path": r.assigned_mask_path,
        }
        for r in (train_records + val_records + quarantined)
    ]

    _write_csv(accepted_rows, manifests_dir / "accepted.csv")
    _write_csv(quarantined_rows, manifests_dir / "quarantined.csv")
    _write_csv(rename_rows, manifests_dir / "rename_map.csv")

    summary = IntegrationSummary(
        timestamp=timestamp,
        total_render_images=len(render_images),
        accepted_count=len(train_records) + len(val_records),
        quarantined_count=len(quarantined),
        train_count=len(train_records),
        val_count=len(val_records),
        quarantine_written_count=len(quarantined),
        split_ratio=split_ratio,
        starting_lssd_id=next_id,
        ending_lssd_id=assign_counter - 1,
        threshold=threshold,
        dry_run=False,
    )
    (manifests_dir / "split_summary.json").write_text(
        json.dumps(asdict(summary), indent=2),
        encoding="utf-8",
    )

    post_checksums = {
        "shadow_images": _compute_tree_digest(public_root / "ShadowImages"),
        "shadow_masks": _compute_tree_digest(public_root / "ShadowMasks"),
    }
    (manifests_dir / "checksums_postmerge.json").write_text(
        json.dumps(post_checksums, indent=2),
        encoding="utf-8",
    )
    print(f"[OK] Integration manifests written to: {manifests_dir}")
    return summary
