"""Prepare render-domain pseudo labels using external-only shadow models."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import cv2
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def parse_args() -> argparse.Namespace:
    """Parse command-line args for render-domain integration pipeline."""
    parser = argparse.ArgumentParser(
        description="External-only pseudo-label integration for data/render."
    )
    parser.add_argument("--render-root", default="data/render", help="Render source folder.")
    parser.add_argument(
        "--public-root",
        default="data/public/Rover_Shadow_Public_Dataset",
        help="Public dataset root with ShadowImages/ShadowMasks.",
    )
    parser.add_argument(
        "--external-model",
        default="auto",
        choices=["auto", "mtmt", "dhan"],
        help="External checkpoint source preference.",
    )
    parser.add_argument(
        "--external-weights",
        default=None,
        help="Optional explicit external checkpoint path.",
    )
    parser.add_argument(
        "--external-config",
        default="configs/shadow_external_segformer_b0.py",
        help="Config used for external inference.",
    )
    parser.add_argument(
        "--fallback-config",
        default="configs/shadow_external_segformer_b0.py",
        help="Config used for fallback external model training.",
    )
    parser.add_argument(
        "--fallback-work-dir",
        default="work_dirs/shadow_external_segformer_b0_fallback",
        help="Work directory for fallback training when downloads fail.",
    )
    parser.add_argument(
        "--fallback-max-iters",
        type=int,
        default=12000,
        help="Max iterations for fallback external training.",
    )
    parser.add_argument(
        "--fallback-val-interval",
        type=int,
        default=1000,
        help="Validation interval for fallback external training.",
    )
    parser.add_argument(
        "--fallback-batch-size",
        type=int,
        default=2,
        help="Batch size for fallback external training.",
    )
    parser.add_argument(
        "--cache-dir",
        default="data/_model_cache/external_shadow",
        help="Local cache for auto-fetched external checkpoints.",
    )
    parser.add_argument(
        "--archive-root",
        default="data/archive",
        help="Archive root for snapshots and raw render backups.",
    )
    parser.add_argument(
        "--workspace",
        default="data/_staging_render",
        help="Workspace root for manifests and intermediate artifacts.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Execution device. auto selects cuda when available.",
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.9,
        help="Train split ratio for accepted pseudo-labeled render samples.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--qa-samples",
        type=int,
        default=200,
        help="Number of QA panels to export.",
    )
    parser.add_argument(
        "--calibration-thresholds",
        nargs="+",
        type=float,
        default=[0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65],
        help="Threshold candidates for public-val F0.5 calibration.",
    )
    parser.add_argument(
        "--calibration-max-images",
        type=int,
        default=None,
        help="Optional cap for public-val images used in threshold calibration.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan and manifest only; do not modify public/render datasets.",
    )
    parser.add_argument(
        "--simulate-download-failure",
        action="store_true",
        help="Force download resolver failure to test fallback training path.",
    )
    parser.add_argument(
        "--max-render-images",
        type=int,
        default=None,
        help="Optional cap on processed render images (for quick smoke checks).",
    )
    parser.add_argument(
        "--keep-render",
        action="store_true",
        help="Do not archive+clear data/render after successful integration.",
    )
    parser.add_argument(
        "--fallback-only-smoke",
        action="store_true",
        help="Run external-resolution/fallback smoke only, then exit without integration.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    """Resolve explicit/auto device requests."""
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no GPU is visible to PyTorch.")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _collect_render_preview(render_root: Path, limit: int = 5) -> list[Path]:
    """Collect sorted preview image list from render folder."""
    files = sorted([p for p in render_root.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    return files[: min(limit, len(files))]


def _write_json(path: Path, payload: dict) -> None:
    """Write JSON helper with parent directory creation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _collect_public_pairs(public_root: Path) -> list[tuple[Path, Path]]:
    """Collect public validation image/mask pairs."""
    image_root = public_root / "ShadowImages" / "val"
    mask_root = public_root / "ShadowMasks" / "val"
    pairs: list[tuple[Path, Path]] = []
    for image_path in sorted(image_root.glob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        mask_path = mask_root / f"{image_path.stem}.png"
        if mask_path.is_file():
            pairs.append((image_path, mask_path))
    return pairs


def main() -> None:
    """Entry point for render pseudo-label integration."""
    args = parse_args()
    device = resolve_device(args.device)
    from rovershadow.pseudo_labeling import (
        ExternalShadowInferencer,
        calibrate_shadow_threshold,
        integrate_render_dataset,
        plan_render_integration_dry_run,
        resolve_external_checkpoint,
        train_fallback_external_model,
    )

    render_root = Path(args.render_root)
    public_root = Path(args.public_root)
    external_config = Path(args.external_config)
    fallback_config = Path(args.fallback_config)
    external_weights = Path(args.external_weights) if args.external_weights else None
    cache_dir = Path(args.cache_dir)
    fallback_work_dir = Path(args.fallback_work_dir)
    archive_root = Path(args.archive_root)
    workspace = Path(args.workspace)

    if not render_root.is_dir():
        raise FileNotFoundError(f"Render root not found: {render_root}")
    if not public_root.is_dir():
        raise FileNotFoundError(f"Public root not found: {public_root}")
    if not external_config.is_file():
        raise FileNotFoundError(f"External config not found: {external_config}")
    if not fallback_config.is_file():
        raise FileNotFoundError(f"Fallback config not found: {fallback_config}")
    if not (0.0 < args.split_ratio < 1.0):
        raise ValueError("--split-ratio must be in (0,1).")

    manifests_dir = workspace / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        dry_summary = plan_render_integration_dry_run(
            render_root=render_root,
            public_root=public_root,
            workspace=workspace,
            split_ratio=args.split_ratio,
        )
        _write_json(manifests_dir / "dry_run_report.json", dry_summary)
        print("[DONE] Dry-run completed without dataset mutations.")
        return

    resolved_ckpt, ckpt_source = resolve_external_checkpoint(
        cache_dir=cache_dir,
        external_model=args.external_model,
        explicit_checkpoint=external_weights,
        simulate_download_failure=args.simulate_download_failure,
    )

    if resolved_ckpt is None:
        print("[WARN] External checkpoint unavailable. Triggering fallback training.")
        resolved_ckpt = train_fallback_external_model(
            config_path=fallback_config,
            public_root=public_root,
            work_dir=fallback_work_dir,
            device=device,
            max_iters=args.fallback_max_iters,
            val_interval=args.fallback_val_interval,
            batch_size=args.fallback_batch_size,
            seed=args.seed,
        )
        ckpt_source = "fallback_training"

    inferencer: ExternalShadowInferencer | None = None
    try:
        inferencer = ExternalShadowInferencer(
            config_path=external_config,
            checkpoint_path=resolved_ckpt,
            device=device,
        )
    except Exception as exc:
        print(f"[WARN] Resolved checkpoint could not initialize inferencer: {exc}")
        print("[WARN] Reverting to fallback external training.")
        resolved_ckpt = train_fallback_external_model(
            config_path=fallback_config,
            public_root=public_root,
            work_dir=fallback_work_dir,
            device=device,
            max_iters=args.fallback_max_iters,
            val_interval=args.fallback_val_interval,
            batch_size=args.fallback_batch_size,
            seed=args.seed,
        )
        ckpt_source = "fallback_training_after_init_failure"
        inferencer = ExternalShadowInferencer(
            config_path=external_config,
            checkpoint_path=resolved_ckpt,
            device=device,
        )

    public_pairs = _collect_public_pairs(public_root)
    if args.calibration_max_images is not None:
        public_pairs = public_pairs[: args.calibration_max_images]
    if not public_pairs:
        raise RuntimeError("No public validation pairs found for threshold calibration.")
    calibration = calibrate_shadow_threshold(
        inferencer=inferencer,
        public_pairs=public_pairs,
        thresholds=args.calibration_thresholds,
        tta="flip-ms",
        progress_interval=100,
    )
    _write_json(
        manifests_dir / "threshold_calibration.json",
        {
            "best_threshold": calibration.best_threshold,
            "best_f0_5": calibration.best_f05,
            "rows": calibration.rows,
        },
    )
    print(
        f"[INFO] Calibrated threshold: {calibration.best_threshold:.3f} "
        f"(shadow F0.5={calibration.best_f05:.4f})"
    )

    preview_images = _collect_render_preview(render_root, limit=5)
    smoke_rows: list[dict] = []
    for path in preview_images:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        prob = inferencer.predict_shadow_probability(image, tta="flip-ms")
        smoke_rows.append(
            {
                "image": path.name,
                "mean_prob": float(prob.mean()),
                "max_prob": float(prob.max()),
                "min_prob": float(prob.min()),
                "shadow_frac_at_best_threshold": float((prob >= calibration.best_threshold).mean()),
            }
        )
    _write_json(manifests_dir / "external_smoke_inference.json", {"rows": smoke_rows})
    print(f"[OK] External smoke inference rows: {len(smoke_rows)}")

    if args.fallback_only_smoke:
        _write_json(
            manifests_dir / "integration_summary.json",
            {
                "mode": "fallback_only_smoke",
                "external_checkpoint": str(resolved_ckpt),
                "external_checkpoint_source": ckpt_source,
                "calibrated_threshold": calibration.best_threshold,
                "smoke_rows_count": len(smoke_rows),
            },
        )
        print("[DONE] Fallback-only smoke mode completed; integration skipped.")
        return

    integration_summary = integrate_render_dataset(
        render_root=render_root,
        public_root=public_root,
        workspace=workspace,
        archive_root=archive_root,
        inferencer=inferencer,
        threshold=calibration.best_threshold,
        split_ratio=args.split_ratio,
        seed=args.seed,
        qa_samples=args.qa_samples,
        max_render_images=args.max_render_images,
        dry_run=False,
        cleanup_render=not args.keep_render,
    )
    summary_payload = asdict(integration_summary)
    summary_payload["external_checkpoint"] = str(resolved_ckpt)
    summary_payload["external_checkpoint_source"] = ckpt_source
    _write_json(manifests_dir / "integration_summary.json", summary_payload)
    print("[DONE] Render pseudo-label integration completed.")


if __name__ == "__main__":
    main()
