"""Fallback trainer for external-only pseudo-label generation models."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from mmengine.config import Config
from mmengine.runner import Runner

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rovershadow.runtime import ensure_runtime_env, install_mmcv_ops_shim_if_needed

ensure_runtime_env()
install_mmcv_ops_shim_if_needed()


def _set_if_present(cfg, path: tuple[str, ...], value) -> None:
    """Set nested config values when all keys in the path exist."""
    node = cfg
    for key in path[:-1]:
        if key not in node:
            return
        node = node[key]
    if path[-1] in node:
        node[path[-1]] = value


def _best_checkpoint_from_workdir(work_dir: Path) -> Path | None:
    """Find best checkpoint by mmengine naming conventions."""
    candidates = sorted(work_dir.glob("best_*.pth"))
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    last = work_dir / "last_checkpoint"
    if last.is_file():
        ckpt = Path(last.read_text(encoding="utf-8").strip())
        if ckpt.is_file():
            return ckpt
    return None


def train_fallback_external_model(
    config_path: Path,
    public_root: Path,
    work_dir: Path,
    device: str = "cuda",
    max_iters: int = 12000,
    val_interval: int = 1000,
    batch_size: int = 2,
    seed: int = 42,
) -> Path:
    """Train fallback external model and return best checkpoint path."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Fallback config not found: {config_path}")
    if not public_root.is_dir():
        raise FileNotFoundError(f"Public dataset root not found: {public_root}")
    work_dir.mkdir(parents=True, exist_ok=True)

    if device == "cpu":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

    cfg = Config.fromfile(str(config_path))
    cfg.launcher = "none"
    cfg.device = device
    cfg.work_dir = str(work_dir)
    cfg.data_root = str(public_root)
    cfg.train_cfg.max_iters = max_iters
    cfg.train_cfg.val_interval = val_interval
    cfg.randomness = dict(seed=seed, deterministic=False)

    if "param_scheduler" in cfg:
        schedulers = cfg.param_scheduler
        if isinstance(schedulers, dict):
            schedulers = [schedulers]
        for scheduler in schedulers:
            if scheduler.get("by_epoch", False):
                continue
            if "end" in scheduler:
                scheduler["end"] = max_iters

    _set_if_present(cfg, ("train_dataloader", "batch_size"), batch_size)
    _set_if_present(cfg, ("default_hooks", "checkpoint", "interval"), val_interval)
    _set_if_present(cfg, ("default_hooks", "checkpoint", "by_epoch"), False)

    for loader_name in ("train_dataloader", "val_dataloader", "test_dataloader"):
        if loader_name not in cfg:
            continue
        loader = cfg[loader_name]
        if loader is None:
            continue
        if loader.get("num_workers", 0) == 0:
            loader["persistent_workers"] = False

    print(f"[INFO] Fallback training config: {config_path}")
    print(f"[INFO] Fallback training work_dir: {work_dir}")
    print(f"[INFO] Fallback training max_iters: {max_iters}")
    print(f"[INFO] Fallback training val_interval: {val_interval}")
    print(f"[INFO] Fallback training batch_size: {batch_size}")
    print(f"[INFO] Fallback training device: {device}")

    runner = Runner.from_cfg(cfg)
    runner.train()

    best = _best_checkpoint_from_workdir(work_dir)
    if best is None:
        raise RuntimeError(
            "Fallback training completed but no checkpoint could be resolved from work_dir."
        )
    print(f"[OK] Fallback external checkpoint: {best}")
    return best
