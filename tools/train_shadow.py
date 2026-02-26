"""Train RoverShadow segmentation models with reproducible CLI overrides."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    """Parse command-line options for training."""
    parser = argparse.ArgumentParser(description="Train a RoverShadow model.")
    parser.add_argument(
        "--config",
        default="configs/shadow_deeplabv3plus_r50.py",
        help="Path to the training config.",
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Output directory for checkpoints/logs.",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=None,
        help="Override total training iterations.",
    )
    parser.add_argument(
        "--val-interval",
        type=int,
        default=None,
        help="Override validation interval (iterations).",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Disable validation loop (useful for quick plumbing smoke tests).",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=None,
        help="Override checkpoint save interval (iterations).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override optimizer learning rate.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Execution device. auto selects cuda when available.",
    )
    parser.add_argument(
        "--load-from",
        default=None,
        help="Checkpoint path to initialize model weights.",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Resume training state from this checkpoint.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic algorithms where possible.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use AMP training wrapper.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    """Resolve explicit/auto device request."""
    if requested == "cpu":
        return "cpu"
    import torch

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no GPU is visible to PyTorch.")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _set_if_present(cfg, path: tuple[str, ...], value) -> None:
    """Set nested config values when all keys exist."""
    node = cfg
    for key in path[:-1]:
        if key not in node:
            return
        node = node[key]
    if path[-1] in node:
        node[path[-1]] = value


def apply_overrides(cfg, args: argparse.Namespace, device: str) -> None:
    """Apply CLI overrides to loaded MMEngine config."""
    cfg.launcher = "none"
    cfg.device = device

    if args.work_dir:
        cfg.work_dir = args.work_dir

    if args.max_iters is not None:
        cfg.train_cfg.max_iters = args.max_iters
        if "param_scheduler" in cfg:
            schedulers = cfg.param_scheduler
            if isinstance(schedulers, dict):
                schedulers = [schedulers]
            for scheduler in schedulers:
                if scheduler.get("by_epoch", False):
                    continue
                if "end" in scheduler:
                    scheduler["end"] = args.max_iters

    if args.no_validate:
        cfg.val_cfg = None
        cfg.val_dataloader = None
        cfg.val_evaluator = None
    elif args.val_interval is not None:
        cfg.train_cfg.val_interval = args.val_interval

    if args.checkpoint_interval is not None:
        cfg.default_hooks.checkpoint.interval = args.checkpoint_interval
        cfg.default_hooks.checkpoint.by_epoch = False

    if args.lr is not None:
        _set_if_present(cfg, ("optim_wrapper", "optimizer", "lr"), args.lr)
        _set_if_present(cfg, ("optimizer", "lr"), args.lr)

    if args.load_from is not None:
        cfg.load_from = args.load_from
    if args.resume_from is not None:
        cfg.resume = True
        cfg.load_from = args.resume_from

    if args.seed is not None:
        cfg.randomness = dict(seed=args.seed, deterministic=args.deterministic)

    if args.amp:
        optimizer = cfg.optim_wrapper.optimizer
        cfg.optim_wrapper = dict(type="AmpOptimWrapper", optimizer=optimizer, loss_scale="dynamic")

    # num_workers=0 requires persistent_workers=False on Windows/PyTorch.
    for loader_name in ("train_dataloader", "val_dataloader", "test_dataloader"):
        if loader_name not in cfg:
            continue
        dataloader = cfg[loader_name]
        if dataloader is None:
            continue
        if dataloader.get("num_workers", 0) == 0:
            dataloader["persistent_workers"] = False


def main() -> None:
    """Entry point for training."""
    args = parse_args()
    device = resolve_device(args.device)
    if device == "cpu":
        # Must be set before importing torch/mmengine modules that initialize CUDA.
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

    from rovershadow.runtime import ensure_runtime_env, install_mmcv_ops_shim_if_needed

    ensure_runtime_env()
    install_mmcv_ops_shim_if_needed()

    import rovershadow.losses  # noqa: F401
    from mmengine.config import Config
    from mmengine.runner import Runner

    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    cfg = Config.fromfile(str(cfg_path))
    apply_overrides(cfg, args, device)

    print(f"[INFO] config: {cfg_path}")
    print(f"[INFO] work_dir: {cfg.work_dir}")
    print(f"[INFO] device: {device}")
    print(f"[INFO] max_iters: {cfg.train_cfg.max_iters}")
    print(f"[INFO] val_interval: {cfg.train_cfg.val_interval}")

    runner = Runner.from_cfg(cfg)
    runner.train()

    last_ckpt = Path(cfg.work_dir) / "last_checkpoint"
    if last_ckpt.is_file():
        ckpt_path = last_ckpt.read_text(encoding="utf-8").strip()
        print(f"[OK] last_checkpoint: {ckpt_path}")
    else:
        print("[WARN] last_checkpoint was not produced.")


if __name__ == "__main__":
    main()
