"""Run MMSeg inference for RoverShadow models."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rovershadow.runtime import ensure_runtime_env, install_mmcv_ops_shim_if_needed

ensure_runtime_env()
install_mmcv_ops_shim_if_needed()

import rovershadow.losses  # noqa: F401
import torch
from mmengine.config import Config
from mmseg.apis import inference_model, init_model
from mmseg.apis.inference import show_result_pyplot


def die(msg: str, code: int = 1) -> None:
    """Exit with a formatted error message."""
    print(f"[ERROR] {msg}")
    sys.exit(code)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for one-image inference."""
    parser = argparse.ArgumentParser(description="Run MMSeg inference for RoverShadow model.")
    parser.add_argument(
        "--img",
        default=r"data/public/Rover_Shadow_Public_Dataset/ShadowImages/val/lssd4000.jpg",
        help="Input image path.",
    )
    parser.add_argument(
        "--cfg",
        default=r"configs/shadow_deeplabv3plus_r50.py",
        help="Config path.",
    )
    parser.add_argument(
        "--ckpt",
        default=None,
        help="Checkpoint path. Required unless you provide a valid local default.",
    )
    parser.add_argument(
        "--out",
        default=r"demo_result.png",
        help="Output visualization path.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Execution device. auto picks cuda if available, else cpu.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    """Resolve explicit/auto device request."""
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            die("CUDA requested but no GPU is visible to PyTorch.")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    """Run inference and write a rendered prediction image."""
    args = parse_args()
    img_path = args.img
    cfg_path = args.cfg
    ckpt_path = args.ckpt
    out_path = args.out
    device = resolve_device(args.device)

    print("[INFO] Current working directory:", os.getcwd())
    print("[INFO] img:", img_path)
    print("[INFO] cfg:", cfg_path)
    print("[INFO] ckpt:", ckpt_path)
    print("[INFO] out:", out_path)
    print("[INFO] device:", device)

    if not ckpt_path:
        die("Checkpoint path is required. Pass --ckpt <path/to/checkpoint.pth>.")
    if not os.path.isfile(img_path):
        die(f"Image not found: {img_path}")
    if not os.path.isfile(cfg_path):
        die(f"Config not found: {cfg_path}")
    if not os.path.isfile(ckpt_path):
        die(f"Checkpoint not found: {ckpt_path}")

    # Load config so test pipeline can be injected when absent.
    cfg = Config.fromfile(cfg_path)

    # If cfg.test_pipeline is missing, copy it from test/val dataloader.
    if not hasattr(cfg, "test_pipeline"):
        print("[WARN] cfg.test_pipeline missing. Injecting from test_dataloader.dataset.pipeline ...")
        if "test_dataloader" in cfg and "dataset" in cfg.test_dataloader and "pipeline" in cfg.test_dataloader.dataset:
            cfg.test_pipeline = cfg.test_dataloader.dataset.pipeline
        elif "val_dataloader" in cfg and "dataset" in cfg.val_dataloader and "pipeline" in cfg.val_dataloader.dataset:
            cfg.test_pipeline = cfg.val_dataloader.dataset.pipeline
        else:
            die("Could not find a pipeline in test_dataloader/val_dataloader to use as test_pipeline.")

    print(f"[INFO] Initializing model ({device})...")
    model = init_model(cfg, ckpt_path, device=device)

    print("[INFO] Running inference...")
    result = inference_model(model, img_path)

    print("[INFO] Saving visualization to:", out_path)
    out_parent = Path(out_path).parent
    if str(out_parent) and str(out_parent) != ".":
        out_parent.mkdir(parents=True, exist_ok=True)
    show_result_pyplot(
        model,
        img_path,
        result,
        show=False,
        out_file=out_path,
        opacity=0.6
    )

    if os.path.isfile(out_path):
        print("[OK] Saved:", os.path.abspath(out_path))
    else:
        die("Inference finished but output image was not created. (Unexpected)")

if __name__ == "__main__":
    main()
