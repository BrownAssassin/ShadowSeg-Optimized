# ShadowSeg: Lighting-Aware Terrain Intelligence

Binary semantic segmentation for shadow detection using MMSegmentation.
Originally created as a DeveloperWeek 2026 Hackathon project, with continued post-submission model development in this repository.

## Overview

This repository contains:

- DeepLabV3+ configs for ShadowSeg/RoverShadow (`R50` baseline, `R101` candidate)
- a custom false-positive-aware loss (`ShadowFalsePositiveLoss`)
- a safe class-weighted cross-entropy variant (`SafeCrossEntropyLoss`)
- shared runtime compatibility helpers for `mmcv-lite`
- CLI tools for training, evaluation, inference, and diagnostics

## Project Origin and Credits

- Original hackathon submission: [ShadowSeg: Lighting-Aware Terrain Intelligence (Devpost)](https://devpost.com/software/shadowseg-lighting-aware-terrain-intelligence)
- Origin: DeveloperWeek 2026 Hackathon
- Original team members:
  - Mrinank Sivakumar ([BrownAssassin](https://github.com/BrownAssassin))
  - Arv Bali ([ArvBali2101](https://github.com/ArvBali2101))
  - Myles Liu
  - Kenji Baritua

## Project Layout

- `configs/shadow_deeplabv3plus_r50.py`: main DeepLabV3+ config
- `configs/shadow_deeplabv3plus_r101.py`: R101 candidate config
- `configs/shadow_external_segformer_b0.py`: fallback external pseudo-label model
- `configs/shadow_fcn_r50.py`: legacy FCN baseline config
- `rovershadow/runtime/mmcv_ops_shim.py`: shared `mmcv.ops` shim logic
- `rovershadow/losses/shadow_false_positive_loss.py`: custom loss module
- `rovershadow/pseudo_labeling/*`: external-only pseudo-labeling pipeline modules
- `tools/train_shadow.py`: reproducible training CLI
- `tools/eval_shadow.py`: public/private evaluation CLI
- `tools/prepare_render_domain_data.py`: external-only render integration pipeline
- `tools/verify_dataset_integrity.py`: dataset integrity verification CLI
- `tools/export_private_triptychs.py`: side-by-side diagnostic export
- `run_infer.py`: single-image inference CLI

## Environment

Use one of the pinned requirement files in this repo:

- CPU baseline: `requirements-cpu.txt`
- GPU baseline (CUDA 13.0): `requirements-gpu-cu130.txt`

Example install (GPU):

```powershell
python3.10 -m pip install -r requirements-gpu-cu130.txt --index-url https://download.pytorch.org/whl/cu130
```

Use `python3.10` for all commands below to avoid accidentally using a different Python install.

## Data

Expected dataset paths:

- public train images: `data/public/Rover_Shadow_Public_Dataset/ShadowImages/train`
- public train masks: `data/public/Rover_Shadow_Public_Dataset/ShadowMasks/train`
- public val images: `data/public/Rover_Shadow_Public_Dataset/ShadowImages/val`
- public val masks: `data/public/Rover_Shadow_Public_Dataset/ShadowMasks/val`
- private holdout images: `data/private/LunarShadowDataset/ShadowImages`
- private holdout masks: `data/private/LunarShadowDataset/ShadowMasks`

Exported best model artifact:

- best checkpoint: `artifacts/best_private_model/iter_11000.pth`
- paired metrics: `artifacts/best_private_model/metrics_*.json`

## Holdout Policy

Private data is evaluation-only holdout:

- do model selection on public train/val only
- run private evaluation as a final locked check
- do not repeatedly tune hyperparameters on private metrics

## Training

Standard DeepLabV3+ R50 run:

```powershell
python3.10 tools/train_shadow.py --config configs/shadow_deeplabv3plus_r50.py --work-dir work_dirs/shadow_deeplabv3plus_r50_exp1 --max-iters 12000 --val-interval 1000 --device cuda
```

Short sweep run example:

```powershell
python3.10 tools/train_shadow.py --config configs/shadow_deeplabv3plus_r50.py --work-dir work_dirs/shadow_deeplabv3plus_r50_sweep_a --max-iters 2000 --val-interval 1000 --checkpoint-interval 1000 --device cuda --lr 0.01
```

R101 candidate run example:

```powershell
python3.10 tools/train_shadow.py --config configs/shadow_deeplabv3plus_r101.py --work-dir work_dirs/shadow_deeplabv3plus_r101_candidate --max-iters 8000 --val-interval 1000 --device cuda
```

Fast plumbing smoke (skip validation):

```powershell
python3.10 tools/train_shadow.py --config configs/shadow_deeplabv3plus_r50.py --work-dir work_dirs/shadow_deeplabv3plus_r50_smoke --max-iters 1 --checkpoint-interval 1 --device cuda --no-validate
```

## Evaluation

Public validation:

```powershell
python3.10 tools/eval_shadow.py --config configs/shadow_deeplabv3plus_r50.py --ckpt work_dirs/shadow_deeplabv3plus_r50_exp1/iter_12000.pth --split public-val --device cuda --save-json work_dirs/shadow_deeplabv3plus_r50_exp1/public_val_metrics.json
```

Private holdout final check:

```powershell
python3.10 tools/eval_shadow.py --config configs/shadow_deeplabv3plus_r50.py --ckpt work_dirs/shadow_deeplabv3plus_r50_exp1/iter_12000.pth --split private --device cuda --tta flip-ms --shadow-threshold 0.55 --save-json work_dirs/shadow_deeplabv3plus_r50_exp1/private_metrics.json
```

## Inference

Single image inference:

```powershell
python3.10 run_infer.py --img data/public/Rover_Shadow_Public_Dataset/ShadowImages/val/lssd4000.jpg --cfg configs/shadow_deeplabv3plus_r50.py --ckpt work_dirs/shadow_deeplabv3plus_r50_exp1/iter_12000.pth --device cuda --out outputs/demo_result.png
```

## Diagnostics

Export private triptychs (`image / GT / prediction`) and overlay triptychs:

```powershell
python3.10 tools/export_private_triptychs.py --cfg configs/shadow_deeplabv3plus_r50.py --ckpt work_dirs/shadow_deeplabv3plus_r50_exp1/iter_12000.pth --out-dir work_dirs/private_triptychs --out-overlay-dir work_dirs/private_triptychs_overlay --tta flip-ms --shadow-threshold 0.55 --device cuda
```

## Render Domain Integration (External-Only)

Dry-run preflight (no public/render mutations):

```powershell
python3.10 tools/prepare_render_domain_data.py --render-root data/render --public-root data/public/Rover_Shadow_Public_Dataset --external-model auto --device cuda --split-ratio 0.9 --seed 42 --qa-samples 200 --archive-root data/archive --workspace data/_staging_render --dry-run
```

Full integration run:

```powershell
python3.10 tools/prepare_render_domain_data.py --render-root data/render --public-root data/public/Rover_Shadow_Public_Dataset --external-model auto --device cuda --split-ratio 0.9 --seed 42 --qa-samples 200 --archive-root data/archive --workspace data/_staging_render
```

Fallback-path smoke (simulate failed downloads, train fallback external model, skip merge):

```powershell
python3.10 tools/prepare_render_domain_data.py --render-root data/render --public-root data/public/Rover_Shadow_Public_Dataset --simulate-download-failure --fallback-only-smoke --fallback-max-iters 1 --fallback-val-interval 1 --calibration-max-images 20 --device cuda --workspace data/_staging_render_smoke
```

Optional explicit external checkpoint:

```powershell
python3.10 tools/prepare_render_domain_data.py --render-root data/render --public-root data/public/Rover_Shadow_Public_Dataset --external-weights path/to/external_model.pth --device cuda
```

Post-run integrity verification:

```powershell
python3.10 tools/verify_dataset_integrity.py --public-root data/public/Rover_Shadow_Public_Dataset --workspace data/_staging_render
```

## Mask Normalization Utilities

Split-based normalization (`train`, `val`):

```powershell
python3.10 tools/fix_masks_to_01.py
```

Flat private folder normalization:

```powershell
python3.10 tools/fix_private_masks_to_01.py
```

## Metrics

`tools/eval_shadow.py` reports:

- `IoU_background`: Intersection over Union for class `background`
- `IoU_shadow`: Intersection over Union for class `shadow`
- `mIoU`: mean Intersection over Union across classes
- `Acc_background`: per-class pixel accuracy for `background`
- `Acc_shadow`: per-class pixel accuracy for `shadow`
- `mAcc`: mean per-class pixel accuracy
- `aAcc`: all-pixel (global) accuracy
- `public_proxy_score`: `0.6 * mIoU + 0.4 * harmonic(IoU_background, IoU_shadow)`

## Validation Commands

Static check:

```powershell
python3.10 -m compileall run_infer.py tools rovershadow configs
```

## GitHub Preparation

This repo is set up to:

- ignore generated experiment folders (`work_dirs/`) and staging/archive folders under `data/`
- include canonical datasets under `data/public/` and `data/private/`
- include curated model artifact(s) under `artifacts/`

Large files in this project (datasets and `.pth` checkpoints) should be pushed with Git LFS. This repo includes `.gitattributes` rules for that.

One-time local setup:

```powershell
git lfs install
```

Recommended pre-push checks:

```powershell
python3.10 -m compileall run_infer.py tools rovershadow configs
python3.10 tools/verify_dataset_integrity.py --public-root data/public/Rover_Shadow_Public_Dataset --workspace data/_staging_render
```

Example Git bootstrap (source + datasets + best artifact):

```powershell
git init
git add .gitignore .gitattributes README.md requirements-cpu.txt requirements-gpu-cu130.txt run_infer.py configs rovershadow tools artifacts data/public data/private
git commit -m "Initial RoverShadow source commit"
```

## Notes

- `mmcv-lite` is used in this project; missing compiled ops are handled by `rovershadow/runtime/mmcv_ops_shim.py`.
- `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` is set by runtime helpers for compatibility with MMEngine checkpoint loading behavior.
