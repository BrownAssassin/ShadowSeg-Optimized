# Best Private Model Artifact

This folder contains the current best-performing private-eval checkpoint and its paired metrics.

- Config: `configs/shadow_deeplabv3plus_r50.py`
- Checkpoint: `iter_11000.pth`
- Public metrics file: `metrics_public_val_iter11000_tta_none.json`
- Private metrics file (default inference): `metrics_private_iter11000_tta_none.json`
- Private metrics file (TTA + threshold): `metrics_private_iter11000_tta_flipms_thr060.json`

Checkpoint source run (local historical path before pruning old runs):
`work_dirs/shadow_deeplabv3plus_r50_render_expanded_12000/iter_11000.pth`

Recommended inference command:

```powershell
python3.10 run_infer.py --img data/public/Rover_Shadow_Public_Dataset/ShadowImages/val/lssd4000.jpg --cfg configs/shadow_deeplabv3plus_r50.py --ckpt artifacts/best_private_model/iter_11000.pth --device cuda --out outputs/demo_result.png
```
