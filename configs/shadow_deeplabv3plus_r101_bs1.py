"""DeepLabV3+ R101 candidate with reduced batch size for stability."""

_base_ = ["./shadow_deeplabv3plus_r101.py"]

train_dataloader = dict(batch_size=1)

work_dir = "work_dirs/shadow_deeplabv3plus_r101_bs1"
