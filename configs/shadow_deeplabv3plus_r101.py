"""DeepLabV3+ R101 candidate config for RoverShadow."""

_base_ = ["./shadow_deeplabv3plus_r50.py"]

model = dict(
    pretrained="open-mmlab://resnet101_v1c",
    backbone=dict(depth=101),
)

work_dir = "work_dirs/shadow_deeplabv3plus_r101"
