"""Legacy FCN-R50 baseline config retained for regression and comparisons."""

_base_ = [
    "mmseg::_base_/models/fcn_r50-d8.py",
    "mmseg::_base_/default_runtime.py",
    "mmseg::_base_/schedules/schedule_20k.py",
]

num_classes = 2
metainfo = dict(
    classes=("background", "shadow"),
    palette=[(0, 0, 0), (255, 255, 255)],
)

# Keep preprocessing behavior explicit for stable inference dimensions.
data_preprocessor = dict(
    type="SegDataPreProcessor",
    bgr_to_rgb=True,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    pad_val=0,
    seg_pad_val=255,
    size=(512, 512),
    size_divisor=None,
)

model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=num_classes),
    auxiliary_head=dict(num_classes=num_classes),
)

dataset_type = "BaseSegDataset"
data_root = "data/public/Rover_Shadow_Public_Dataset"

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="RandomResize", scale=(1024, 1024), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type="RandomCrop", crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PackSegInputs"),
]

val_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img_path="ShadowImages/train", seg_map_path="ShadowMasks/train"),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img_path="ShadowImages/val", seg_map_path="ShadowMasks/val"),
        pipeline=val_pipeline,
    ),
)

test_dataloader = val_dataloader

val_evaluator = dict(type="IoUMetric", iou_metrics=["mIoU"])
test_evaluator = val_evaluator
test_pipeline = val_pipeline
