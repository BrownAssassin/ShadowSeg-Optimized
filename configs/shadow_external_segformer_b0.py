"""Fallback external model config: SegFormer-B0 for shadow segmentation."""

_base_ = [
    "mmseg::_base_/models/segformer_mit-b0.py",
    "mmseg::_base_/default_runtime.py",
    "mmseg::_base_/schedules/schedule_160k.py",
]

num_classes = 2
crop_size = (512, 512)
dataset_type = "BaseSegDataset"
data_root = "data/public/Rover_Shadow_Public_Dataset"

metainfo = dict(
    classes=("background", "shadow"),
    palette=[(0, 0, 0), (255, 255, 255)],
)

data_preprocessor = dict(
    type="SegDataPreProcessor",
    bgr_to_rgb=True,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
    size_divisor=None,
)

model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=num_classes),
)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="RandomResize", scale=(1024, 1024), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type="RandomCrop", crop_size=crop_size, cat_max_ratio=0.85),
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
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type="InfiniteSampler", shuffle=True),
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
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
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

optim_wrapper = dict(
    _delete_=True,
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=6e-5, betas=(0.9, 0.999), weight_decay=0.01),
)

param_scheduler = [
    dict(
        type="PolyLR",
        eta_min=1e-6,
        power=1.0,
        begin=0,
        end=12000,
        by_epoch=False,
    )
]

train_cfg = dict(type="IterBasedTrainLoop", max_iters=12000, val_interval=1000)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=False,
        interval=1000,
        max_keep_ckpts=6,
        save_best="mIoU",
        rule="greater",
    ),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=False),
)

randomness = dict(seed=42, deterministic=False)
work_dir = "work_dirs/shadow_external_segformer_b0"
