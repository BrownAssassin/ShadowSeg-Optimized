"""Phase-1 Run A: DeepLabV3+ R50 with CE-only and light augmentations."""

_base_ = [
    "mmseg::_base_/models/deeplabv3plus_r50-d8.py",
    "mmseg::_base_/default_runtime.py",
    "mmseg::_base_/schedules/schedule_20k.py",
]

custom_imports = dict(imports=["rovershadow.losses"], allow_failed_imports=False)

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
    decode_head=dict(
        num_classes=num_classes,
        loss_decode=dict(
            type="SafeCrossEntropyLoss",
            use_sigmoid=False,
            class_weight=[1.3, 1.0],
            avg_non_ignore=True,
            loss_weight=1.0,
        ),
    ),
    auxiliary_head=dict(
        num_classes=num_classes,
        loss_decode=dict(
            type="SafeCrossEntropyLoss",
            use_sigmoid=False,
            class_weight=[1.3, 1.0],
            avg_non_ignore=True,
            loss_weight=0.4,
        ),
    ),
)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(
        type="RandomResize",
        scale=(1024, 1024),
        ratio_range=(0.5, 2.0),
        keep_ratio=True,
    ),
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
    type="OptimWrapper",
    optimizer=dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=5e-4),
    clip_grad=None,
)

param_scheduler = [
    dict(
        type="PolyLR",
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=2000,
        by_epoch=False,
    )
]

train_cfg = dict(type="IterBasedTrainLoop", max_iters=2000, val_interval=1000)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

default_hooks = dict(
    checkpoint=dict(type="CheckpointHook", by_epoch=False, interval=1000, max_keep_ckpts=3),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=False),
)

randomness = dict(seed=42, deterministic=False)
work_dir = "work_dirs/shadow_deeplabv3plus_r50_run_a_ce"
