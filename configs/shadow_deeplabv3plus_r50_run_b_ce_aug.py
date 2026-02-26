"""Phase-1 Run B: DeepLabV3+ R50 with CE-only and strong augmentations."""

_base_ = ["./shadow_deeplabv3plus_r50_run_a_ce.py"]

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(
        type="RandomResize",
        scale=(1024, 1024),
        ratio_range=(0.5, 2.0),
        keep_ratio=True,
    ),
    dict(type="RandomCrop", crop_size=(512, 512), cat_max_ratio=0.85),
    dict(type="RandomFlip", prob=0.5),
    dict(type="RandomRotate", prob=0.3, degree=20),
    dict(
        type="RandomApply",
        prob=0.8,
        transforms=[dict(type="PhotoMetricDistortion")],
    ),
    dict(
        type="RandomChoice",
        transforms=[
            [
                dict(
                    type="RandomApply",
                    prob=0.5,
                    transforms=[
                        dict(type="CLAHE", clip_limit=2.0, tile_grid_size=(8, 8))
                    ],
                )
            ],
            [
                dict(
                    type="RandomApply",
                    prob=0.5,
                    transforms=[dict(type="AdjustGamma", gamma=0.7)],
                )
            ],
            [
                dict(
                    type="RandomApply",
                    prob=0.3,
                    transforms=[dict(type="AdjustGamma", gamma=1.5)],
                )
            ],
        ],
    ),
    dict(
        type="RandomCutOut",
        prob=0.15,
        n_holes=(1, 2),
        cutout_ratio=[(0.05, 0.05), (0.1, 0.1)],
        seg_fill_in=255,
    ),
    dict(type="PackSegInputs"),
]

train_dataloader = dict(
    dataset=dict(
        pipeline=train_pipeline,
    ),
)

work_dir = "work_dirs/shadow_deeplabv3plus_r50_run_b_ce_aug"
