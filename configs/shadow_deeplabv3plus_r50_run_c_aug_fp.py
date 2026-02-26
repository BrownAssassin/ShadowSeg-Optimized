"""Phase-1 Run C: DeepLabV3+ R50 with strong augments and FP-aware loss stack."""

_base_ = ["./shadow_deeplabv3plus_r50.py"]

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
default_hooks = dict(
    checkpoint=dict(type="CheckpointHook", by_epoch=False, interval=1000, max_keep_ckpts=3),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=False),
)

work_dir = "work_dirs/shadow_deeplabv3plus_r50_run_c_aug_fp"
