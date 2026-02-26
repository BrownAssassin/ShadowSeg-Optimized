"""DeepLabV3+ R50 variant: Tversky(0.8, 0.2) + FP loss weight 0.10."""

_base_ = ["./shadow_deeplabv3plus_r50.py"]

model = dict(
    decode_head=dict(
        loss_decode=[
            dict(
                type="SafeCrossEntropyLoss",
                use_sigmoid=False,
                class_weight=[1.3, 1.0],
                avg_non_ignore=True,
                loss_weight=1.0,
            ),
            dict(
                type="TverskyLoss",
                alpha=0.8,
                beta=0.2,
                loss_weight=0.4,
            ),
            dict(
                type="ShadowFalsePositiveLoss",
                shadow_class=1,
                ignore_index=255,
                loss_weight=0.10,
            ),
        ]
    )
)
