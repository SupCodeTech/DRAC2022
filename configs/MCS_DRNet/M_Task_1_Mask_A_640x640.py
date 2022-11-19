_base_ = [
    '../_base_/models/upernet_mae.py', '../_base_/datasets/DRAC2022_Task_1_Mask_A_640x640.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_130k.py'
]

model = dict(
    pretrained='./Se_sup/work_dirs/mae/pretrain_backbone_4800.pth',
    backbone=dict(
        type='MAE',
        img_size=(640, 640),
        patch_size=16,
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4,
        init_values=1.0,
        drop_path_rate=0.1,
        out_indices=[3, 5, 7, 11]),
    neck=dict(embed_dim=768, rescales=[4, 2, 1, 0.5]),
    decode_head=dict(
        in_channels=[768, 768, 768, 768], num_classes=3, channels=768),
    auxiliary_head=dict(in_channels=768, num_classes=3),
    test_cfg=dict(mode='slide', crop_size=(640, 640), stride=(341, 341)))

optimizer = dict(
     _delete_=True,
     type='AdamW',
     lr=1e-5,
     betas=(0.9, 0.999),
     weight_decay=0.05,
     constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.65))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1e-5,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)


# mixed precision
fp16 = dict(loss_scale='dynamic')


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 640),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True, min_size=512),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline),
    samples_per_gpu=2)
