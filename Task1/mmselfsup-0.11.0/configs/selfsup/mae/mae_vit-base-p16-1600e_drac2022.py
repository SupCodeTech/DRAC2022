_base_ = [
    '../_base_/models/mae_vit-base-p16.py',
    '../_base_/datasets/DRAC22_MAE.py',
    '../_base_/schedules/adamw_coslr-200e_in1k.py',
    '../_base_/default_runtime.py',
]

# dataset
data = dict(samples_per_gpu=32, workers_per_gpu=8)

# optimizer
optimizer = dict(
    lr=1.5e-7,
    paramwise_options={
        'ln': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
        'pos_embed': dict(weight_decay=0.),
        'mask_token': dict(weight_decay=0.),
        'cls_token': dict(weight_decay=0.)
    })
optimizer_config = dict()

# learning policy
lr_config = dict(
    policy='StepFixCosineAnnealing',
    min_lr=0.0,
    warmup='linear',
    warmup_iters=40,
    warmup_ratio=1e-7,
    warmup_by_epoch=True,
    by_epoch=False)

# schedule
runner = dict(max_epochs=1600)

# runtime
checkpoint_config = dict(interval=400, max_keep_ckpts=4, out_dir='')
persistent_workers = True
log_config = dict(
    interval=50, hooks=[
        dict(type='TextLoggerHook'),
    ])
