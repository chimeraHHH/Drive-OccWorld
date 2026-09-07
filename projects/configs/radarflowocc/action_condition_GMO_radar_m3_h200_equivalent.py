_base_ = ['./action_condition_GMO_radar_m3.py']

# Protocol template. Resolve container-visible data, checkpoint and output
# paths with tools/m3_prepare_h200_config.py before launching this experiment.
data = dict(
    samples_per_gpu=1, workers_per_gpu=2,
    shuffler_sampler=dict(type='VirtualDistributedGroupSampler', reference_world_size=4))
optimizer = dict(
    type='AdamW', lr=1e-4, weight_decay=0.01,
    paramwise_cfg=dict(custom_keys=dict(img_backbone=dict(lr_mult=0.1))))
optimizer_config = dict(
    type='EquivalentCumulativeOptimizerHook', cumulative_iters=2,
    reference_world_size=4, expected_optimizer_steps_per_epoch=5983,
    grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='EquivalentCosineAnnealing', cumulative_iters=2, by_epoch=True,
    warmup='linear', warmup_iters=500, warmup_ratio=1.0 / 3.0,
    min_lr_ratio=1e-3)
runner = dict(type='EpochBasedRunner', max_epochs=24)
total_epochs = 24
seed = 0
gpu_ids = range(2)
fp16 = None
tf32_policy = dict(matmul=True, cudnn=True)
close_tf32 = False
cudnn_benchmark = False
log_config = dict(interval=20)  # raw microsteps: ten optimizer updates
checkpoint_config = dict(interval=1, max_keep_ckpts=1)
load_from = 'pretrained/r101_dcn_fcos3d_pretrain.pth'
resume_from = None
m3_h200_requires_resolution = True
