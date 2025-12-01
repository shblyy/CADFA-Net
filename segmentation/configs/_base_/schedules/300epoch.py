# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', clip_grad=None)
# optim_wrapper = dict(
#     _delete_=True,
#     type='OptimWrapper',
#     clip_grad=dict(max_norm=0.1, norm_type=2),
#     optimizer=dict(
#         type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05),
#     paramwise_cfg=dict(
#         custom_keys={
#             'absolute_pos_embed': dict(decay_mult=0.),
#             'relative_position_bias_table': dict(decay_mult=0.),
#             'norm': dict(decay_mult=0.)
#         }))
# learning policy (scheduler active for first 100 epochs)
# param_scheduler = [
#     dict(
#         type='PolyLR',
#         eta_min=1e-4,
#         power=0.9,
#         begin=0,
#         end=100,
#         by_epoch=True)  # Change scheduler to work by epoch
# ]

# param_scheduler = [
#     dict(
#         type='LinearLR',
#         start_factor=1e-6,
#         by_epoch=True,
#         begin=0,
#         end=20  # 设置为 20 个 epoch 的热身
#     ),
#     dict(
#         type='PolyLR',
#         eta_min=1e-5,  # 假设最小学习率为 1e-5
#         power=1.0,
#         begin=30,      # 热身结束后开始衰减
#         end=300,       # 总训练轮数为 300
#         by_epoch=True
#     )
# ]


# param_scheduler = [
#     dict(
#         type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
#     dict(
#         type='PolyLR',
#         eta_min=0.0,
#         power=1.0,
#         begin=1500,
#         end=54000,
#         by_epoch=True,
#     )
# ]

# training schedule for 300 epochs
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=300, val_interval=5  # Adjusted for epoch-based scheduling
)
val_cfg = dict(type='ValLoop')  # Ensure validation loop is set up correctly
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100, log_metric_by_epoch=True),  # 按epoch记录日志，每个epoch记录一次
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=True,  # 按epoch保存
        interval=5,  # 每个epoch保存一次
        save_best='mIoU',
        max_keep_ckpts=1
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook')
)



# # optimizer
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
# optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
# # learning policy
# param_scheduler = [
#     dict(
#         type='PolyLR',
#         eta_min=1e-4,
#         power=0.9,
#         begin=0,
#         end=160000,
#         by_epoch=False)
# ]
# # training schedule for 160k
# train_cfg = dict(
#     type='IterBasedTrainLoop', max_iters=54000, val_interval=180)
# val_cfg = dict(type='ValLoop')
# test_cfg = dict(type='TestLoop')
# # default_hooks = dict(
# #     timer=dict(type='IterTimerHook'),
# #     logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
# #     param_scheduler=dict(type='ParamSchedulerHook'),
# #     checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=16000),
# #     sampler_seed=dict(type='DistSamplerSeedHook'),
# #     visualization=dict(type='SegVisualizationHook'))
#
# default_hooks = dict(
#     timer=dict(type='IterTimerHook'),
#     logger=dict(type='LoggerHook', interval=30, log_metric_by_epoch=False),
#     param_scheduler=dict(type='ParamSchedulerHook'),
#     checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=180,save_best='mIoU',
#         max_keep_ckpts=1,),
#     sampler_seed=dict(type='DistSamplerSeedHook'),
#     visualization=dict(type='SegVisualizationHook'))

