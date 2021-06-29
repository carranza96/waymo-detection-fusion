data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type='WaymoOpenDataset',
        ann_file='data/waymococo_f0/annotations/instances_train2020.json',
        img_prefix='data/waymococo_f0/train2020/',
        # ann_file='data/waymococo_f0/annotations/instances_val2020_example.json',
        # img_prefix='data/waymococo_f0/val2020/',
        # ann_file='data/waymococo_f0/annotations/instances_val2020_sample2000.json',
        # img_prefix='data/waymococo_f0/val2020/',
        # filter_empty_gt=False,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', img_scale=(1280, 1920), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ]),
    # train=dict(
    #     type='RepeatDataset',
    #     times=1000,
    #     dataset=dict(
    #     type='WaymoOpenDataset',
    #     # ann_file='data/waymococo_f0/annotations/instances_train2020.json',
    #     # img_prefix='data/waymococo_f0/train2020/',
    #     ann_file='data/waymococo_f0/annotations/instances_val2020_example.json',
    #     img_prefix='data/waymococo_f0/val2020/',
    #     # filter_empty_gt=False,
    #     pipeline=[
    #         dict(type='LoadImageFromFile'),
    #         dict(type='LoadAnnotations', with_bbox=True),
    #         dict(type='Resize', img_scale=(1280, 1920), keep_ratio=True),
    #         dict(type='RandomFlip', flip_ratio=0.),
    #         dict(
    #             type='Normalize',
    #             mean=[123.675, 116.28, 103.53],
    #             std=[58.395, 57.12, 57.375],
    #             to_rgb=True),
    #         dict(type='Pad', size_divisor=32),
    #         dict(type='DefaultFormatBundle'),
    #         dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    #     ])),
    val=dict(
        type='WaymoOpenDataset',
        # ann_file='data/waymococo_f0/annotations/instances_val2020.json',
        ann_file='data/waymococo_f0/annotations/instances_val2020_sample2000.json',
        # ann_file='data/waymococo_f0/annotations/instances_val2020_example.json',
        img_prefix='data/waymococo_f0/val2020/',
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1280, 1920),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='WaymoOpenDataset',
        # ann_file='data/waymococo_f0/annotations/instances_train2020.json',
        # img_prefix='data/waymococo_f0/train2020/',
        ann_file='data/waymococo_f0/annotations/instances_val2020_sample2000.json',
        # ann_file='data/waymococo_f0/annotations/instances_val2020.json',
        # ann_file='data/waymococo_f0/annotations/instances_val2020_example.json',
        img_prefix='data/waymococo_f0/val2020/',
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1280, 1920),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
seed = 1
evaluation = dict(interval=1, metric='bbox')
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[2, 4])
runner = dict(type='EpochBasedRunner', max_epochs=5)
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=1,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
fp16 = dict(loss_scale=512.0)
work_dir = 'ensemble/'
gpu_ids = range(0, 1)