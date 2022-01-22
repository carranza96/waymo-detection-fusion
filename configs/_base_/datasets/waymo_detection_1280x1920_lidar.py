dataset_type = 'WaymoOpenDataset'
data_root = '/home/manuel/Escritorio/mmdetection/data/waymococo_f0/'
# data_root = '/home/javgal/mmdetection/data/waymococo_f0/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53,  15.892], std=[58.395, 57.12, 57.375, 13.732], to_rgb=True)
train_pipeline = [
    dict(type='LoadMultiChannelImageFromFiles'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1280, 1920), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadMultiChannelImageFromFiles'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1280, 1920),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2020_lidar.json',
        img_prefix=data_root + 'train2020/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2020_lidar.json',
        img_prefix=data_root + 'val2020/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2020_lidar.json',
        img_prefix=data_root + 'val2020/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')    #mAP
