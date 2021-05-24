_base_ = [
    '../../_base_/models/faster_rcnn_r50_fpn.py',
    '../../_base_/datasets/waymo_detection_1280x1920.py',
    '../../_base_/schedules/schedule_1x.py',
    '../../_base_/default_runtime.py'
]

# model
model = dict(roi_head=dict(bbox_head=dict(num_classes=3)),
             test_cfg=dict(
                 rcnn=dict(
                     score_thr=0.05,
                     nms=dict(type='nms', iou_threshold=0.5),
                     max_per_img=100)
             )
        )

dataset_type = 'WaymoOpenDataset'
data_root = '/media/hd2/WaymoCOCO/data/waymococo_full/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=[(640, 960), (1280, 1920)], multiscale_mode='range', keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
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
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=[
        dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/instances_train2020.json',
            img_prefix=data_root + 'train2020/',
            pipeline=train_pipeline),
        dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/instances_val2020.json',
            img_prefix=data_root + 'val2020/',
            pipeline=train_pipeline)

    ],
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2020.json',
        img_prefix=data_root + 'val2020/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2020.json',
        img_prefix=data_root + 'val2020/',
        pipeline=test_pipeline))


# lr is set for a batch size of 8
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

# Learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[1, 2])
runner = dict(type='EpochBasedRunner', max_epochs=3)

# load_from = 'saved_models/study/faster_rcnn_r50_fpn_fp16_4x2_1x_1280x1920/epoch_12.pth'
resume_from = 'saved_models/study/faster_rcnn_r50_fpn_fp16_4x2_1x_1280x1920_final/epoch_1.pth'
# fp16 settings
fp16 = dict(loss_scale=512.)