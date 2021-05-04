_base_ = [
    '../_base_/models/faster_rcnn_r50_caffe_dc5.py',
    '../_base_/datasets/waymo_detection_1280x1920.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
# model
model = dict(
    roi_head=dict(bbox_head=dict(num_classes=3)),
    train_cfg=dict(
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
        )
    )
)
# data
data = dict(samples_per_gpu=2)

dataset_type = 'WaymoOpenDataset'
data_root = '../waymococo_f0/'

# use caffe img_norm
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1280, 1920), keep_ratio=True),
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
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2020.json',
        img_prefix=data_root + 'train2020/',
        pipeline=train_pipeline),
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


# LR is set for a batch size of 8
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

# Learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[3, 5])
total_epochs = 6
runner = dict(type='EpochBasedRunner', max_epochs=6)


load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_caffe_dc5_1x_coco/faster_rcnn_r50_caffe_dc5_1x_coco_20201030_151909-531f0f43.pth'
# fp16 settings
fp16 = dict(loss_scale=512.)