
_base_ = [
    '../_base_/models/faster_rcnn_r50_caffe_c4.py',
    '../_base_/datasets/waymo_detection_1280x1920.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# model
model = dict(
    rpn_head=dict(
        type='RPNHead_LA',
        anchor_generator=dict(
            type='AnchorGenerator_LA',
            # 4 GTS::
            # ids = [  22,   14,   12,   10]
            # Ideales: [1.503, 2.899, 5.193, 6.097], Alterados: [1.25, 3.50, 4.30, 7.30]
            scales=[0.70, 2.30, 5.50],
            ratios=[0.40, 0.70, 1.80],
            # scales=[0.9962409989425024,2.0676464608446268,5.959522203519279],
            # ratios=[0.5994572530916505,0.7306191964865241,1.1637696342029067],
            strides=[16])),
    roi_head=dict(bbox_head=dict(num_classes=3)),
    train_cfg=dict(
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
        )
    ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0
        )
    )
)

# dataset
dataset_type = 'WaymoOpenDataset'
data_root = 'data/waymococo_f0/' # Real dataset: '../waymococo_f0/'

# use caffe img_norm
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1280, 1920), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.), # Original: 0.5),
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
    samples_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=1000,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/instances_train2020_22gt.json',
            img_prefix=data_root + 'train2020/',
            pipeline=train_pipeline)),
    # train=dict( # Original
    #     type=dataset_type,
    #     ann_file=data_root + 'annotations/instances_train2020.json',
    #     img_prefix=data_root + 'train2020/',
    #     pipeline=train_pipeline),
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
# optimizer = dict(type='SGD', lr=0.075, momentum=0.9, weight_decay=0.0001) # Original: lr=0.0025
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001) # Original: lr=0.0025

optimizer_config = dict(grad_clip=None)

# # Learning policy
# lr_config = dict(
#     policy='step',
#     warmup='constant')

# # Original Learning policy:
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     step=[3, 5])

# New learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500, # batches
    warmup_ratio=0.001,
    step=[2, 4])


# # without warmup
# lr_config = dict(
#     policy='step',
#     warmup=None,    # <---      CAMBIO       <-----
#     warmup_iters=0) # batches

# Args:
#   by_epoch (bool): LR changes epoch by epoch
#   warmup (string): Type of warmup used. It can be None(use no warmup),
#       'constant', 'linear' or 'exp'
#   warmup_iters (int): The number of iterations or epochs that warmup
#       lasts
#   warmup_ratio (float): LR used at the beginning of warmup equals to
#       warmup_ratio * initial_lr
#   warmup_by_epoch (bool): When warmup_by_epoch == True, warmup_iters
#       means the number of epochs that warmup lasts, otherwise means the
#       number of iteration that warmup lasts
#
# Default:
#   by_epoch=True,
#   warmup=None,
#   warmup_iters=0,
#   warmup_ratio=0.1,
#   warmup_by_epoch=False

runner = dict(type='EpochBasedRunner', max_epochs=6)

# load_from = 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_r50_c4_2x-6e4fdf4f.pth'
# load_from = 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_r50_caffe_c4_2x-71c67f27.pth'
#load_from = 'saved_models/pretrained/faster_rcnn_r50_caffe_c4_2x-71c67f27_mod.pth' # <==
load_from = 'saved_models/pretrained/c4_1ar.pth'

# fp16 settings
fp16 = dict(loss_scale=512.)