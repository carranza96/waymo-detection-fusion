_base_ = [
    '../../_base_/models/retinanet_r50_fpn.py',
    '../../_base_/datasets/waymo_detection_1280x1920.py',
    '../../_base_/schedules/schedule_1x.py', '../../_base_/default_runtime.py'
]

model = dict(pretrained='open-mmlab://resnext101_32x4d',
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch'),
        bbox_head=dict(num_classes=3))

data = dict(samples_per_gpu=2)


optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[2])
runner = dict(type='EpochBasedRunner', max_epochs=3)

load_from = 'http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_x101_32x4d_fpn_2x_coco/retinanet_x101_32x4d_fpn_2x_coco_20200131-237fc5e1.pth'
# fp16 settings
fp16 = dict(loss_scale=512.)