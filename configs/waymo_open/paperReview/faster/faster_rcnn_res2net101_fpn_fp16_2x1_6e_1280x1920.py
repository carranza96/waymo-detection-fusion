_base_ = [
    '../../_base_/models/faster_rcnn_r50_fpn.py',
    '../../_base_/datasets/waymo_detection_1280x1920.py',
    '../../_base_/schedules/schedule_1x.py', '../../_base_/default_runtime.py'
]
# model
model = dict(pretrained='open-mmlab://res2net101_v1d_26w_4s',
            backbone=dict(type='Res2Net', depth=101, scales=4, base_width=26),
            roi_head=dict(bbox_head=dict(num_classes=3)))
# data
data = dict(samples_per_gpu=2)


# lr is set for a batch size of 8
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[3, 5])
runner = dict(type='EpochBasedRunner', max_epochs=6)

load_from = 'http://download.openmmlab.com/mmdetection/v2.0/res2net/faster_rcnn_r2_101_fpn_2x_coco/faster_rcnn_r2_101_fpn_2x_coco-175f1da6.pth'# fp16 settings
fp16 = dict(loss_scale=512.)

