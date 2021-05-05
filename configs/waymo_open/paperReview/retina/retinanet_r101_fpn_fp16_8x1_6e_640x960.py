_base_ = [
    '../../_base_/models/retinanet_r50_fpn.cpy',
    '../../_base_/datasets/waymo_detection_640x960.py',
    '../../_base_/schedules/schedule_1x.py', '../../_base_/default_runtime.py'
]

model = dict(pretrained='torchvision://resnet101',
             backbone=dict(depth=101),
             bbox_head=dict(num_classes=3))

data = dict(samples_per_gpu=8)


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

load_from = 'http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r101_fpn_2x_coco/retinanet_r101_fpn_2x_coco_20200131-5560aee8.pth'
# fp16 settings
fp16 = dict(loss_scale=512.)