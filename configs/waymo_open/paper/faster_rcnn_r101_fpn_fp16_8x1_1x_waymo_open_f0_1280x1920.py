_base_ = [
    '../../_base_/models/faster_rcnn_r50_fpn.py',
    '../../_base_/datasets/waymo_detection_1280x1920.py',
    '../../_base_/schedules/schedule_1x.py', '../../_base_/default_runtime.py'
]
# model
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101),
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
    step=[2, 4])
total_epochs = 6

# load_from = 'http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r101_fpn_2x_coco/faster_rcnn_r101_fpn_2x_coco_bbox_mAP-0.398_20200504_210455-1d2dac9c.pth'
resume_from = 'saved_models/paper/faster_rcnn_r101_fpn_fp16_8x1_1x_waymo_open_f0_1280x1920/latest.pth'
# fp16 settings
fp16 = dict(loss_scale=512.)
