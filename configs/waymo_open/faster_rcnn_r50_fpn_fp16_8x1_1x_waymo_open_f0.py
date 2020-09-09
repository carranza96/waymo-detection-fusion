_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/waymo_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# model
model = dict(roi_head=dict(bbox_head=dict(num_classes=3)))
# data
data = dict(samples_per_gpu=8)


# lr is set for a batch size of 8
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)


load_from = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'  # noqa
resume_from = 'faster_rcnn_r50_fpn_fp16_8x1_1x_waymo_open_f0/epoch_6.pth'

# fp16 settings
fp16 = dict(loss_scale=512.)