_base_ = [
    '../../_base_/models/retinanet_r50_fpn.py',
    # '../../_base_/datasets/waymo_detection_1280x1920.py',
    '../../_base_/datasets/waymo_detection_1280x1920_lidar.py',
    '../../_base_/schedules/schedule_1x.py', '../../_base_/default_runtime.py'
]
# model
# model = dict(backbone=dict(in_channels=4),
model = dict(backbone=dict(in_channels=3),
             bbox_head=dict(
                 num_classes=3))
# data
data = dict(samples_per_gpu=4)


# lr is set for a batch size of 16
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)

# load_from = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'  # noqa
# resume_from = '/home/javgal/mmdetection/saved_models/lidar/faster_rcnn_r50_fpn_fp16_4x2_1x_1280x1920_enet/epoch_1.pth'

# fp16 settings
fp16 = dict(loss_scale=512.)
# resume_from = "saved_models/lidar/retinanet_r50_fpn_fp16_4x2_1x_1280x1920_ipfdc/epoch_4.pth"
# evaluation = dict(interval=12, metric='bbox')    #mAP
