# MODEL=paper/yolo_640x960

# configs/waymo_open/lidar/faster_rcnn_r50_fpn_fp16_4x2_1x_1280x1920.py

python tools/analysis_tools/benchmark.py configs/waymo_open/lidar/faster_rcnn_r50_fpn_fp16_4x2_1x_1280x1920_ipfdc.py \
       saved_models/lidar/faster_rcnn_r50_fpn_fp16_4x2_1x_1280x1920_ipfdc_early_pretrained_epoch12/latest.pth \
