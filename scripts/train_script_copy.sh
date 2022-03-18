MODEL=faster_rcnn_r50_fpn_fp16_8x1_1x_waymo_open_f0_lidar

python tools/train.py /home/javgal/mmdetection/configs/waymo_open/lidar/faster_rcnn_r50_fpn_fp16_4x2_1x_1280x1920.py \
    --work-dir=saved_models/${MODEL}
    
# python tools//train.py checkpoints//faster_rcnn_r50_fpn_fp16_8x1_1x_waymo_open_f0.py --work-dir=saved_models//faster_rcnn_r50_fpn_fp16_8x1_1x_waymo_open_f0




#   cd /home/javgal/mmdetection ; /usr/bin/env /home/javgal/torch_penet/bin/python /home/javgal/.vscode-server/extensions/ms-python.python-2021.5.926500501/pythonFiles/lib/python/debugpy/launcher 43463 -- /home/javgal/mmdetection/tools//train.py /home/javgal/mmdetection/checkpoints//faster_rcnn_r50_fpn_fp16_8x1_1x_waymo_open_f0.py --work-dir=/home/javgal/mmdetection/saved_models//faster_rcnn_r50_fpn_fp16_8x1_1x_waymo_open_f0