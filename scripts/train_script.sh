MODEL=faster_rcnn_r50_fpn_fp16_8x1_1x_waymo_open_f0

python tools/train.py configs/waymo_open/${MODEL}.py \
    --work-dir=saved_models/${MODEL}
