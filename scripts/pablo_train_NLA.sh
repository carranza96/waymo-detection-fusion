#MODEL=faster_rcnn_r50_fpn_fp16_8x1_1x_waymo_open_1280x1920
MODEL=faster_rcnn_r50_c4_fp16_2x1_6e_waymo_open_1280x1920_NLA

echo "START TRAIN"

python tools/train.py configs/waymo_pablo/${MODEL}.py \
    --work-dir=saved_models/waymo_pablo/${MODEL}

echo "END TRAIN"
