
echo "START TRAIN 1"

MODEL=faster_rcnn_r50_c4_fp16_2x1_6e_waymo_open_1280x1920_NLA

python tools/train.py configs/waymo_pablo/${MODEL}.py \
    --work-dir=saved_models/waymo_pablo/${MODEL}

echo "START TRAIN 2"

MODEL=faster_rcnn_r50_c4_fp16_2x1_6e_waymo_open_1280x1920_LA

python tools/train.py configs/waymo_pablo/${MODEL}.py \
    --work-dir=saved_models/waymo_pablo/${MODEL}

echo "END TRAIN"
