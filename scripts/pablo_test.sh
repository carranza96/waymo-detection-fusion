MODEL=faster_rcnn_r50_fpn_fp16_2x1_6e_waymo_open_1280x1920
#MODEL=faster_rcnn_r50_c4_fp16_2x1_6e_waymo_open_1280x1920

CHECKPOINT_DIR=FRCNN-FPN-normal_001
#CHECKPOINT_DIR=FRCNN-FPN-learn-anchors_002
#CHECKPOINT_DIR=FRCNN-C4-normal_001

python tools/test.py saved_models/${CHECKPOINT_DIR}/${MODEL}/${MODEL}.py \
    saved_models/${CHECKPOINT_DIR}/${MODEL}/latest.pth \
    --eval bbox \
    --out saved_models/${CHECKPOINT_DIR}/${MODEL}/results.pkl  \
    --eval-options "classwise=True" "outfile_prefix=saved_models/"${CHECKPOINT_DIR}"/"${MODEL}/"predictions_waymo" \
#    --show-dir saved_models/${MODEL}/predictions   # Save images with predictions
#    --format-only \
#    --eval-options "outfile_prefix=./faster_rcnn_res" "format_type=waymo"

    #--show