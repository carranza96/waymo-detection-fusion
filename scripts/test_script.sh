MODEL=faster_rcnn_r50_fpn_fp16_8x1_1x_waymo_open_f0


python tools/test.py configs/waymo_open/${MODEL}.py \
    saved_models/${MODEL}/latest.pth \
    --eval bbox \
    --out saved_models/${MODEL}/predictions_waymo.bbox.pkl  \
    --eval-options "classwise=True" "outfile_prefix=saved_models/"${MODEL}/"predictions_waymo" \
#    --show-dir saved_models/${MODEL}/predictions   # Save images with predictions
#    --format-only \
#    --eval-options "outfile_prefix=./faster_rcnn_res" "format_type=waymo"

    #--show
