MODEL=faster_rcnn_r50_fpn_fp16_4x2_2x_1280x1920


python tools/test.py configs/waymo_pablo/${MODEL}/${MODEL}.py \
    saved_models/study/${MODEL}/latest.pth \
    --eval bbox \
    --out saved_models/study/${MODEL}/results.pkl  \
    --eval-options "classwise=True" "outfile_prefix=saved_models/"${MODEL}/"predictions_waymo" \
#    --show-dir saved_models/${MODEL}/predictions   # Save images with predictions
#    --format-only \
#    --eval-options "outfile_prefix=./faster_rcnn_res" "format_type=waymo"

    #--show
