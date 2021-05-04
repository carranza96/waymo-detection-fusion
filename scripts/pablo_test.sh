MODEL=faster_rcnn_r50_fpn_fp16_8x1_1x_waymo_open_1280x1920
DIR=saved_models/modelo-normal/


python tools/test.py ${DIR}/${MODEL}/${MODEL}.py \
    ${DIR}/${MODEL}/latest.pth \
    --eval bbox \
    --out ${DIR}/${MODEL}/results.pkl  \
    --eval-options "classwise=True" "outfile_prefix=saved_models/"${MODEL}/"predictions_waymo" \
#    --show-dir saved_models/${MODEL}/predictions   # Save images with predictions
#    --format-only \
#    --eval-options "outfile_prefix=./faster_rcnn_res" "format_type=waymo"

    #--show