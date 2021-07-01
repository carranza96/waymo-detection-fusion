MODEL=faster_rcnn_r50_fpn_modassign_fp16_4x2_1x_1280x1920


./tools/dist_test.sh saved_models/study/${MODEL}/${MODEL}.py \
    saved_models/study/${MODEL}/epoch_12.pth 2 \
    --eval bbox \
    --out saved_models/study/${MODEL}/results.pkl  \
    --eval-options "classwise=True" "outfile_prefix=saved_models/study/"${MODEL}/"predictions_waymo"