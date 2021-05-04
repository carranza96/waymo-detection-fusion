MODEL=retinanet_r50_fpn_fp16_4x2_1x_1280x1920


./tools/dist_test.sh saved_models/study/${MODEL}/${MODEL}_training.py \
    saved_models/study/${MODEL}/latest.pth 2 \
    --eval bbox \
    --out saved_models/study/${MODEL}/results_training.pkl  \
    --eval-options "classwise=True" "outfile_prefix=saved_models/study/"${MODEL}/"predictions_waymo"