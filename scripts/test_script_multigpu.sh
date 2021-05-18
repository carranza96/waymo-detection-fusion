MODEL=yolof_r50_c5_fp16_4x2_1x_1280x1920


./tools/dist_test.sh saved_models/study/${MODEL}/${MODEL}_full_test.py \
    saved_models/study/${MODEL}/epoch_12.pth 2 \
    --eval bbox \
    --out saved_models/study/${MODEL}/results_full_val.pkl  \
    --eval-options "classwise=True" "outfile_prefix=saved_models/study/"${MODEL}/"predictions_waymo_full_val"