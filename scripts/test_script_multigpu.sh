# MODEL=yolof_r50_c5_fp16_4x2_1x_1280x1920
MODEL=faster_rcnn_r50_fpn_fp16_2x2_1x_1280x1920_enet_git_early_pretrained_epoch12


./tools/dist_test.sh configs/waymo_open/lidar/faster_rcnn_r50_fpn_fp16_4x2_1x_1280x1920_enet.py \
    saved_models/lidar/${MODEL}/latest.pth 2 \
    --eval bbox \
    --out saved_models/lidar/${MODEL}/results_full_val.pkl  \
    --eval-options "classwise=True" "outfile_prefix=saved_models/lidar/"${MODEL}/"predictions_waymo_full_val"