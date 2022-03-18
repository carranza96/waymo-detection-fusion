MODEL=lidar/faster_rcnn_r50_fpn_fp16_2x2_1x_1280x1920_enet_git_early_pretrained_epoch12


# Eval
python tools/analysis_tools/eval_metric.py saved_models/lidar/faster_rcnn_r50_fpn_fp16_2x2_1x_1280x1920_enet_git_early_pretrained_epoch12/faster_rcnn_r50_fpn_fp16_4x2_1x_1280x1920_enet.py \
saved_models/${MODEL}/results_full_val.pkl \
--eval bbox \
--eval-options "classwise=True" "outfile_prefix=saved_models/"${MODEL}/"predictions_waymo" "time_of_day=Day"