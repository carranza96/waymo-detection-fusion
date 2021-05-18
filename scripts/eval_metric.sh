MODEL=study/yolof_r50_c5_fp16_4x2_1x_1280x1920


# Format-only
#python tools/eval_metric.py configs/waymo_open/${MODEL}.py \
#saved_models/${MODEL}/results.pkl \
#--format-only \


# Eval
#python tools/analysis_tools/eval_metric.py configs/waymo_open/${MODEL}.py \
#saved_models/${MODEL}/results.pkl \
#--eval bbox \
#--eval-options "classwise=True" "outfile_prefix=saved_models/"${MODEL}/"predictions_waymo"
#
#
#
PREDICTIONS_FILE=saved_models/${MODEL}/predictions_waymo_full_val.bin
GTS_FILE=saved_models/${MODEL}/predictions_waymo_full_val_gt.bin
METRICS_FILE=saved_models/${MODEL}/metrics.csv
#
cd waymo-open-dataset
bazel build waymo_open_dataset/metrics/tools/compute_detection_metrics_main
cd ..

waymo-open-dataset/bazel-bin/waymo_open_dataset/metrics/tools/compute_detection_metrics_main \
${PREDICTIONS_FILE} ${GTS_FILE} > ${METRICS_FILE}


python tools/parse_waymo_metrics.py --metrics_file=${METRICS_FILE}
