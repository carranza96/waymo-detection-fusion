MODEL=faster_rcnn_r50_fpn_fp16_8x1_1x_waymo_open_f0


# Format-only
#python tools/eval_metric.py configs/waymo_open/${MODEL}.py \
#saved_models/${MODEL}/results.pkl \
#--format-only \


# Eval
python tools/eval_metric.py configs/waymo_open/${MODEL}.py \
saved_models/${MODEL}/results.pkl \
--eval bbox \
--eval-options "classwise=True" "outfile_prefix=saved_models/"${MODEL}/"predictions_waymo"
#
#
#
PREDICTIONS_FILE=saved_models/${MODEL}/predictions_waymo.bin
GTS_FILE=saved_models/${MODEL}/predictions_waymo_gt.bin
METRICS_FILE=saved_models/${MODEL}/metrics.csv
#
cd waymo-open-dataset
bazel build waymo_open_dataset/metrics/tools/compute_detection_metrics_main
cd ..

waymo-open-dataset/bazel-bin/waymo_open_dataset/metrics/tools/compute_detection_metrics_main \
${PREDICTIONS_FILE} ${GTS_FILE} > ${METRICS_FILE}


python tools/parse_waymo_metrics.py --metrics_file=${METRICS_FILE}
