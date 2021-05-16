MODEL=faster_rcnn_r50_fpn_fp16_2x1_6e_waymo_open_1280x1920
#MODEL=faster_rcnn_r50_c4_fp16_2x1_6e_waymo_open_1280x1920

CHECKPOINT_DIR=FRCNN-FPN-normal_001
#CHECKPOINT_DIR=FRCNN-FPN-learn-anchors_002
#CHECKPOINT_DIR=FRCNN-C4-normal_001

# Format-only
#python tools/eval_metric.py configs/waymo_open/${MODEL}.py \
#       saved_models/${MODEL}/results.pkl \
#       --format-only \

# Eval
python tools/analysis_tools/eval_metric.py configs/waymo_pablo/${MODEL}.py \
        saved_models/${CHECKPOINT_DIR}/${MODEL}/results.pkl \
        --eval bbox \
        --eval-options "classwise=True" "outfile_prefix=saved_models/"${CHECKPOINT_DIR}"/"${MODEL}/"predictions_waymo"

# PREDICTIONS_FILE=saved_models/waymo_pablo/${MODEL}/predictions_waymo.bin
# GTS_FILE=saved_models/waymo_pablo/${MODEL}/predictions_waymo_gt.bin
# METRICS_FILE=saved_models/waymo_pablo/${MODEL}/metrics.csv

# cd waymo-open-dataset
# bazel build waymo_open_dataset/metrics/tools/compute_detection_metrics_main
# cd ..

# waymo-open-dataset/bazel-bin/waymo_open_dataset/metrics/tools/compute_detection_metrics_main \
# ${PREDICTIONS_FILE} ${GTS_FILE} > ${METRICS_FILE}

# python tools/parse_waymo_metrics.py --metrics_file=${METRICS_FILE}
