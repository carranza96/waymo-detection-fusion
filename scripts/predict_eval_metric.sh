declare -a StringArray=("paper/res2net_1280x1920")
for MODEL in ${StringArray[@]}; do



CUDA_VISIBLE_DEVICES="0" python tools/test.py configs/waymo_open/paper/${MODEL}.py \
    saved_models/paper/${MODEL}/latest.pth \
    --eval bbox \
    --out saved_models/paper/${MODEL}/predictions_waymo.bbox.pkl  \
    --eval-options "classwise=True" "outfile_prefix=saved_models/"${MODEL}/"predictions_waymo" \


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
done
