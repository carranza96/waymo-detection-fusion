MODEL=faster_rcnn_r50_c4_fp16_8x1_7e_waymo_open_f0

python tools/browse_dataset.py configs/waymo_open/${MODEL}.py
#[-h] [--skip-type ${SKIP_TYPE[SKIP_TYPE...]}] [--output-dir ${OUTPUT_DIR}] [--not-show] [--show-interval ${SHOW_INTERVAL}]