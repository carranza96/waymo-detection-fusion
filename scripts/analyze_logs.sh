MODEL=faster_rcnn_r50_fpn_fp16_8x1_1x_waymo_open_f0
LOG_FILE=$(ls saved_models/${MODEL}/*.log.json | tail -n 1)


MODEL2=faster_rcnn_r50_fpn_fp16_8x1_7e_waymo_open_f0
LOG_FILE2=$(ls saved_models/${MODEL2}/*.log.json | tail -n 1)

# Single model
#python tools/analyze_logs.py plot_curve ${LOG_FILE} --keys loss_cls --legend loss_cls
#python tools/analyze_logs.py plot_curve ${LOG_FILE} --keys acc --legend acc
#python tools/analyze_logs.py plot_curve ${LOG_FILE} --keys loss_cls loss_bbox --legend loss_cls loss_bbox
#python tools/analyze_logs.py plot_curve ${LOG_FILE} --keys bbox_mAP --legend mAP


# Compare two models
#python tools/analyze_logs.py plot_curve ${LOG_FILE} ${LOG_FILE2} --keys bbox_mAP --legend run1 run2

# Training time
#python tools/analyze_logs.py cal_train_time ${LOG_FILE}

# FLOPS
python tools/get_flops.py saved_models/${MODEL}/${MODEL}.py --shape 1248 832


