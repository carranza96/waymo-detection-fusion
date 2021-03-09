MODEL=paper/faster_rcnn_r50_fpn_fp16_8x1_1x_waymo_open_1280x1920

python DetVisGUI/DetVisGUI.py \
configs/waymo_open/${MODEL}.py \
--det_file saved_models/${MODEL}/results.pkl

# [--stage ${STAGE}] [--output ${SAVE_DIRECTORY}]