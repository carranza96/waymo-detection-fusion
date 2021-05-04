MODEL=faster_rcnn_r50_fpn_fp16_8x1_1x_waymo_open_1280x1920
DIR=saved_models/modelo-normal/

python DetVisGUI/DetVisGUI.py \
${DIR}/${MODEL}/${MODEL}.py \
--det_file ${DIR}/${MODEL}/${MODEL}/results.pkl

# [--stage ${STAGE}] [--output ${SAVE_DIRECTORY}]