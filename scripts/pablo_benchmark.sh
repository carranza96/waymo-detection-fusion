MODEL=faster_rcnn_r50_fpn_fp16_2x1_6e_waymo_open_1280x1920
#MODEL=faster_rcnn_r50_c4_fp16_2x1_6e_waymo_open_1280x1920

CHECKPOINT_DIR=FRCNN-FPN-normal_001
#CHECKPOINT_DIR=FRCNN-FPN-learn-anchors_002
#CHECKPOINT_DIR=FRCNN-C4-normal_001

python tools/analysis_tools/benchmark.py configs/waymo_pablo/${MODEL}.py \
       saved_models/${CHECKPOINT_DIR}/${MODEL}/latest.pth
