MODEL=faster_rcnn_r50_fpn_fp16_2x1_6e_waymo_open_1280x1920

python tools/analysis_tools/benchmark.py configs/waymo_pablo/${MODEL}.py \
       saved_models/modelo-normal/${MODEL}/latest.pth
