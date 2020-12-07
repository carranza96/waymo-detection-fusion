MODEL=faster_rcnn_r50_c4_fp16_8x1_7e_waymo_open_f0

python tools/benchmark.py configs/waymo_open/${MODEL}.py \
       saved_models/${MODEL}/latest.pth \
