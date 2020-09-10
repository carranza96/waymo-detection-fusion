MODEL=faster_rcnn_r50_fpn_fp16_8x1_1x_waymo_open_f0.py


python tools/test.py configs/waymo_open/${MODEL} \
    configs/waymo_open/${MODEL}/latest.pth \
    #--out results.pkl
    #--show

