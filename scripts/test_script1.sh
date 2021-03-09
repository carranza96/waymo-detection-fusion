MODEL=faster_rcnn_r50_fpn_fp16_8x1_1x_waymo_open_f0


CUDA_VISIBLE_DEVICES=0  python tools/test.py configs/waymo_open/${MODEL}.py saved_models/${MODEL}/latest.pth \
    --eval bbox
    #--out results.pkl
    #--show

