MODEL=lidar/faster_rcnn_r50_fpn_fp16_4x2_1x_1280x1920_enet


CUDA_VISIBLE_DEVICES=0  python tools/test.py configs/waymo_open/${MODEL}.py saved_models/${MODEL}/latest.pth \
    --eval bbox
    #--out results.pkl
    #--show

