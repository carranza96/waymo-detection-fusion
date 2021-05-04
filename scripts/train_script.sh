MODEL=faster_rcnn_r50_dc5_fp16_8x1_7e_waymo_open_f0_pab

CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/waymo_open/${MODEL}.py \
    --work-dir=saved_models/${MODEL}




