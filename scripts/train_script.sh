MODEL=paper/detr_r50

python tools/train.py configs/waymo_open/${MODEL}.py \
    --work-dir=saved_models/${MODEL}




