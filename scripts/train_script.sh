MODEL=paper/fcos_640x960

python tools/train.py configs/waymo_open/${MODEL}.py \
    --work-dir=saved_models/${MODEL}




