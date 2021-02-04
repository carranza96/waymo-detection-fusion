MODEL=paper/yolo_640x960

python tools/analysis_tools/benchmark.py configs/waymo_open/${MODEL}.py \
       saved_models/${MODEL}/latest.pth \
