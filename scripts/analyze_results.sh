MODEL=paper/faster_rcnn_r101_fpn_fp16_8x1_1x_waymo_open_f0_1280x1920

python tools/analysis_tools/analyze_results.py \
        --config=configs/waymo_open/paper/faster_rcnn_r101_fpn_fp16_8x1_1x_waymo_open_f0_1280x1920.py \
        --prediction_path=saved_models/paper/faster_rcnn_r101_fpn_fp16_8x1_1x_waymo_open_f0_1280x1920/predictions_waymo.bbox.pkl \
        --show_dir=show_dir \
        --eval=bbox \
        --eval-options "classwise=True"

