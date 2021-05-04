MODEL=faster_rcnn_r50_fpn_fp16_4x1_3e_1280x1920
MODEL=retinanet_r50_fpn_fp16_4x1_3e_1280x1920


CUDA_VISIBLE_DEVICES=1 python tools/test.py saved_models/study/${MODEL}/${MODEL}.py \
    saved_models/study/${MODEL}/latest.pth \
    --eval bbox \
    --out saved_models/study/${MODEL}/results_sample_nms_0.7.pkl  \
    --eval-options "classwise=True" "outfile_prefix=saved_models/study/"${MODEL}/"predictions_waymo" \
#    --show
#    --show-dir saved_models/${MODEL}/predictions   # Save images with predictions
#    --format-only \
#    --eval-options "outfile_prefix=./faster_rcnn_res" "format_type=waymo"



