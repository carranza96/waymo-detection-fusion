MODEL=lidar/faster_rcnn_r50_fpn_fp16_4x2_1x_1280x1920_lidar
# MODEL=lidar/faster_rcnn_r50_fpn_fp16_4x2_1x_1280x1920
#MODEL=study/faster_rcnn_r50_dc5_fp16_4x2_1x_1280x1920

./tools/dist_train.sh configs/waymo_open/${MODEL}.py 2 \
    --work-dir=saved_models/${MODEL}

