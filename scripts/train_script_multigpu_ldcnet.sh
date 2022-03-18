MODEL=lidar/faster_rcnn_r50_fpn_fp16_4x2_1x_1280x1920_ldcnet

./tools/dist_train.sh configs/waymo_open/${MODEL}.py 2 \
    --work-dir=saved_models/${MODEL}
