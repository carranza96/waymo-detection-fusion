MODEL=yolof_r50_c5_fp16_4x2_1x_1280x1920
SPLIT=testing


INPUT_FILENAME=saved_models/study/${MODEL}/predictions_waymo_full_val.bin
OUTPUT_FILENAME=saved_models/study/${MODEL}/predictions_waymo_full_val_submission.bin
SUBMISSION_METADATA=saved_models/study/${MODEL}/submission.txtpb  # This file must be edited before executing the script

cd waymo-open-dataset
bazel build waymo_open_dataset/metrics/tools/create_submission
cd ..


waymo-open-dataset/bazel-bin/waymo_open_dataset/metrics/tools/create_submission  \
--input_filenames=${INPUT_FILENAME} \
--output_filename=${OUTPUT_FILENAME} \
--submission_filename=${SUBMISSION_METADATA} \
--num_shards=1

#tar cvf /tmp/my_model.tar /tmp/my_model
#gzip /tmp/my_model.tar
