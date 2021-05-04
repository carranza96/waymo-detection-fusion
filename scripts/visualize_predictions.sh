#MODEL=study/faster_rcnn_r50_fpn_fp16_4x2_1x_1280x1920
MODEL=study/faster_rcnn_r50_fpn_fp16_4x1_3e_1280x1920
#MODEL=study/retinanet_r50_fpn_fp16_4x1_3e_1280x1920
#MODEL=study/retinanet_r50_fpn_fp16_4x2_1x_1280x1920
#MODEL=study/ensemble/wbf_faster50_retina50_3e
#MODEL=study/ensemble/fusion_faster50_retina50_3e


python DetVisGUI/DetVisGUI.py \
--img_root data/waymococo_f0/val2020/ \
--anno_root data/waymococo_f0/annotations/instances_val2020_sample2000.json \
--det_file results_fusing_reverse.pkl
#--det_file saved_models/${MODEL}/results_sample.pkl
#--det_file results_fusing.pkl

#python DetVisGUI/DetVisGUI.py \
#--img_root data/waymococo_f0/train2020/ \
#--anno_root data/waymococo_f0/annotations/instances_train2020.json \
#--det_file saved_models/${MODEL}/results_training.pkl

# [--stage ${STAGE}] [--output ${SAVE_DIRECTORY}]