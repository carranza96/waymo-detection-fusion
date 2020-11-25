MODEL=faster_rcnn_r50_fpn_fp16_8x1_1x_waymo_open_f0


python tools/coco_error_analysis.py saved_models/${MODEL}/predictions.bbox.json saved_models/${MODEL}/ \
--ann=data/waymococo_f0/annotations/instances_val2020.json