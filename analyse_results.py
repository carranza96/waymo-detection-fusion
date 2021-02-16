import mmcv
from mmcv import Config
from mmdet.datasets import build_dataset
from mean_ap import eval_map, get_cls_results
import os
import pandas as pd



# models = ['faster_rcnn_r50_fpn_fp16_4x1_3e_1280x1920']
models = [dir for dir in os.listdir("saved_models/study/")]
resFiles = ["saved_models/study/{}/results.pkl".format(model) for model in models]
cfg = Config.fromfile("saved_models/study/{}/{}.py".format(models[0], models[0]))
cfg.data.test.test_mode = True  #To avoid filtering out images without gts
dataset = build_dataset(cfg.data.test)


df = pd.DataFrame()
for i, res in enumerate(resFiles):
    dets = mmcv.load(res)
    anns = [dataset.get_ann_info(n) for n in range(len(dets))]
    mean_ap, eval_results, df_summary = eval_map(dets, anns, nproc=4, model_name=models[i])
    df = pd.concat([df, df_summary])

df.to_csv("test_res.csv")


# Validate training set
# models = ['faster_rcnn_r50_fpn_fp16_4x1_3e_1280x1920']
# resFiles = ["saved_models/study/{}/results_training.pkl".format(model) for model in models]
# cfg = Config.fromfile("saved_models/study/{}/{}_training.py".format(models[0], models[0]))
# cfg.data.test.test_mode = True  #To avoid filtering out images without gts
# dataset = build_dataset(cfg.data.test)
# dets = mmcv.load(resFiles[0])
# anns = [dataset.get_ann_info(n) for n in range(len(dets))]
# mean_ap, eval_results, df_summary2 = eval_map(dets, anns, nproc=4, model_name=models[0]+"_train")
# df_summary = pd.concat([df_summary, df_summary2])
# df_summary.to_csv("faster_r50_train_test_1280.csv")



# cls_dets, cls_gts, cls_gts_ignore = get_cls_results(
#     dets, anns, 0)