from mean_ap import eval_map
import matplotlib.pyplot as plt
import mmcv
from mmdet.datasets import  build_dataset
import numpy as np

# cfg = mmcv.Config.fromfile('saved_models/study/faster_rcnn_r50_fpn_fp16_4x1_3e_1280x1920/faster_rcnn_r50_fpn_fp16_4x1_3e_1280x1920.py')
# cfg.data.test.test_mode = True
# dataset = build_dataset(cfg.data.test)
# anns = [dataset.get_ann_info(n) for n in range(len(dataset))]
#
#
# fusing = mmcv.load("results_fusing_reverse.pkl")
# original_res = mmcv.load("saved_models/study/faster_rcnn_r50_fpn_fp16_4x1_3e_1280x1920/results_sample.pkl")
# original_res2 = mmcv.load("saved_models/study/retinanet_r50_fpn_fp16_4x1_3e_1280x1920/results_sample.pkl")
# wbf = mmcv.load("saved_models/study/ensemble/wbf_faster50_retina50_3e/results_sample.pkl")
# nms = mmcv.load("saved_models/study/ensemble/nms_faster50_retina50_3e/results_sample.pkl")
#
# models = [fusing, original_res, original_res2, wbf, nms]

def score_th_results(anns, models_res, models_labels):
    ths = np.arange(0.05, 1, 0.05)
    aps = [[] for _ in range(len(models_res))]
    tps = [[] for _ in range(len(models_res))]
    fps = [[] for _ in range(len(models_res))]
    for sc in ths:
        for j, m in enumerate(models_res):
            for i in range(len(m)):
                m[i][0] = m[i][0][m[i][0][:, 4] > sc]
            mean_ap, eval_results, df_summary, recalls, precisions = eval_map(m, anns, nproc=4, model_name="Ensemble")
            aps[j].append(mean_ap)
            tps[j].append(df_summary["TP"].values[0])
            fps[j].append(df_summary["FP"].values[0])

    def plot_score_th(res, label):
        plt.figure()
        for i, r in enumerate(res):
            plt.plot(ths, r, label=models_labels[i])
        plt.legend()
        plt.xlabel("Score Threshold")
        plt.ylabel(label)
        plt.show()

    plot_score_th(aps, "AP")
    plot_score_th(tps, "TP")
    plot_score_th(fps, "FP")


def pr_curves(recalls_list, precisions_list, model_labels):
    plt.figure()
    for rec, prec, name in zip(recalls_list,precisions_list,model_labels):
        plt.plot(rec, prec, label=name)
    plt.legend()
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()


# fusing = mmcv.load("results_fusing_reverse.pkl")
# original_res = mmcv.load("saved_models/study/faster_rcnn_r50_fpn_fp16_4x1_3e_1280x1920/results_sample.pkl")
# original_res2 = mmcv.load("saved_models/study/retinanet_r50_fpn_fp16_4x1_3e_1280x1920/results_sample.pkl")
# wbf = mmcv.load("saved_models/study/ensemble/wbf_faster50_retina50_3e/results_sample.pkl")
# nms = mmcv.load("saved_models/study/ensemble/nms_faster50_retina50_3e/results_sample.pkl")
#
# for i in range(len(fusing)):
#     fusing[i][0] = fusing[i][0][fusing[i][0][:, 4] > 0.05]
#     original_res[i][0] = original_res[i][0][original_res[i][0][:, 4] > 0.05]
#     original_res2[i][0] = original_res2[i][0][original_res2[i][0][:, 4] > 0.05]
#     wbf[i][0] = wbf[i][0][wbf[i][0][:, 4] > 0.05]
#     nms[i][0] = nms[i][0][nms[i][0][:, 4] > 0.05]
#
# plt.figure()
#
# mean_ap, eval_results, df_summary, recalls, precisions = eval_map(fusing, anns, nproc=4, model_name="Ensemble")
# plt.plot(recalls, precisions, label="Fuse")
#
# mean_ap, eval_results, df_summary, recalls, precisions = eval_map(original_res, anns, nproc=4, model_name="Ensemble")
# plt.plot(recalls, precisions, label="FRCNN")
#
# mean_ap, eval_results, df_summary, recalls, precisions = eval_map(original_res2, anns, nproc=4, model_name="Ensemble")
# plt.plot(recalls, precisions, label="RetinaNet")
#
# mean_ap, eval_results, df_summary, recalls, precisions = eval_map(wbf, anns, nproc=4, model_name="Ensemble")
# plt.plot(recalls, precisions, label="WBF")
#
# mean_ap, eval_results, df_summary, recalls, precisions = eval_map(nms, anns, nproc=4, model_name="Ensemble")
# plt.plot(recalls, precisions, label="NMS")
#
# plt.legend()
# plt.xlabel("Recall")
# plt.ylabel("Precision")
# plt.show()
#

