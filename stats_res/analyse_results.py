import mmcv
from mmcv import Config
from mmdet.datasets import build_dataset
from mean_ap import eval_map, get_cls_results
# from mmdet.core.evaluation.mean_ap import eval_map, tpfp_imagenet
import os
import pandas as pd
import numpy as np

def bbox_area(bboxes):
    """Compute the area of an array of bboxes.

    Args:
        bboxes (Tensor): The coordinates ox bboxes. Shape: (m, 4)

    Returns:
        Tensor: Area of the bboxes. Shape: (m, )
    """
    w = (bboxes[2] - bboxes[0])
    h = (bboxes[3] - bboxes[1])
    areas = w * h
    return areas

def categorize_false_positives(df):
    cats = []
    for i, row in df.iterrows():
        cat = ""
        if row["fp"] == 1:
            if row["fp_redundant"] == 1:
                cat = "Redundant"
            elif row["matched_gt_class"] == -1:
                cat = "Background"
            elif row["matched_gt_class"] != -1 and row["matched_gt_class"] != row["class"]:
                cat = "Wrong class"
            # elif row["matched_gt_class"] != -1 and row["matched_gt_class"] != row["class"]:

            else:
                cat = "Unknown"
        cats.append(cat)
    return cats

def stats_gt_dets(eval_results, dataset, detections, annotations, model_name):

    rows_dets, rows_gts = [], []
    for c, stats_class in enumerate(eval_results):

        for i in range(len(dataset)):
            tps = list(stats_class['tp'][i].squeeze(axis=0))
            fps = list(stats_class['fp'][i].squeeze(axis=0))
            fps_red = list(np.array(stats_class['fp_redundant'][i]).squeeze(axis=0))
            iou_matched_gt = list(stats_class["dets_iou_max"][i])
            matched_gt_index = list(stats_class["dets_iou_argmax"][i])
            matched_gt_class = list(stats_class["dets_matched_class"][i])

            for j, (tp, fp, fp_red, iou, gt_index, gt_class) in enumerate(
                    zip(tps, fps, fps_red, iou_matched_gt, matched_gt_index, matched_gt_class)):
                original_det = detections[i][c][j]
                area = bbox_area(original_det)
                row = [dataset.data_infos[i]["filename"], i, j, c] + list(original_det) + [area, int(tp), int(fp),
                                                                                           int(fp_red), iou,
                                                                                           int(gt_index), int(gt_class)]
                rows_dets.append(row)

            gts_covered = list(np.array(stats_class["gt_covered"][i], dtype=np.int32))
            gts_iou_max = list(stats_class["gts_iou_max"][i])
            gts_iou_argmax = list(stats_class["gts_iou_argmax"][i])

            gt_inds = annotations[i]['labels'] == c
            cls_gts = annotations[i]['bboxes'][gt_inds, :]

            for j, (gt_cov, gt_iou_max, gt_iou_argmax) in enumerate(zip(gts_covered, gts_iou_max, gts_iou_argmax)):
                original_gt = cls_gts[j]
                area = bbox_area(original_gt)
                row = [dataset.data_infos[i]["filename"], i, j, c] + list(original_gt) + \
                      [area, gt_cov, gt_iou_max, int(gt_iou_argmax)]
                rows_gts.append(row)

    save_dir = 'stats_res/{}/'.format(model_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    df_dets = pd.DataFrame(rows_dets,
                      columns=["img_name", "img_index", "det_index", "class", "x1", "y1", "x2", "y2", "score", "area",
                               "tp", "fp", "fp_redundant", "iou_matched_gt", "matched_gt_index", "matched_gt_class"])
    df_dets.sort_values(by=["img_index", "class", "det_index"], axis=0, inplace=True)
    df_dets["frontal"] = [1 if any(s in img_name for s in ["camera1", "camera2", "camera3"]) else 0 for img_name in df_dets["img_name"]]
    df_dets["width"] = (df_dets["x2"] - df_dets["x1"])
    df_dets["height"] = (df_dets["y2"] - df_dets["y1"])
    df_dets["x_center"] = df_dets["x1"] + (df_dets["x2"] - df_dets["x1"])/2
    df_dets["y_center"] = df_dets["y1"] + (df_dets["y2"] - df_dets["y1"]) / 2
    df_dets["fp_cats"] = categorize_false_positives(df_dets)
    df_dets.drop(columns=["x1", "x2", "y1", "y2"], inplace=True)
    df_dets.to_csv(save_dir + "det_stats.csv", index=False)

    df_gts = pd.DataFrame(rows_gts,
                      columns=["img_name", "img_index", "gt_index", "class", "x1", "y1", "x2", "y2", "area", "gt_covered",
                               "gt_iou_max", "gt_iou_argmax"])

    df_gts.sort_values(by=["img_index", "class", "gt_index"], axis=0, inplace=True)
    df_gts["frontal"] = [1 if any(s in img_name for s in ["camera1", "camera2", "camera3"]) else 0 for img_name in df_gts["img_name"]]
    df_gts["x_center"] = df_gts["x1"] + (df_gts["x2"] - df_gts["x1"])/2
    df_gts["y_center"] = df_gts["y1"] + (df_gts["y2"] - df_gts["y1"]) / 2
    df_gts.drop(columns=["x1", "x2", "y1", "y2"], inplace=True)
    df_gts.to_csv(save_dir + "gt_stats.csv", index=False)



def stats_results(models, filt_scores=[0], training_results=False, output='stat_res.csv',
                  dataset_config='configs/_base_/datasets/waymo_detection_1280x1920.py', full_stats=False):
    list_eval_results = []

    resFiles = ["saved_models/study/{}/results_sample.pkl".format(model) for model in models]

    cfg = Config.fromfile(dataset_config)
    cfg.data.test.test_mode = True  # To avoid filtering out images without gts
    dataset = build_dataset(cfg.data.test)

    anns = [dataset.get_ann_info(n) for n in range(len(dataset.img_ids))]

    df = pd.DataFrame()
    for i, res in enumerate(resFiles):
        dets = mmcv.load(res)

        # img_index = 0
        # dets = [dets[img_index]]
        # anns = [anns[img_index]]
        # dataset.data_infos = [dataset.data_infos[img_index]]
        # dataset.img_ids = [img_index]

        for sc in filt_scores:
            filt_dets = dets
            model_name = models[i]
            if sc!=0:
                filt_dets = [[dets_class[dets_class[:, 4] > sc] for dets_class in dets_img] for dets_img in dets]
                model_name += "_"+str(sc)
            mean_ap, eval_results, df_summary = eval_map(filt_dets, anns, nproc=4, model_name=model_name)
            list_eval_results.append(eval_results)
            df = pd.concat([df, df_summary])

            if full_stats:
                stats_gt_dets(eval_results, dataset, dets, anns, model_name)

            # mean_ap, eval_results = eval_map(filt_dets, anns, nproc=4, tpfp_fn=tpfp_imagenet, iou_thr=0.7)
            # mean_ap, eval_results = eval_map(filt_dets, anns, nproc=4, tpfp_fn=tpfp_imagenet)
            # mean_ap, eval_results = eval_map(filt_dets, anns, nproc=4, iou_thr=0.7)
            # mean_ap, eval_results = eval_map(filt_dets, anns, nproc=4)
            #
            # dataset.coco.load_anns(232)
            # dataset.coco.dataset['images'] = [dataset.coco.dataset['images'][232]]
            # anns = [v for k, v in dataset.coco.anns.items() if v['image_id']==232 ]
            # dataset.coco.anns = {0: dataset.coco.anns[232]}
            # dataset.coco.imgs = {0: dataset.coco.imgs[232]}
            # # dataset.coco["annotations"] = {0: dataset.coco.anns[232]}

            # dataset.evaluate(filt_dets)


    if training_results:
        cfg.data.train.test_mode = True
        dataset_train = build_dataset(cfg.data.train)
        anns_train = [dataset_train.get_ann_info(n) for n in range(len(dataset_train.img_ids))]

        resFiles = ["saved_models/study/{}/results_training.pkl".format(model) for model in models
                    if os.path.exists("saved_models/study/{}/results_training.pkl".format(model))]

        for i, res in enumerate(resFiles):
            dets = mmcv.load(res)
            for sc in filt_scores:
                filt_dets = dets
                model_name = models[i] + "_train"
                if sc != 0:
                    filt_dets = [[dets_class[dets_class[:, 4] > sc] for dets_class in dets_img] for dets_img in dets]
                    model_name += "_" + str(sc)
                mean_ap, eval_results, df_summary = eval_map(filt_dets, anns_train, nproc=4, model_name=model_name)
                list_eval_results.append(eval_results)
                df = pd.concat([df, df_summary])

    df.to_csv("stats_res/"+output)
    return list_eval_results


# models = ['faster_rcnn_r50_fpn_fp16_4x1_3e_1280x1920', 'retinanet_r50_fpn_fp16_4x1_3e_1280x1920',
#           'ensemble/nms_faster50_retina50_3e', 'ensemble/wbf_faster50_retina50_3e']
# models = ['retinanet_r50_fpn_fp16_4x2_1x_1280x1920']
# models = ['yolof_r50_c5_fp16_4x2_1x_1280x1920']
# models = ['ensemble/wbf_faster50_retina50_1x']
# models = ['ensemble/fusion_faster50_retina50_3e']
models = ['faster_rcnn_r50_fpn_fp16_4x2_3x_1280x1920']
models = ['atss_r50_fpn_fp16_4x2_1x_1280x1920']

# models = ['retinanet_r50_fpn_fp16_4x1_3e_1280x1920']
# models = ['retinanet_r50_fpn_fp16_4x2_1x_1280x1920']
# models = ['cascade_rcnn_r50_fpn_fp16_4x2_1x_1280x1920']
# models = [dir for dir in os.listdir("saved_models/study/")]
list_eval_results = stats_results(models, filt_scores=[0], training_results=False, output='stat_res.csv', full_stats=False,
                                  dataset_config='saved_models/study/faster_rcnn_r50_fpn_fp16_4x2_1x_1280x1920/faster_rcnn_r50_fpn_fp16_4x2_1x_1280x1920.py')


#
# import seaborn as sns
# # sns.displot(data=df[df["fp"]==1], x="score")
# sns.histplot(data=df[df["fp"]==1], x="score", bins=10)
#
# cls_dets, cls_gts, cls_gts_ignore = get_cls_results(
#     dets, anns, 0)