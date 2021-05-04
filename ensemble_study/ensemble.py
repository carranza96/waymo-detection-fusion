import mmcv
from mmcv import Config
from mmdet.datasets import build_dataset
import torch
import numpy as np
from mmcv.ops.nms import nms, soft_nms
from ensemble_boxes.ensemble_boxes_nms import nms_float_fast
from time import time
from numba import jit
from mean_ap import eval_map, get_cls_results
from wbf import wbf





def loadResults(resFiles):
    """
        Load results of the given pkl files and combine them in a list of lists with irregular shape
        [N_images, N_models, N_classes, N_detections, 5 (x1,y1,x2,y2,score)]

        Args:
            resFiles: list containing the paths to the pkl files containing detections
        Returns:
            combined_dets: list with the combined detections from all models
    """
    list_dets = [mmcv.load(resFile) for resFile in resFiles]
    ## TODO: Assert all results have same number of images
    N_images = len(list_dets[0])
    combined_dets = [[dets[n] for dets in list_dets] for n in range(N_images)]
    return combined_dets

    # for i in range(N_images):
    #     dets_img = [ [dets[i] for dets in list_dets]]


    # N_classes = len(list_dets[0][0])
    #
    # # combined_dets = [       for ]
    # combined_dets = []
    # for i in range(N_images):
    #     dets_img = []
    #     for j in range(N_classes):
    #         dets_class = np.concatenate([dets[i][j] for dets in list_dets])
    #         dets_img.append(dets_class)
    #     combined_dets.append(dets_img)
    # return combined_dets



def ensembleDetections(dets, cfg):
    """"
    Ensemble detections of different models for a single image
        Args:
            dets: list of boxes predictions from each model
            It has 4 dimensions (N_models, N_classes, N_detections, 5 (x1,y1,x2,y2,score))
        Returns:
            merged_dets: list with the merged detections using the ensemble method
    """""

    cfg_ = cfg.copy()
    method = cfg_.pop('type', 'nms')
    method_op = eval(method)

    n_classes = len(dets[0])
    merged_dets = []
    # Run ensemble for each class independently
    t = time()
    for c in range(n_classes):
        dets_c = np.concatenate([d[c] for d in dets])   # Concatenate detections of class C of all models
        if 'nms' in method:
            merged_dets_c = method_op(torch.tensor(dets_c[:, :4]), torch.tensor(dets_c[:, -1]), **cfg_)[0].numpy()
        else:
            merged_dets_c = method_op(dets_c, weights=np.ones(len(dets)), **cfg_)
        # o_nms = o[nms_float_fast(o[:, :4], o[:, -1], 0.7)]
        merged_dets.append(merged_dets_c)
    t2 = time() - t
    return merged_dets   #, t2



if __name__ == "__main__":

    # models = ['faster_rcnn_r50_fpn_fp16_4x1_3e_1280x1920', 'cascade_rcnn_r50_fpn_fp16_2x1_3e_1280x1920',
    #           'cascade_rcnn_res2net_fpn_fp16_1x1_3e_1280x1920']
    # models = ['faster_rcnn_r50_fpn_fp16_4x2_1x_1280x1920', 'retinanet_r50_fpn_fp16_4x2_1x_1280x1920']
    # models = ['faster_rcnn_r50_fpn_fp16_4x1_3e_1280x1920']
    models = ['faster_rcnn_r50_fpn_fp16_4x1_3e_1280x1920', 'retinanet_r50_fpn_fp16_4x1_3e_1280x1920']

    resFiles = ["saved_models/study/{}/results_sample.pkl".format(model) for model in models]

    cfg = Config.fromfile("saved_models/study/{}/{}.py".format(models[0], models[0]))
    cfg.data.test.test_mode = True  # To avoid filtering out images without gts
    dataset = build_dataset(cfg.data.test)


    cfg = {'type': 'nms', 'iou_threshold': 0.7}
    # cfg = {'type': 'soft_nms', 'iou_threshold': 0.3, 'sigma': 0.5, 'min_score': 1e-3, 'method': 'linear'}
    # cfg = {'type': 'wbf', 'iou_threshold': 0.7}

    combined_dets = loadResults(resFiles)
    ensemble_dets = [ensembleDetections(dets, cfg) for dets in combined_dets]
    # time = [n[1] for n in nms_output]
    # print(np.mean(time))
    # res = dataset.evaluate(ensemble_dets)
    print()

    for i in range(len(ensemble_dets)):
        ensemble_dets[i][0] = ensemble_dets[i][0][ensemble_dets[i][0][:, 4] > 0.05]
        ensemble_dets[i][1] = ensemble_dets[i][1][ensemble_dets[i][1][:, 4] > 0.05]
        ensemble_dets[i][2] = ensemble_dets[i][2][ensemble_dets[i][2][:, 4] > 0.05]

    anns = [dataset.get_ann_info(n) for n in range(len(ensemble_dets))]
    mean_ap, eval_results, df_summary, recalls, precisions = eval_map(ensemble_dets, anns, nproc=4, model_name=models[0])
    mmcv.dump(ensemble_dets, "results_sample.pkl")