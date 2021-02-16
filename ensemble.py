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

models = ['faster_rcnn_r50_fpn_fp16_8x1_1x_waymo_open_1280x1920', 'retinanet_r50_fpn_fp16_8x1_1x_waymo_open_1280x1920']
resFiles = ["saved_models/paper/{}/results.pkl".format(model) for model in models]

cfg = Config.fromfile("saved_models/paper/{}/{}.py".format(models[0], models[0]))
cfg.data.test.test_mode = True  #To avoid filtering out images without gts
dataset = build_dataset(cfg.data.test)



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









@jit(nopython=True)
def bb_intersection_over_union(A, B) -> float:
    xA = max(A[0], B[0])
    yA = max(A[1], B[1])
    xB = min(A[2], B[2])
    yB = min(A[3], B[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    if interArea == 0:
        return 0.0

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (A[2] - A[0]) * (A[3] - A[1])
    boxBArea = (B[2] - B[0]) * (B[3] - B[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def get_weighted_box(boxes, conf_type='avg'):
    """
    Create weighted box for set of boxes
    :param boxes: set of boxes to fuse
    :param conf_type: type of confidence one of 'avg' or 'max'
    :return: weighted box
    """

    box = np.zeros(5, dtype=np.float32)
    conf = 0
    conf_list = []
    for b in boxes:
        box[:4] += (b[4] * b[:4])
        conf += b[4]
        conf_list.append(b[4])
    if conf_type == 'avg':
        box[4] = conf / len(boxes)
    elif conf_type == 'max':
        box[4] = np.array(conf_list).max()
    box[:4] /= conf
    return box


def find_matching_box(boxes_list, new_box, match_iou):
    best_iou = match_iou
    best_index = -1
    for i in range(len(boxes_list)):
        box = boxes_list[i]
        iou = bb_intersection_over_union(box[:4], new_box[:4])
        if iou > best_iou:
            best_index = i
            best_iou = iou

    return best_index, best_iou


def wbf(boxes_with_scores, weights=None, iou_threshold=0.55, skip_box_thr=0.0, conf_type='avg', allows_overflow=False):

    ## TODO: Probably need to normalize coordinates
    # Sort by score in descending order
    if len(boxes_with_scores) == 0:
        return boxes_with_scores

    boxes = boxes_with_scores[boxes_with_scores[:, 4].argsort()[::-1]]

    # _, order = scores.sort(0, descending=True)
    # boxes = boxes.index_select(0, order)    # dets_sorted
    # dets = torch.cat((boxes[inds], scores[inds].reshape(-1, 1)), dim=1)

    new_boxes = []
    weighted_boxes = []

    for j in range(len(boxes)):
        index, best_iou = find_matching_box(weighted_boxes, boxes[j], iou_threshold)
        if index != -1:
            new_boxes[index].append(boxes[j])
            weighted_boxes[index] = get_weighted_box(new_boxes[index], conf_type)
        else:
            new_boxes.append([boxes[j]])
            weighted_boxes.append(boxes[j])

    # Rescale confidence based on number of models and boxes
    for i in range(len(new_boxes)):
        if not allows_overflow:
            weighted_boxes[i][4] = weighted_boxes[i][4] * min(weights.sum(), len(new_boxes[i])) / weights.sum()
        else:
            weighted_boxes[i][4] = weighted_boxes[i][4] * len(new_boxes[i]) / weights.sum()

    weighted_boxes = np.array(weighted_boxes)
    sorted_weighted_boxes = weighted_boxes[weighted_boxes[:, 4].argsort()[::-1]]

    return sorted_weighted_boxes




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
            merged_dets_c = method_op(torch.tensor(dets_c[:, :4]), torch.tensor(dets_c[:, -1]), **cfg_)[0]
        else:
            merged_dets_c = method_op(dets_c, weights=np.ones(len(dets)), **cfg_)
        # o_nms = o[nms_float_fast(o[:, :4], o[:, -1], 0.7)]
        merged_dets.append(merged_dets_c)
    t2 = time() - t
    return merged_dets   #, t2




# cfg = {'type': 'nms', 'iou_threshold': 0.7}
# cfg = {'type': 'soft_nms', 'iou_threshold': 0.3, 'sigma': 0.5, 'min_score': 1e-3, 'method': 'linear'}
cfg = {'type': 'wbf'}

combined_dets = loadResults(resFiles)
ensemble_dets = [ensembleDetections(dets, cfg) for dets in combined_dets]
# time = [n[1] for n in nms_output]
# print(np.mean(time))
# res = dataset.evaluate(ensemble_dets)
print()

anns = [dataset.get_ann_info(n) for n in range(len(ensemble_dets))]
mean_ap, eval_results, df_summary = eval_map(ensemble_dets, anns, nproc=4, model_name=models[0])