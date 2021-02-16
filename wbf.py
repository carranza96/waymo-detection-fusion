from numba import jit
import numpy as np

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

