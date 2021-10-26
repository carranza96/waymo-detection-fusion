import itertools
import logging
import os.path as osp
from collections import OrderedDict
import tempfile

import mmcv
import numpy as np
from mmcv.utils import print_log
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from terminaltables import AsciiTable
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2

from mmdet.core import eval_recalls
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class WaymoOpenDataset(CustomDataset):
    # "ALL_NS" setting (all object types except signs)
    CLASSES = ('TYPE_VEHICLE', 'TYPE_PEDESTRIAN', 'TYPE_CYCLIST')
    CLASSWISE_IOU = {
        'TYPE_VEHICLE': 0.7,
        'TYPE_PEDESTRIAN': 0.5,
        'TYPE_CYCLIST': 0.5
    }
    # class id mapping to "enum Type" in
    # https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/label.proto
    CLASS_TYPE_TO_SUBMIT = {
        'TYPE_VEHICLE': 1,
        'TYPE_PEDESTRIAN': 2,
        'TYPE_CYCLIST': 4
    }

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
        return data_infos

    def get_ann_info(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return self._parse_ann_info(self.data_infos[idx], ann_info)

    def get_cat_ids(self, idx):
        """Get COCO category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return [ann['category_id'] for ann in ann_info]

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.data_infos):
            if self.filter_empty_gt and self.img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def get_subset_by_classes(self):
        """Get img ids that contain any category in class_ids.

        Different from the coco.getImgIds(), this function returns the id if
        the img contains one of the categories rather than all.

        Args:
            class_ids (list[int]): list of category ids

        Return:
            ids (list[int]): integer list of img ids
        """

        ids = set()
        for i, class_id in enumerate(self.cat_ids):
            ids |= set(self.coco.cat_img_map[class_id])
        self.img_ids = list(ids)

        data_infos = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
        return data_infos

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename']#.replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

    def xyxy2xywh(self, bbox):
        """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
        evaluation.

        Args:
            bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
                ``xyxy`` order.

        Returns:
            list[float]: The converted bounding boxes, in ``xywh`` order.
        """

        _bbox = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]

    def xyxy2cxcywh(self, bbox):
        _bbox = bbox.tolist()
        return [
            (_bbox[0] + _bbox[2]) / 2,
            (_bbox[1] + _bbox[3]) / 2,
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]

    def _det2dicts(self, results):
        dict_results = []
        for idx in range(len(self)):
            img_info = self.data_infos[idx]
            result = results[idx]
            num_valid_labels = min(len(result), len(self.cat_ids))
            for label in range(num_valid_labels):
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    cx, cy, w, h = self.xyxy2cxcywh(bboxes[i])
                    class_name = self.CLASSES[label]
                    data = dict()
                    data['filename'] = img_info['filename']
                    data['context_name'] = img_info['context_name']
                    data['timestamp_micros'] = img_info['timestamp_micros']
                    data['camera_name'] = img_info['camera_id']
                    data['frame_index'] = img_info['frame_id']
                    data['time_of_day'] = img_info['time_of_day']
                    data['location'] = img_info['location']
                    data['weather'] = img_info['weather']
                    data['center_x'] = cx
                    data['center_y'] = cy
                    data['length'] = w  # length: dim x
                    data['width'] = h  # width: dim y
                    data['score'] = float(bboxes[i][4])
                    data['type'] = self.CLASS_TYPE_TO_SUBMIT[class_name]
                    data['id'] = f'{idx}_{label}_{i}'  # dummy tracking id
                    dict_results.append(data)

        return dict_results

    def _proposal2json(self, results):
        """Convert proposal results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            bboxes = results[idx]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = 1
                json_results.append(data)
        return json_results

    def _det2json(self, results):
        """Convert detection results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            num_valid_labels = min(len(result), len(self.cat_ids))
            for label in range(num_valid_labels):
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    json_results.append(data)

        num_dets = len(json_results)
        num_imgs = len(self)
        avg_dets = num_dets / num_imgs
        print(
            f'{num_dets} detections, {num_imgs} images (avg. {avg_dets:.2f})')

        return json_results

    def _segm2json(self, results):
        """Convert instance segmentation results to COCO json style."""
        bbox_json_results = []
        segm_json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            det, seg = results[idx]
            for label in range(len(det)):
                # bbox results
                bboxes = det[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    bbox_json_results.append(data)

                # segm results
                # some detectors use different scores for bbox and mask
                if isinstance(seg, tuple):
                    segms = seg[0][label]
                    mask_score = seg[1][label]
                else:
                    segms = seg[label]
                    mask_score = [bbox[4] for bbox in bboxes]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(mask_score[i])
                    data['category_id'] = self.cat_ids[label]
                    if isinstance(segms[i]['counts'], bytes):
                        segms[i]['counts'] = segms[i]['counts'].decode()
                    data['segmentation'] = segms[i]
                    segm_json_results.append(data)
        return bbox_json_results, segm_json_results

    def results2dicts(self, results, outfile_prefix):
        result_files = dict()
        if isinstance(results[0], list):
            dict_results = self._det2dicts(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.pkl'
            mmcv.dump(dict_results, result_files['bbox'])
        else:
            raise TypeError('invalid type of results')
        return result_files

    def results2proto(self, results, outfile_prefix):
        if isinstance(results[0], list):
            dict_results = self._det2dicts(results)

        detections = metrics_pb2.Objects()

        for detection in dict_results:
            obj = metrics_pb2.Object()

            # fig = plt.figure()
            # img = mpimg.imread('data/waymococo_f0/val2020/'+detection['filename'])
            # plt.imshow(img)
            #
            # rect = patches.Rectangle((detection['center_x']-detection['length']/2, detection['center_y'] -detection['width']/2 )
            #                          , detection['length'], detection['width'], linewidth=1, edgecolor='r', facecolor='none')
            #
            # ax = plt.gca()
            # # Add the patch to the Axes
            # ax.add_patch(rect)
            # plt.show()

            lab = label_pb2.Label()
            lab.box.center_x = detection['center_x']
            lab.box.center_y = detection['center_y']
            lab.box.length = detection['length']
            lab.box.width = detection['width']
            lab.type = detection['type']

            obj.object.MergeFrom(lab)
            if detection['score']:
                obj.score = detection['score']

            obj.context_name = detection["context_name"]
            obj.frame_timestamp_micros = detection["timestamp_micros"]
            obj.camera_name = detection["camera_name"]

            detections.objects.append(obj)

        f = open(outfile_prefix + ".bin", 'wb')
        serialized = detections.SerializeToString()
        f.write(serialized)
        f.close()

    def anns2proto(self, outfile_prefix):
        anns = self.coco.anns
        ground_truths = metrics_pb2.Objects()

        for _, ann in anns.items():
            obj = metrics_pb2.Object()

            img_info = self.data_infos[ann['image_id']]

            x1, y1, w, h = ann['bbox']

            cx = x1 + w / 2
            cy = y1 + h / 2

            lab = label_pb2.Label()
            lab.box.center_x = cx
            lab.box.center_y = cy
            lab.box.length = w
            lab.box.width = h
            lab.type = 4 if ann['category_id'] == 3 else ann['category_id']
            lab.detection_difficulty_level = 1 if ann['det_difficult'] == 0 else ann['det_difficult']
            obj.object.MergeFrom(lab)

            obj.context_name = img_info["context_name"]
            obj.frame_timestamp_micros = img_info["timestamp_micros"]
            obj.camera_name = img_info["camera_id"]

            ground_truths.objects.append(obj)

        f = open(outfile_prefix + "_gt.bin", 'wb')
        serialized = ground_truths.SerializeToString()
        f.write(serialized)
        f.close()

    def results2json(self, results, outfile_prefix):
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict[str: str]: Possible keys are "bbox", "segm", "proposal", and
                values are corresponding filenames.
        """
        result_files = dict()
        if isinstance(results[0], list):
            json_results = self._det2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            mmcv.dump(json_results, result_files['bbox'])
        elif isinstance(results[0], tuple):
            json_results = self._segm2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            result_files['segm'] = f'{outfile_prefix}.segm.json'
            mmcv.dump(json_results[0], result_files['bbox'])
            mmcv.dump(json_results[1], result_files['segm'])
        elif isinstance(results[0], np.ndarray):
            json_results = self._proposal2json(results)
            result_files['proposal'] = f'{outfile_prefix}.proposal.json'
            mmcv.dump(json_results, result_files['proposal'])
        else:
            raise TypeError('invalid type of results')
        return result_files

    def fast_eval_recall(self, results, proposal_nums, iou_thrs, logger=None):
        gt_bboxes = []
        for i in range(len(self.img_ids)):
            ann_ids = self.coco.get_ann_ids(img_ids=self.img_ids[i])
            ann_info = self.coco.load_anns(ann_ids)
            if len(ann_info) == 0:
                gt_bboxes.append(np.zeros((0, 4)))
                continue
            bboxes = []
            for ann in ann_info:
                if ann.get('ignore', False) or ann['iscrowd']:
                    continue
                x1, y1, w, h = ann['bbox']
                bboxes.append([x1, y1, x1 + w, y1 + h])
            bboxes = np.array(bboxes, dtype=np.float32)
            if bboxes.shape[0] == 0:
                bboxes = np.zeros((0, 4))
            gt_bboxes.append(bboxes)

        recalls = eval_recalls(
            gt_bboxes, results, proposal_nums, iou_thrs, logger=logger)
        ar = recalls.mean(axis=1)
        return ar

    def format_results(self,
                       results,
                       outfile_prefix=None,
                       format_type='waymo',
                       **kwargs):
        """Format the results to list[dict] or json.

        Args:
            results (list[tuple | numpy.ndarray]): Testing results of the
                dataset.
            outfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing
                the json filepaths, tmp_dir is the temporal directory created
                for saving json files when outfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        if format_type == 'waymo':
            result_files = self.results2dicts(results, outfile_prefix)
            self.results2proto(results, outfile_prefix)
            self.anns2proto(outfile_prefix)

        elif format_type == 'coco':
            result_files = self.results2json(results, outfile_prefix)
        else:
            raise ValueError('invalid format type')

        return result_files, tmp_dir

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 outfile_prefix=None,
                 classwise=True,
                 proposal_nums=(100, 300, 1000),
                 largest_max_dets=None,
                 iou_thrs=np.arange(0.5, 0.96, 0.05),
                 waymo_metrics=False,
                 time_of_day=None,
                 weather=None):
        """Evaluation in COCO protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            outfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float]): IoU threshold used for evaluating
                recalls. If set to a list, the average recall of all IoUs will
                also be computed. Default: 0.5.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        self.format_results(results, outfile_prefix, format_type='waymo')

        result_files, tmp_dir = self.format_results(
            results, outfile_prefix, format_type='coco')

        eval_results = {}
        cocoGt = self.coco
        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'proposal_fast':
                ar = self.fast_eval_recall(
                    results, proposal_nums, iou_thrs, logger='silent')
                log_msg = []
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
                    log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                cocoDt = cocoGt.loadRes(result_files[metric])
            except IndexError:
                print_log(
                    'The testing results of the whole dataset is empty.',
                    logger=logger,
                    level=logging.ERROR)
                break

            iou_type = 'bbox' if metric == 'proposal' else metric

            # Fix id first gt
            if 0 in cocoGt.anns.keys():
                n_gts = len(cocoGt.anns)
                ann = cocoGt.anns.pop(0)
                ann['id'] = n_gts
                cocoGt.anns[n_gts] = ann

            if time_of_day: # Day, Night, Dawn/Dusk
                filt_imgs = [k for k, v in cocoGt.imgs.items() if v['time_of_day'] == time_of_day]
                cocoGt.imgToAnns = {
                    imgId: anns
                    for (imgId, anns) in cocoGt.imgToAnns.items() if imgId in filt_imgs
                }
                #
                anns = {k: v for (k, v) in cocoGt.anns.items() if v['image_id'] in filt_imgs}
                cocoGt.anns = anns


                cocoDt.imgToAnns = {
                    imgId: anns
                    for (imgId, anns) in cocoDt.imgToAnns.items() if imgId in filt_imgs
                }
                #
                anns = {k: v for (k, v) in cocoDt.anns.items() if v['image_id'] in filt_imgs}
                cocoDt.anns = anns

            if weather: # rain, sunny
                filt_imgs = [k for k, v in cocoGt.imgs.items() if v['weather'] == weather]
                cocoGt.imgToAnns = {
                    imgId: anns
                    for (imgId, anns) in cocoGt.imgToAnns.items() if imgId in filt_imgs
                }
                #
                anns = {k: v for (k, v) in cocoGt.anns.items() if v['image_id'] in filt_imgs}
                cocoGt.anns = anns


                cocoDt.imgToAnns = {
                    imgId: anns
                    for (imgId, anns) in cocoDt.imgToAnns.items() if imgId in filt_imgs
                }
                #
                anns = {k: v for (k, v) in cocoDt.anns.items() if v['image_id'] in filt_imgs}
                cocoDt.anns = anns


            cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
            cocoEval.params.catIds = self.cat_ids
            cocoEval.params.imgIds = self.img_ids
            # cocoEval.params.iouThrs = [0.7]
            # cocoEval.params.areaRng = [[0, 10000000000.0]]
            # cocoEval.params.maxDets = [100]
            # Uncomment this line to match with mmdetection mean_ap script
            # cocoEval.params.recThrs = np.arange(0, 1 + 1e-3, 0.1)

            if metric == 'proposal':
                cocoEval.params.useCats = 0
                cocoEval.params.maxDets = list(proposal_nums)
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                metric_items = [
                    'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000', 'AR_m@1000',
                    'AR_l@1000'
                ]
                for i, item in enumerate(metric_items):
                    val = float(f'{cocoEval.stats[i + 6]:.3f}')
                    eval_results[item] = val
            else:
                if largest_max_dets is not None:
                    assert largest_max_dets > 100, \
                        'specify largest_max_dets only when' \
                        'you need to evaluate more than 100 detections'
                    cocoEval.params.maxDets[-1] = largest_max_dets
                    cocoEval.params.maxDets[-2] = 100
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                if classwise:  # Compute per-category AP
                    # Compute per-category AP
                    # from https://github.com/facebookresearch/detectron2/
                    precisions = cocoEval.eval['precision']
                    # precision: (iou, recall, cls, area range, max dets)
                    assert len(self.cat_ids) == precisions.shape[2]

                    table_data = []
                    waymo_iou_metrics = {}
                    iouThr_dict = {None: 'AP', 0.5: 'AP 0.5', 0.7: 'AP 0.7'}
                    for iouThr, metric_name in iouThr_dict.items():
                        results_per_category = []
                        for idx, catId in enumerate(self.cat_ids):
                            # area range index 0: all area ranges
                            # max dets index -1: typically 100 per image
                            nm = self.coco.loadCats(catId)[0]
                            precision = precisions[:, :, idx, 0, -1]
                            if iouThr is not None:
                                t = np.where(
                                    iouThr == cocoEval.params.iouThrs)[0]
                                precision = precision[t]
                            precision = precision[precision > -1]
                            if precision.size:
                                ap = np.mean(precision)
                            else:
                                ap = float('nan')
                            results_per_category.append(
                                (f'{nm["name"]}', f'{float(ap):0.4f}'))

                            if self.CLASSWISE_IOU[nm["name"]] == iouThr:
                                waymo_iou_metrics[nm["name"]] = ap

                        num_columns = min(6, len(results_per_category) * 2)
                        results_flatten = list(
                            itertools.chain(*results_per_category))
                        headers = ['category', metric_name] * (
                            num_columns // 2)
                        results_2d = itertools.zip_longest(*[
                            results_flatten[i::num_columns]
                            for i in range(num_columns)
                        ])
                        table_data += [headers]
                        table_data += [result for result in results_2d]
                    table = AsciiTable(table_data)
                    table.inner_heading_row_border = False
                    table.inner_row_border = True
                    print_log('\n' + table.table, logger=logger)

                    for category, category_iou in self.CLASSWISE_IOU.items():
                        if category not in waymo_iou_metrics:
                            continue
                        print_log(
                            f'AP{category_iou} ({category}): ' +
                            f'{waymo_iou_metrics[category]:0.4f}',
                            logger=logger)
                    ap_waymo = np.mean(list(waymo_iou_metrics.values()))
                    print_log(
                        'AP (Waymo challenge IoU, COCO script): ' +
                        f'{ap_waymo:0.4f}',
                        logger=logger)

                metric_items = [
                    'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                ]
                for i in range(len(metric_items)):
                    key = f'{metric}_{metric_items[i]}'
                    val = float(f'{cocoEval.stats[i]:.3f}')
                    eval_results[key] = val
                ap = cocoEval.stats[:6]
                map_copypaste = (f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} '
                                 f'{ap[3]:.3f} {ap[4]:.3f} {ap[5]:.3f}')
                eval_results[f'{metric}_mAP_copypaste'] = map_copypaste
                print_log(
                    f'{metric}_mAP_copypaste: {map_copypaste}', logger=logger)

        # Filter Level 1
        # eval_results = {}
        # cocoGt.imgToAnns = {
        #     imgId: [ann for ann in anns if ann['det_difficult'] == 0]
        #     for (imgId, anns) in cocoGt.imgToAnns.items()
        # }
        #
        # annsL1 = {k: v for (k, v) in cocoGt.anns.items() if v['det_difficult'] == 0}
        # cocoGt_levels.anns = annsL1

        if tmp_dir is not None:
            tmp_dir.cleanup()

        if not waymo_metrics:
            return eval_results
        else:
            return eval_results, waymo_iou_metrics
