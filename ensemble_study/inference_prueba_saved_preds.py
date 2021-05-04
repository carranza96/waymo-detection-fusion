from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from time import time, perf_counter
import numpy as np
import torch
from torch import nn
import mmcv
import warnings
from mmcv.runner import auto_fp16, force_fp32
from mmdet.models.losses import accuracy
from mmdet.core.bbox.transforms import bbox_xyxy_to_cxcywh
from mmdet.core import get_classes
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmcv.parallel import MMDataParallel,MMDistributedDataParallel
import torch
from mmcv import Config
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model

from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from torch.nn import  Sequential
from mmdet.core.bbox.iou_calculators import BboxOverlaps2D, bbox_overlaps
from tools.train import train_detector, set_random_seed
from mmdet.models.detectors.base import BaseDetector
from mmdet.models.builder import HEADS, build_loss
from mmdet.core import (anchor_inside_flags, build_anchor_generator,
                        build_assigner, build_bbox_coder, build_sampler,
                        images_to_levels, multi_apply, multiclass_nms, unmap)

from collections import OrderedDict
import torch.distributed as dist
from tools.test import single_gpu_test, multi_gpu_test
from mean_ap import eval_map
from ensemble import ensembleDetections
import asyncio
from mmdet.utils.contextmanagers import concurrent
from mmcv.runner import get_dist_info, init_dist
import os
import wandb

# 1. Start a new run
wandb.init(project='ensemble_od', entity='minerva')


res1 = mmcv.load("saved_models/study/faster_rcnn_r50_fpn_fp16_4x1_3e_1280x1920/results_training_all.pkl")
# res1_filt = [[l[0].astype((np.float16))] for l in res1]

res2 = mmcv.load("saved_models/study/retinanet_r50_fpn_fp16_4x1_3e_1280x1920/results_training_all_cars.pkl")
# res2_filt = [[l[0].astype((np.float16))] for l in res2]

class Fusion(nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()
        self.name = 'fusion_layer'

        self.fuse = Sequential(
            nn.Conv2d(7, 18, 1),
            nn.ReLU(),
            nn.Conv2d(18, 36, 1),
            nn.ReLU(),
            nn.Conv2d(36, 36, 1),
            nn.ReLU(),
            nn.Conv2d(36, 1, 1),
        )
        self.maxpool = Sequential(
            nn.MaxPool2d([2000, 1], 1),
        )


    def forward(self, input, T_out, T_indices):

        x = self.fuse(input)

        T_out[:, :, :] = -9999.
        T_out[:, T_indices[0], T_indices[1]] = x[0, 0, :, :]

        x = self.maxpool(T_out)

        return x.squeeze()



class EnsembleModel(nn.Module):

    def __init__(self, models):
        super(EnsembleModel, self).__init__()
        self.models = models
        # TODO: Fix half()
        self.fusion = Fusion().half()

        self.num_classes = 1
        loss_cls = dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0)

        assigner = dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            match_low_quality=False,
            ignore_iof_thr=-1)

        self.assigner = build_assigner(assigner)
        self.loss_cls = build_loss(loss_cls)

        sampler_cfg = dict(type='PseudoSampler')
        self.sampler = build_sampler(sampler_cfg)


    def get_targets(self,
                    sampling_results,
                    gt_bboxes,
                    gt_labels,
                    rcnn_train_cfg,
                    concat=True):
        """Calculate the ground truth for all samples in a batch according to
        the sampling_results.

        Almost the same as the implementation in bbox_head, we passed
        additional parameters pos_inds_list and neg_inds_list to
        `_get_target_single` function.

        Args:
            sampling_results (List[obj:SamplingResults]): Assign results of
                all images in a batch after sampling.
            gt_bboxes (list[Tensor]): Gt_bboxes of all images in a batch,
                each tensor has shape (num_gt, 4),  the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            gt_labels (list[Tensor]): Gt_labels of all images in a batch,
                each tensor has shape (num_gt,).
            rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.
            concat (bool): Whether to concatenate the results of all
                the images in a single batch.

        Returns:
            Tuple[Tensor]: Ground truth for proposals in a single image.
            Containing the following list of Tensors:

                - labels (list[Tensor],Tensor): Gt_labels for all
                  proposals in a batch, each tensor in list has
                  shape (num_proposals,) when `concat=False`, otherwise
                  just a single tensor has shape (num_all_proposals,).
                - label_weights (list[Tensor]): Labels_weights for
                  all proposals in a batch, each tensor in list has
                  shape (num_proposals,) when `concat=False`, otherwise
                  just a single tensor has shape (num_all_proposals,).
                - bbox_targets (list[Tensor],Tensor): Regression target
                  for all proposals in a batch, each tensor in list
                  has shape (num_proposals, 4) when `concat=False`,
                  otherwise just a single tensor has shape
                  (num_all_proposals, 4), the last dimension 4 represents
                  [tl_x, tl_y, br_x, br_y].
                - bbox_weights (list[tensor],Tensor): Regression weights for
                  all proposals in a batch, each tensor in list has shape
                  (num_proposals, 4) when `concat=False`, otherwise just a
                  single tensor has shape (num_all_proposals, 4).
        """
        # TODO: Remove train cfg
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        labels, label_weights, bbox_targets, bbox_weights = multi_apply(
            self._get_target_single,
            pos_bboxes_list,
            neg_bboxes_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            cfg=rcnn_train_cfg)

        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
        return labels, label_weights, bbox_targets, bbox_weights


    def _get_target_single(self, pos_bboxes, neg_bboxes, pos_gt_bboxes,
                           pos_gt_labels, cfg):
        # TODO: Remove train cfg
        """Calculate the ground truth for proposals in the single image
        according to the sampling results.

        Args:
            pos_bboxes (Tensor): Contains all the positive boxes,
                has shape (num_pos, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            neg_bboxes (Tensor): Contains all the negative boxes,
                has shape (num_neg, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            pos_gt_bboxes (Tensor): Contains all the gt_boxes,
                has shape (num_gt, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            pos_gt_labels (Tensor): Contains all the gt_labels,
                has shape (num_gt).
            cfg (obj:`ConfigDict`): `train_cfg` of R-CNN.

        Returns:
            Tuple[Tensor]: Ground truth for proposals
            in a single image. Containing the following Tensors:

                - labels(Tensor): Gt_labels for all proposals, has
                  shape (num_proposals,).
                - label_weights(Tensor): Labels_weights for all
                  proposals, has shape (num_proposals,).
                - bbox_targets(Tensor):Regression target for all
                  proposals, has shape (num_proposals, 4), the
                  last dimension 4 represents [tl_x, tl_y, br_x, br_y].
                - bbox_weights(Tensor):Regression weights for all
                  proposals, has shape (num_proposals, 4).
        """
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_bboxes.new_full((num_samples, ),
                                     self.num_classes,
                                     dtype=torch.long)
        # labels = pos_bboxes.new_full((num_samples, ),
        #                              0.,
        #                              dtype=torch.long)
        label_weights = pos_bboxes.new_zeros(num_samples)
        bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
        bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
        if num_pos > 0:
            # TODO: Fix foregound first class
            labels[:num_pos] = pos_gt_labels #+ 1.
            # pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            pos_weight = 1.0
            label_weights[:num_pos] = pos_weight
            # if not self.reg_decoded_bbox:
            #     pos_bbox_targets = self.bbox_coder.encode(
            #         pos_bboxes, pos_gt_bboxes)
            # else:
            #     # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            #     # is applied directly on the decoded bounding boxes, both
            #     # the predicted boxes and regression targets should be with
            #     # absolute coordinate format.
            pos_bbox_targets = pos_gt_bboxes
            bbox_targets[:num_pos, :] = pos_bbox_targets
            bbox_weights[:num_pos, :] = 1
        if num_neg > 0:
            label_weights[-num_neg:] = 1.0

        return labels, label_weights, bbox_targets, bbox_weights

    @auto_fp16(apply_to=('img', ))
    def forward(self, img, img_metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def _expand_onehot_labels(self, labels, label_weights, label_channels):
        bin_labels = labels.new_full((labels.size(0), label_channels), 0)
        inds = torch.nonzero(
            (labels >= 0) & (labels < label_channels), as_tuple=False).squeeze()
        if inds.numel() > 0:
            bin_labels[inds, labels[inds]] = 1

        if label_weights is None:
            bin_label_weights = None
        else:
            bin_label_weights = label_weights.view(-1, 1).expand(
                label_weights.size(0), label_channels)

        return bin_labels, bin_label_weights



    def generate_T(self):


    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):

        id = img_metas[0]['id']
        # X1: RetinaNet, X2: FasterRCNN (Supongo que X2 es el mejor modelo)
        x2 = [res1[id]]
        x1 = [res2[id]]


        # TODO: Esta parte (Tensor preparation) es muy lenta
        t = time()
        x1[0][0] = x1[0][0][x1[0][0][:, 4].argsort()][::-1][:2000].copy()
        x2[0][0] = x2[0][0][x2[0][0][:, 4].argsort()][::-1][:1000].copy()


        o_cars = [torch.tensor(x1[0][0]), torch.tensor(x2[0][0])]
        K = x1[0][0].shape[0]
        N = x2[0][0].shape[0]
        F = 7
        T = np.zeros(((K, N, F)), dtype=np.float16)  # TODO: torch.zeros Avoid numpy Float 16

        t = time()
        overlaps = bbox_overlaps(o_cars[0][:, :4], o_cars[1][:, :4])

        # neighbor_pair_idxs = torch.where(torch.greater_equal(overlaps, 0.2))
        # neighbor_pair_idxs = tf.where(tf.greater_equal(overlaps, 0.2))

        # print("BBox overlaps:", time() - t)
        scores_1 = o_cars[0][:, 4].unsqueeze(1).repeat((1, N))
        scores_2 = o_cars[1][:, 4].unsqueeze(1).repeat((1, K)).T

        img_h, img_w = img_metas[0]['img_shape'][0], img_metas[0]['img_shape'][1]
        bboxes_1 = o_cars[0][:, :4]
        bboxes_2 = o_cars[1][:, :4]
        # Normalize boxes
        bboxes_1[:, [0, 2]], bboxes_1[:, [1, 3]] = bboxes_1[:, [0, 2]]/img_w, bboxes_1[:, [1, 3]]/img_h
        bboxes_2[:, [0, 2]], bboxes_2[:, [1, 3]] = bboxes_2[:, [0, 2]]/img_w, bboxes_2[:, [1, 3]]/img_h

        bboxes_1 = bbox_xyxy_to_cxcywh(bboxes_1)
        bboxes_2 = bbox_xyxy_to_cxcywh(bboxes_2)

        # Scale difference
        w1, w2 = bboxes_1[:, 2].unsqueeze(1).repeat((1, N)), bboxes_2[:, 2].unsqueeze(1).repeat((1, K)).T
        h1, h2 = bboxes_1[:, 3].unsqueeze(1).repeat((1, N)), bboxes_2[:, 3].unsqueeze(1).repeat((1, K)).T
        # w_diff = torch.log(w1 / w2)
        # h_diff = torch.log(h1 / h2)
        aspect_diff = torch.log(w1 / h1) - torch.log(w2 / h2)

        # Distance x, y
        cx1, cx2 = bboxes_1[:, 0].unsqueeze(1).repeat((1, N)), bboxes_2[:, 0].unsqueeze(1).repeat((1, K)).T
        cy1, cy2 = bboxes_1[:, 1].unsqueeze(1).repeat((1, N)), bboxes_2[:, 1].unsqueeze(1).repeat((1, K)).T
        x_dist = cx1-cx2
        y_dist = cy1-cy2
        l2_dist = torch.sqrt(x_dist**2 + y_dist**2)

        # T[:, :, 0] = overlaps
        # T[:, :, 1] = scores_1
        # T[:, :, 2] = scores_2
        # T[:, :, 3] = w_diff
        # T[:, :, 4] = h_diff
        # T[:, :, 5] = aspect_diff
        # T[:, :, 6] = x_dist
        # T[:, :, 7] = y_dist
        # T[:, :, 8] = l2_dist


        T[:, :, 0] = overlaps
        T[:, :, 1] = scores_1  # Score Retina
        T[:, :, 2] = scores_2   # Score Faster
        T[:, :, 3] = x_dist
        T[:, :, 4] = y_dist
        T[:, :, 5] = l2_dist
        T[:, :, 6] = l2_dist

        T = torch.tensor(T).cuda().half()

        # Fill last element of column with all IoU zeros with -1
        non_overlapping_dets = ~overlaps.sum(dim=0).bool()
        T[-1, non_overlapping_dets, 0] = -1   # IoU -1
        T[-1, non_overlapping_dets, 1] = -1  # Score Retina -1

        # TODO: Maybe increase threshold to 0.2
        non_empty_indices = torch.nonzero(T[:, :, 0])
        non_empty_indices = torch.nonzero(T[:, :, 0], as_tuple=True)
        # non_empty_indices = torch.where((torch.greater_equal(T[:, :, 0], 0.2)) | (torch.less(T[:, :, 0], 0)))

        non_empty_elements = T[non_empty_indices[0], non_empty_indices[1], :]

        non_empty_elements_T = non_empty_elements.permute(1, 0)
        non_empty_elements_T = non_empty_elements_T.unsqueeze(1).unsqueeze(0).cuda()  # Shape [1,3,1, #non-zero]

        T_out = torch.zeros((1, K, N)).cuda().half()
        # print("Tensor preparation:", time() - t)

        t2 = time()
        new_scores = self.fusion(non_empty_elements_T, T_out, non_empty_indices)
        # print("Fusion:", time() - t2)

        # TODO: Uncomment
        # x1[0][0][:, 4] = new_scores.cpu().detach().numpy()

        bboxes = x2     # [# images, #n_classes, # n_boxes]
        losses = dict()

        # assign_result = [self.assigner.assign(
        #     x[0], gt_bboxes[0], gt_bboxes_ignore, gt_labels[0]) for x in x1]
        #
        # sampling_result = self.sampler.sample(assign_result, anchors,
        #
        #                                       gt_bboxes)
        #

        # Filter GT bboxes of class 0 (vehicles)
        gt_bboxes_old = gt_bboxes
        gt_labels_old = gt_labels
        gt_bboxes = [bb[gt_labels[i]==0]  for i,bb in enumerate(gt_bboxes)]
        gt_labels = [lab[lab==0] for lab in gt_labels]

        t2 = time()
        num_imgs = len(img_metas)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
        sampling_results = []
        for i in range(num_imgs):
            assign_result = self.assigner.assign(
                torch.tensor(bboxes[i][0]).cuda(), gt_bboxes[i], gt_bboxes_ignore[i],
                gt_labels[i])
            sampling_result = self.sampler.sample(
                assign_result,
                torch.tensor(bboxes[i][0]).cuda(),
                gt_bboxes[i])
                # gt_labels[i])
            sampling_results.append(sampling_result)

        # rois = bbox2roi([res.bboxes for res in sampling_results])

        bbox_targets = self.get_targets(sampling_results, gt_bboxes,
                                   gt_labels, rcnn_train_cfg=None)

        # El sampler ordena primeros los N targets positivos y luego los N negativos,
        # hay que reordenar la salida de labels para que coincida con el orden de las detecciones originales
        order_inds = torch.cat((sampling_result.pos_inds, sampling_result.neg_inds))
        original_order = bbox_targets[0].clone()
        bbox_targets[0][order_inds] = original_order
        # bbox_targets_s = (bbox_targets[0][order_inds], bbox_targets[1][order_inds], bbox_targets[2], bbox_targets[3])

        # cls_score = torch.tensor(1 - x1[0][0][:, 4], requires_grad=True).cuda()
        # cls_score = 1 - new_scores
        m = nn.Sigmoid()
        cls_score = torch.unsqueeze(new_scores, 1)
        loss_bbox = self.loss(cls_score, *bbox_targets)
        if loss_bbox['loss_cls'] > 1:
            print()

        losses.update(loss_bbox)

        # loss_prev_ce = nn.functional.binary_cross_entropy(torch.tensor(1 - x1[0][0][:, 4]).cuda(), bbox_targets[0].float(), reduction='mean')
        loss_prev_ce = self.loss(torch.unsqueeze(torch.tensor(x2[0][0][:, 4]).cuda(), 1), *bbox_targets) # Only uses first two items of bbox_targets
        label, _ = self._expand_onehot_labels(bbox_targets[0], bbox_targets[1], 1)
        loss_new_ce = nn.functional.binary_cross_entropy_with_logits(cls_score, label.float(), reduction='mean')

        s = nn.functional.binary_cross_entropy_with_logits(cls_score, label.float(), reduction='none')
        max_ind = torch.where(s==s.max())
        wandb.log({"loss_prev_ce": loss_prev_ce['loss_cls'].clone().cpu(),
                  "loss_new_ce": loss_bbox['loss_cls'].clone().cpu(),
                   "acc_prev_ce": loss_prev_ce['acc'].clone().cpu(),
                   "acc_new_ce": loss_bbox['acc'].clone().cpu()
                   })


        # loss_prev_focal = self.loss(torch.unsqueeze(torch.tensor(x1[0][0][:, 4]).cuda(),1), *bbox_targets)
        # loss_new_focal = loss_bbox
        # wandb.log({"loss_prev_ce": loss_prev_ce.clone().cpu(),
        #            "loss_new_ce": loss_new_ce.clone().cpu(),
        #            "loss_prev_focal": loss_prev_focal['loss_cls'].clone().cpu(),
        #            "loss_prev_acc": loss_prev_focal["acc"].clone().cpu(),
        #            "loss_new_focal": loss_new_focal['loss_cls'].clone().cpu(),
        #            "loss_new_acc": loss_new_focal["acc"].clone().cpu()
        #            },
        #           )

        img_meta = img_metas[0]
        image = img[0].cpu().detach()
        image = image.permute((1, 2, 0))

        mean = img_meta['img_norm_cfg']['mean']
        std = img_meta['img_norm_cfg']['std']
        image = mmcv.imdenormalize(image.numpy(), mean, std, False)
        image = image.astype(np.uint8)

        from mmdet.core.visualization.image import imshow_det_bboxes
        # imshow_det_bboxes(image, bboxes[0][0][-2:], np.zeros(2, dtype=np.int))
        gts = gt_bboxes[0].detach().cpu().numpy()
        labs = gt_labels[0].detach().cpu().numpy()
        # print()
        # p = imshow_det_bboxes(image, gts, labs, show=False)

        # p = next(model.parameters()).clone()
        # print(torch.all(torch.eq(next(model.parameters()), p)))
        # print(p[0][0])
        # print("Loss Assigner:", time() - t2)


        # o_last = T_out[0][-1]
        # a = x1[0][0][-1]
        # a2 = x1[0][0][1]
        # c = x2[0][0][965]
        # iou_ab = T[-1][0][0]
        # s = m(cls_score)[-1]

        # imshow_det_bboxes(image, np.concatenate((a[np.newaxis, :], c[np.newaxis, :] )), np.array([0, 1]), show=False)

        # maxim = x2[0][0][222]
        # gt3 = np.concatenate((gts[0], [200]))
        # imshow_det_bboxes(image, np.concatenate((maxim[np.newaxis, :], gt3[np.newaxis, :])), np.array([0, 1]), show=False)


        # aa = x1[0][0][25]
        # aa2 = x1[0][0][25]
        # b = x2[0][0][42]
        # gt = np.concatenate((gts[5], [0]))


        # imshow_det_bboxes(image, np.concatenate((gt[np.newaxis, :], aa[np.newaxis, :], b[np.newaxis, :])), np.array([5, 0, 1]), show=False)

        return losses




    # Esto está inspirado en la función de bbox_head.py
    def loss(self,
             cls_score,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):

        # TODO: In the future add bbox_pred
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                losses['loss_cls'] = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                label, _ = self._expand_onehot_labels(labels, label_weights, self.num_classes)
                losses['acc'] = accuracy(cls_score, label.squeeze())
        return losses

    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary infomation.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    @auto_fp16(apply_to=('img', ))
    def forward_test(self, img, img_metas, proposals=None, rescale=False):
        # for model in self.models[:-1]:
        #     x = F.relu(model(x))
        # x = self.models[-1](x) # don't use relu for last model


        # TODO: Check this
        with torch.no_grad():
            t2 = time()
            x1 = self.models[0](img, img_metas, return_loss=False, rescale=True)
            x2 = self.models[1](img, img_metas, return_loss=False, rescale=True)
            print(time()-t2)
        o = [[np.concatenate(r_c) for r_c in zip(*r_img)] for r_img in zip(x1, x2)]

        x1[0][0] = x1[0][0][x1[0][0][:, 4].argsort()][::-1][:1000].copy()
        x2[0][0] = x2[0][0][x2[0][0][:, 4].argsort()][::-1][:1000].copy()


        o_cars = torch.tensor(o[0][0], dtype=torch.float16).cuda()
        print()

        ## TODO: Add center distance

        o_cars = [torch.tensor(x1[0][0]), torch.tensor(x2[0][0])]
        K = x1[0][0].shape[0]
        N = x2[0][0].shape[0]
        F = 3
        T = np.zeros(((K, N, F)),  dtype=np.float32) # TODO: torch.zeros Avoid numpy

        t = time()
        overlaps = bbox_overlaps(o_cars[0][:, :4], o_cars[1][:, :4])
        scores_1 = o_cars[0][:, 4].unsqueeze(1).repeat((1, N))
        scores_2 = o_cars[1][:, 4].unsqueeze(1).repeat((1, K)).T
        T[:, :, 0] = overlaps
        T[:, :, 1] = scores_1
        T[:, :, 2] = scores_2

        T = torch.tensor(T).cuda().half()
        non_empty_indices = torch.nonzero(T[:, :, 0])
        non_empty_indices = torch.nonzero(T[:, :, 0], as_tuple=True)

        # flat_T = T.reshape(-1, F)
        # non_empty_elements = flat_T[torch.nonzero(flat_T[:, 0], as_tuple=True)]

        non_empty_elements = T[non_empty_indices[0], non_empty_indices[1], :]

        non_empty_elements_T = non_empty_elements.permute(1, 0)
        non_empty_elements_T = non_empty_elements_T.unsqueeze(1).unsqueeze(0).cuda()    # Shape [1,3,1, #non-zero]

        T_out = torch.zeros((1, K, N)).cuda().half()

        new_scores = self.fusion(non_empty_elements_T, T_out, non_empty_indices)

        m = nn.Sigmoid()
        new_scores = m(new_scores)
        x1[0][0][:, 4] = new_scores.cpu()


        # print(time()-t)
        # for k in o_cars[0]:
        #     for n in o_cars[1]:
        #         iou = bbox_overlaps(k[:4], n[:4])





        # x2 = [ [x2[0][ ]   ,   []]
        # o = []
        # for r_img in zip(x1, x2):
        #     o_img = []
        #     for r_c in zip(*r_img):
        #         o_c = np.concatenate(r_c)
        #         o_img.append(o_c)
        #     o.append(o_img)

        # [torch.cat(r_c)  for r_c in zip(r_img)   for r_img in zip(x1,x2)]
        # r = torch.cat([x1, x2])

        # return o, x1
        return x1

    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a \
                weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                logger.
                - ``num_samples`` indicates the batch size (when the model is \
                DDP, it means the batch size on each GPU), which is used for \
                averaging the logs.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs



# if __name__ == '__main__':

model = EnsembleModel([None, None])
model = MMDataParallel(model, device_ids=[0]) # Esto mete el modelo Ensemble en la GPU

wandb.watch(model, log='all')
# model.load_state_dict(torch.load("ensemble_fixed_final.pth"))

# l = torch.load("ensemble.pth")
# for key in list(l.keys()):
#     print("module."+key)
#     l["module."+key] = l.pop(key)

cfg = mmcv.Config.fromfile('saved_models/study/faster_rcnn_r50_fpn_fp16_4x1_3e_1280x1920/faster_rcnn_r50_fpn_fp16_4x1_3e_1280x1920.py')
cfg.seed = None

# model.eval()
cfg.data.test.test_mode = True  #To avoid filtering out images without gts

dataset = build_dataset(cfg.data.train)
# dataset = torch.utils.data.Subset(dataset, [44201])
# dataset = torch.utils.data.Subset(dataset, [29822])
# dataset.flag = np.array([1], dtype=np.uint8)
batch = 1
train_detector(model, dataset, cfg)

torch.save(model.state_dict(), "ensemble_2604.pth")


dataset = build_dataset(cfg.data.test)
data_loader = build_dataloader(
        dataset,
        samples_per_gpu=batch,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=True,
        shuffle=False)


res = single_gpu_test(model, data_loader)
# res = mmcv.load("ensemble_res.pkl")
# original_res = mmcv.load("saved_models/study/faster_rcnn_r50_fpn_fp16_4x1_3e_1280x1920/results_sample.pkl")

anns = [dataset.get_ann_info(n) for n in range(len(res))]
mean_ap, eval_results, df_summary = eval_map(res, anns, nproc=4, model_name="Ensemble")


for i in range(len(res)):
    res[i][0] = res[i][0][res[i][0][:, 4] > 0.05]
    res[i][1] = res[i][1][res[i][1][:, 4] > 0.05]
    res[i][2] = res[i][2][res[i][2][:, 4] > 0.05]

cfg = {'type': 'nms', 'iou_threshold': 0.5}
res = [ensembleDetections([dets], cfg) for dets in res]

for i in range(len(res)):
    res[i][0] = res[i][0][:100]
    res[i][1] = res[i][1][:100]
    res[i][2] = res[i][2][:100]

mean_ap, eval_results, df_summary = eval_map(res, anns, nproc=4, model_name="Ensemble")

