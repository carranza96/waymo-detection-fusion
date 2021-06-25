import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import Sequential
import mmcv
from mmcv.runner import auto_fp16, force_fp32
from mmdet.models.losses import accuracy
from mmdet.models.builder import build_loss
from mmdet.core import (build_assigner, build_sampler, multi_apply)
from mmdet.core.bbox.iou_calculators import bbox_overlaps
from mmdet.core.bbox.transforms import bbox_xyxy_to_cxcywh
from mmdet.core.visualization.image import imshow_det_bboxes
from collections import OrderedDict
from time import time
from copy import deepcopy
from ensemble import ensembleDetections
from neural_ensemble_utils import postprocess_detections
import wandb
import matplotlib.pyplot as plt
import seaborn as sns

class Fusion(nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()
        self.name = 'fusion_layer'
        self.mode = 'max'

        self.fuse = Sequential(
            nn.Conv2d(9, 18, 1),
            nn.ReLU(),
            nn.Conv2d(18, 36, 1),
            nn.ReLU(),
            nn.Conv2d(36, 36, 1),
            nn.ReLU(),
            nn.Conv2d(36, 1, 1),
        )
        self.maxpool = Sequential(
            nn.MaxPool2d([1, 1000], 1),
        )
        self.maxpool2 = Sequential(
            nn.MaxPool2d([2000, 1], 1),
        )


    def forward(self, input, T_out, T_indices):
        #     out_1 = torch.zeros(1,200,70400,dtype = input_1.dtype,device = input_1.device)
        #     out_1[:,:,:] = -9999999
        #     out_1[:,tensor_index[:,0],tensor_index[:,1]] = x[0,:,0,:]
        #     flag = 1
        # x = self.maxpool(out_1)
        # #x, _ = torch.max(out_1,1)
        # x = x.squeeze().reshape(1,-1,1)
        # return x, flag

        x = self.fuse(input)

        T_out[:, :, :] = -9999.
        T_out[:, T_indices[0], T_indices[1]] = x[0, 0, :, :]

        if self.mode=='max':
            x = self.maxpool(T_out)
            x2 = self.maxpool2(T_out)
        else:
            mask = T_out != -9999.
            x = (T_out*mask).sum(dim=2) / mask.sum(dim=2)
            x2 = (T_out*mask).sum(dim=1) / mask.sum(dim=1)

        return x.squeeze(), x2.squeeze()





class EnsembleModel(nn.Module):

    def __init__(self, models, n_dets1=2000, n_dets2=1000, log_wandb=False):
        super(EnsembleModel, self).__init__()
        self.models = models
        self.fusion = Fusion()

        self.fusion.maxpool = Sequential(
            nn.MaxPool2d([1, n_dets2], 1),
        )
        self.fusion.maxpool2 = Sequential(
            nn.MaxPool2d([n_dets1, 1], 1),
        )

        self.num_classes = 1
        loss_cls = dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0)

        # loss_cls=dict(
        #     type='FocalLoss',
        #     use_sigmoid=True,
        #     gamma=2.0,
        #     alpha=0.25,
        #     loss_weight=1.0)

        assigner = dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            match_low_quality=False,
            ignore_iof_thr=-1)

        # assigner=dict(
        #     type='HungarianAssigner',
        #     cls_cost=dict(type='ClassificationCost', weight=1.),
        #     reg_cost=dict(type='BBoxL1Cost', weight=1.0),
        #     iou_cost=dict(type='IoUCost', iou_mode='giou', weight=1.0))

        self.assigner = build_assigner(assigner)
        self.loss_cls = build_loss(loss_cls)

        sampler_cfg = dict(type='PseudoSampler')
        self.sampler = build_sampler(sampler_cfg)

        self.saved_preds = None
        self.n_dets1 = n_dets1
        self.n_dets2 = n_dets2
        self.log_wandb = log_wandb

    # TODO: Import from cross_entropy_loss.py
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

    def get_image(self, img, img_metas):

        img_meta = img_metas[0]
        image = img[0].cpu().detach()
        image = image.permute((1, 2, 0))

        mean = img_meta['img_norm_cfg']['mean']
        std = img_meta['img_norm_cfg']['std']
        image = mmcv.imdenormalize(image.numpy(), mean, std, False)
        image = image.astype(np.uint8)

        return image

    # from mmdet.models.roi_heads.bbox_head.bbox_head
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
            labels[:num_pos] = pos_gt_labels # + 1.
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

    def get_bbox_targets(self, bboxes, gt_bboxes, gt_labels, gt_bboxes_ignore, img_metas):

        num_imgs = len(img_metas)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
        sampling_results = []
        for i in range(num_imgs):
            assign_result = self.assigner.assign(
                torch.tensor(bboxes[i][0]).cuda(), gt_bboxes[i], gt_bboxes_ignore[i],
                gt_labels[i])
            # assign_result = self.assigner.assign(
            #     bbox_pred=torch.tensor(bboxes[i][0]).cuda()[:, :4], cls_pred=torch.unsqueeze(torch.tensor(bboxes[i][0]).cuda()[:, 4], 1),
            #     gt_bboxes=gt_bboxes[i], gt_bboxes_ignore= gt_bboxes_ignore[i], gt_labels=gt_labels[i],
            #     img_meta=img_metas[0])
            sampling_result = self.sampler.sample(
                assign_result,
                torch.tensor(bboxes[i][0]).cuda(),
                gt_bboxes[i])
            # gt_labels[i])
            sampling_results.append(sampling_result)

        bbox_targets = self.get_targets(sampling_results, gt_bboxes,
                                        gt_labels, rcnn_train_cfg=None)

        # El sampler ordena primeros los N targets positivos y luego los N negativos,
        # hay que reordenar la salida de labels para que coincida con el orden de las detecciones originales
        order_inds = torch.cat((sampling_result.pos_inds, sampling_result.neg_inds))
        original_order = bbox_targets[0].clone()
        bbox_targets[0][order_inds] = original_order
        # bbox_targets_s = (bbox_targets[0][order_inds], bbox_targets[1][order_inds], bbox_targets[2], bbox_targets[3])

        return bbox_targets, sampling_results, assign_result

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

    def preds_base_models(self, img, img_metas):
        t = time()
        x1 = self.models[0](img, img_metas, return_loss=False, rescale=True)
        # print("Model 1 inference:", time() - t)
        t2 = time()
        x2 = self.models[1](img, img_metas, return_loss=False, rescale=True)
        # print("Model 2 inference:", time()-t2)
        # print("Total inference time:", time()-t)

        return x1, x2

    def create_T(self, x1, x2, img_metas):
        tt = time()
        # Only batch of one image and predictions from class 0 (vehicles)
        o_vehicles = [torch.tensor(x1[0][0]).cuda(), torch.tensor(x2[0][0]).cuda()]

        # Get shapes and construct T tensor
        K = x1[0][0].shape[0]
        N = x2[0][0].shape[0]
        F = 9
        T = np.zeros((K, N, F), dtype=np.float16)  # TODO: torch.zeros Avoid numpy Float 16
        # T = np.zeros((K, N, F))

        # Calculate overlaps between predictions
        t = time()
        overlaps = bbox_overlaps(o_vehicles[0][:, :4], o_vehicles[1][:, :4])    # TODO: Calculate with FP16
        # print("-----------------------------------")
        # print("BBox overlaps:", time() - t)

        t = time()
        scores_1 = o_vehicles[0][:, 4].unsqueeze(1).repeat((1, N))
        scores_2 = o_vehicles[1][:, 4].unsqueeze(1).repeat((1, K)).T


        # TODO: Log-likelihood score
        scores_1 = torch.log(scores_1/(1-scores_1))
        scores_2 = torch.log(scores_2/(1-scores_2))


        bboxes_1 = o_vehicles[0][:, :4]
        bboxes_2 = o_vehicles[1][:, :4]
        img_h, img_w = img_metas[0][0]['img_shape'][0], img_metas[0][0]['img_shape'][1]
        # Normalize boxes
        bboxes_1[:, [0, 2]], bboxes_1[:, [1, 3]] = bboxes_1[:, [0, 2]]/img_w, bboxes_1[:, [1, 3]]/img_h
        bboxes_2[:, [0, 2]], bboxes_2[:, [1, 3]] = bboxes_2[:, [0, 2]]/img_w, bboxes_2[:, [1, 3]]/img_h

        bboxes_1 = bbox_xyxy_to_cxcywh(bboxes_1)
        bboxes_2 = bbox_xyxy_to_cxcywh(bboxes_2)
        # print("Parte1:", time() - t)

        t = time()
        # Scale difference
        w1, w2 = bboxes_1[:, 2].unsqueeze(1).repeat((1, N)), bboxes_2[:, 2].unsqueeze(1).repeat((1, K)).T
        h1, h2 = bboxes_1[:, 3].unsqueeze(1).repeat((1, N)), bboxes_2[:, 3].unsqueeze(1).repeat((1, K)).T
        w_diff = torch.log(w1 / w2)
        h_diff = torch.log(h1 / h2)
        aspect_diff = torch.log(w1 / h1) - torch.log(w2 / h2)


        # Distance x, y
        cx1, cx2 = bboxes_1[:, 0].unsqueeze(1).repeat((1, N)), bboxes_2[:, 0].unsqueeze(1).repeat((1, K)).T
        cy1, cy2 = bboxes_1[:, 1].unsqueeze(1).repeat((1, N)), bboxes_2[:, 1].unsqueeze(1).repeat((1, K)).T
        x_dist = cx1-cx2
        y_dist = cy1-cy2
        l2_dist = torch.sqrt(x_dist**2 + y_dist**2)
        # print("Parte2:", time() - t)
        #
        # t = time()
        # T[:, :, 0] = overlaps.cpu()
        # T[:, :, 1] = scores_1.cpu()
        # T[:, :, 2] = scores_2.cpu()
        # T[:, :, 3] = x_dist.cpu()
        # T[:, :, 4] = y_dist.cpu()
        # T[:, :, 5] = l2_dist.cpu()
        # T[:, :, 6] = w_diff.cpu()
        # T[:, :, 7] = h_diff.cpu()
        # T[:, :, 8] = aspect_diff.cpu()
        #
        # T = torch.tensor(T).cuda()
        # print("Parte3:", time() - t)

        t = time()
        T = torch.stack((overlaps, scores_1, scores_2, x_dist, y_dist, l2_dist, w_diff, h_diff, aspect_diff), dim=2)
        # print("Parte3:", time() - t)
        # print("Hasta Parte3:", time() - tt)

        t = time()
        # Fill last element of column with all IoU zeros with -1
        # non_overlapping_dets = ~overlaps.sum(dim=0).bool()
        non_overlapping_dets = ~torch.any(torch.greater_equal(T[:, :, 0], 0.2), dim=0)

        T[-1, non_overlapping_dets, 0] = -1   # IoU -1
        T[-1, non_overlapping_dets, 1] = -1  # Score Retina -1
        # print("Parte4:", time() - t)

        t = time()
        # TODO: Maybe increase threshold to 0.2
        # non_empty_indices = torch.nonzero(T[:, :, 0], as_tuple=True)
        non_empty_indices = torch.where((torch.greater_equal(T[:, :, 0], 0.2)) | (torch.less(T[:, :, 0], 0)))

        non_empty_elements = T[non_empty_indices[0], non_empty_indices[1], :]

        non_empty_elements_T = non_empty_elements.permute(1, 0)
        non_empty_elements_T = non_empty_elements_T.unsqueeze(1).unsqueeze(0).cuda()  # Shape [1, 9, 1, #non-zero]
        # print("Parte5:", time() - t)
        # print("Hasta Parte5:", time() - tt)


        # TODO: T_out tensor very slow
        t = time()
        T_out = torch.zeros((1, K, N)).cuda()
        # print("Parte6:", time() - t)
        # print("Hasta Parte6:", time() - tt)
        #

        # T_out = T_out.cuda()


        return T, T_out, non_empty_elements_T, non_empty_indices

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        t = time()
        id = img_metas[0]['id']
        # X1: RetinaNet, X2: FasterRCNN (Supongo que X2 es el mejor modelo) Shape: [# images, #n_classes, # n_boxes]
        if self.saved_preds:
            x1, x2 = [self.saved_preds[0][id]], [self.saved_preds[1][id]]
        else:
            with torch.no_grad():
                # torch.backends.cudnn.enabled = False  # This solves the error of using different types of GPU
                x1, x2 = self.preds_base_models([img], [img_metas])
        # print("Read:", time()-t)
        # o = [[np.concatenate(r_c) for r_c in zip(*r_img)] for r_img in zip(x1, x2)]
        # o_cars = torch.tensor(o[0][0], dtype=torch.float16).cuda()

        x1[0][0] = x1[0][0][x1[0][0][:, 4].argsort()][::-1][:self.n_dets1].copy()
        x2[0][0] = x2[0][0][x2[0][0][:, 4].argsort()][::-1][:self.n_dets2].copy()

        # TODO: Change scores to log-likelihood
        # TODO: Esta parte (Tensor preparation) es muy lenta
        t = time()
        T, T_out, non_empty_elements_T, non_empty_indices = self.create_T(x1, x2, [img_metas])
        # print("Tensor preparation:", time() - t)


        # Fusion network
        t = time()
        fusion_scores1, fusion_scores2 = self.fusion(non_empty_elements_T, T_out, non_empty_indices)

        # m = nn.Sigmoid()
        # fusion_scores_2n = []
        # for i in range(1000):
        #     if i == 221:
        #         print()
        #     tout = T_out[0, T_out[0, :, i] > -10, i]
        #     t_overlap_ind = torch.where(T_out[0, :, i] > -10)[0]
        #     t0 = T[t_overlap_ind, i]
        #     ind02 = torch.where(t0[:, 0] > 0.2)[0]
        #     tout_filt = tout[ind02]
        #     res = m(tout_filt.mean())
        #     if res.isnan():
        #         fusion_scores_2n.append(m(fusion_scores2[i]))
        #     else:
        #         fusion_scores_2n.append(res)
        # fusion_scores_2n = torch.tensor(fusion_scores_2n).cuda()
        #
        # fusion_scores2 = torch.log(fusion_scores_2n/(1-fusion_scores_2n))

        # print("Fusion:", time() - t)
        # TODO: Uncomment
        # x1[0][0][:, 4] = new_scores.cpu().detach().numpy()

        # Training: bbox assignment and loss calculation
        # Filter GT bboxes of class 0 (vehicles)
        gt_bboxes_vehicles = [bb[gt_labels[i] == 0] for i, bb in enumerate(gt_bboxes)]
        gt_labels_vehicles = [lab[lab == 0] for lab in gt_labels]
        bbox_targets1, sampling_results1, assign_result1 = self.get_bbox_targets(x1, gt_bboxes_vehicles, gt_labels_vehicles, gt_bboxes_ignore, img_metas)
        bbox_targets2, sampling_results2, assign_result2 = self.get_bbox_targets(x2, gt_bboxes_vehicles, gt_labels_vehicles, gt_bboxes_ignore, img_metas)


        # cls_score = torch.tensor(1 - x1[0][0][:, 4], requires_grad=True).cuda()
        # cls_score = 1 - new_scores
        old_scores1 = torch.tensor(x1[0][0][:, 4]).cuda().clone()
        old_scores = torch.tensor(x2[0][0][:, 4]).cuda().clone()
        cls_score = fusion_scores2
        cls_score = torch.unsqueeze(cls_score, 1)
        bbox_targets, sampling_results = bbox_targets2, sampling_results2

        losses = dict()
        loss_bbox = self.loss(cls_score, *bbox_targets)
        loss_bbox2 = self.loss(torch.unsqueeze(fusion_scores1, 1), *bbox_targets1)
        if loss_bbox['loss_cls'] > 1:
            print()

        loss_bbox['loss_cls'] = loss_bbox['loss_cls'] #+ loss_bbox2['loss_cls']
        # TODO: Para añadir el loss del primer detector hay que controlar los boxes que no tienen overlapping con el detector 2
        losses.update(loss_bbox)

        # Check loss and compare
        inv_sigmoid = torch.log(old_scores/(1-old_scores))
        loss_bbox_prev = self.loss(torch.unsqueeze(inv_sigmoid, 1), *bbox_targets) # Only uses first two items of bbox_targets
        # loss_bbox_prev = self.loss(old_scores, *bbox_targets) # Only uses first two items of bbox_targets
        label, _ = self._expand_onehot_labels(bbox_targets[0], bbox_targets[1], 1)
        # TODO: Check loss is calculated correctly
        loss_prev_ce = nn.functional.binary_cross_entropy(torch.unsqueeze(old_scores, 1), label.float(), reduction='mean')
        loss_new_ce = nn.functional.binary_cross_entropy_with_logits(cls_score, label.float(), reduction='mean')
        # print(loss_bbox)

        if self.log_wandb:
            wandb.log({"loss_prev_ce": loss_bbox_prev['loss_cls'].clone().cpu(),
                      "loss_new_ce": loss_bbox['loss_cls'].clone().cpu(),
                       "acc_prev_ce": loss_bbox_prev['acc'].clone().cpu(),
                       "acc_new_ce": loss_bbox['acc'].clone().cpu()
                       })
        # print("Loss:", time() - t)

        # Individual losses
        # loss_per_det = nn.functional.binary_cross_entropy_with_logits(cls_score, label.float(), reduction='none')
        # max_loss_ind = torch.where(loss_per_det == loss_per_det.max())[0]
        # #
        # # Plot image
        # image = self.get_image(img, img_metas)
        # #
        # gts = gt_bboxes[0].detach().cpu().numpy()
        # labs = gt_labels[0].detach().cpu().numpy()
        # # imshow_det_bboxes(image, gts, labs)
        # #
        # #
        # max_loss = loss_per_det[max_loss_ind]
        # m = nn.Sigmoid()
        # max_loss_old_sc = old_scores[max_loss_ind]
        # max_loss_sc = m(cls_score[max_loss_ind])
        # max_loss_bbox = x2[0][0][max_loss_ind][:4]
        #
        # target_class = bbox_targets[0][max_loss_ind]  # 0: Vehicle, 1: Background
        #
        # if target_class == 0:
        #     target_gt_ind = sampling_results[0].pos_assigned_gt_inds[torch.where(sampling_results[0].pos_inds == max_loss_ind)[0]]
        #     target_gt = gts[target_gt_ind]
        # print()
        #
        # t_out = T_out[0, :, max_loss_ind]
        # t_out[t_out > -100]
        # t = T[:, max_loss_ind].squeeze(1)
        # t_overlap = t[t[:, 0] > 0.2]
        # imshow_det_bboxes(image, np.concatenate((np.expand_dims(max_loss_bbox, 0), np.expand_dims(target_gt, 0))), np.array([0, 1]), show=False)
        # imshow_det_bboxes(image, np.expand_dims(max_loss_bbox, 0), np.array([0]), show=False)
        #
        #
        # i = max_loss_ind[0]
        # # i = 434
        # # i = 250
        # tout = T_out[0, T_out[0, :, i] > -10, i]
        # t_overlap_ind = torch.where(T_out[0, :, i] > -10)[0]
        # t0 = T[t_overlap_ind, i]
        # ind02 = torch.where(t0[:, 0] > 0.2)[0]
        # tout_filt = tout[ind02]
        # plt.figure()
        # sns.scatterplot(t0[:, 0].detach().cpu().numpy(), tout.detach().cpu().numpy())


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

        # p = next(model.parameters()).clone()
        # print(torch.all(torch.eq(next(model.parameters()), p)))
        # print(p[0][0])


        return losses

    def forward_test(self, img, img_metas, proposals=None, rescale=False, two_outputs=False):

        # if not isinstance(img, list):
        #     img, img_metas = [img], [img_metas]

        image = self.get_image(img[0], img_metas[0])
        # plt.imshow(image)
        # plt.show()


        with torch.no_grad():
            x1, x2 = self.preds_base_models(img, img_metas)

        # o = [[np.concatenate(r_c) for r_c in zip(*r_img)] for r_img in zip(x1, x2)]

        x1[0][0] = x1[0][0][x1[0][0][:, 4].argsort()][::-1][:self.n_dets1].copy()
        x2[0][0] = x2[0][0][x2[0][0][:, 4].argsort()][::-1][:self.n_dets2].copy()

        T, T_out, non_empty_elements_T, non_empty_indices = self.create_T(x1, x2, img_metas)

        fusion_scores1, fusion_scores2 = self.fusion(non_empty_elements_T, T_out, non_empty_indices)

        m = nn.Sigmoid()
        sc1 = m(fusion_scores1)
        sc2 = m(fusion_scores2)


        # fusion_scores_2n = []
        # for i in range(1000):
        #     tout = T_out[0, T_out[0, :, i] > -10, i]
        #     t_overlap_ind = torch.where(T_out[0, :, i] > -10)[0]
        #     t0 = T[t_overlap_ind, i]
        #     ind02 = torch.where(t0[:, 0] > 0.2)[0]
        #     tout_filt = tout[ind02]
        #     res = m(tout_filt.mean())
        #     if res.isnan():
        #         fusion_scores_2n.append(m(fusion_scores2[i]))
        #     else:
        #         fusion_scores_2n.append(res)
        # fusion_scores_2n = torch.tensor(fusion_scores_2n).cuda()
        # sc2 = fusion_scores_2n

        x1old = deepcopy(x1)
        x1[0][0][:, 4] = sc1.cpu()
        x2[0][0][:, 4] = sc2.cpu()

        # print(time()-t)
        # for k in o_cars[0]:
        #     for n in o_cars[1]:
        #         iou = bbox_overlaps(k[:4], n[:4])

        # cfg = {'type': 'nms', 'iou_threshold': 0.5}
        # d1 = [ensembleDetections([dets], cfg) for dets in x1]
        # d2 = [ensembleDetections([dets], cfg) for dets in x2]
        #
        # d1 = d1[0][0][d1[0][0][:, 4] > 0.05]
        # d2 = d2[0][0][d2[0][0][:, 4] > 0.05]
        #
        # box = x1[0][0][15]
        # box_old = x1old[0][0][15]
        # # neig = torch.where(T[15][:, 0])[0]
        # neig = torch.where((torch.greater_equal(T[15][:, 0], 0.2)) | (torch.less(T[15][:, 0], 0)))
        # maxsc_neig = T_out[0][15][881]
        # T_neig = T[15][881]
        #
        # box2 = x2[0][0][881]

        # imshow_det_bboxes(image, np.concatenate((box[np.newaxis, :], box2[np.newaxis,:])), np.array([0,0]), show=False)

        # Postprocess predictions
        if not two_outputs:
            score_th = 0.05
            nms_cfg = {'type': 'nms', 'iou_threshold': 0.5}
            max_dets_class = 100

            x2 = postprocess_detections(x2, score_th, nms_cfg, max_dets_class)
            return x2

        else:
            return x1, x2


    # from mmdet.models.roi_head.bbox_heads.bbox_head
    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
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
                # TODO: introduce cls_score already with sigmoid activation to avoid this code and the change in cross_entropy
                # label, _ = self._expand_onehot_labels(labels, label_weights, self.num_classes)
                m = nn.Sigmoid()
                sg_cls_score = m(cls_score).repeat(1, 2).clone()
                sg_cls_score[:, 1] = 1 - sg_cls_score[:, 0]
                losses['acc'] = accuracy(sg_cls_score, labels)
                losses['acc_vehicles'] = accuracy(sg_cls_score[labels==0], labels[labels==0])
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

