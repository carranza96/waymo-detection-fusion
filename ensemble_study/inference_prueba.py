from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from time import time, perf_counter
import numpy as np
import torch
from torch import nn
import mmcv
import warnings

from mmdet.core import get_classes
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmcv.parallel import MMDataParallel
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


def init_detector(config, checkpoint=None, device='cuda:0', cfg_options=None):
    """Initialize a detector from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        cfg_options (dict): Options to override some settings in the used
            config.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)
    config.model.pretrained = None
    config.model.train_cfg = None
    model = build_detector(config.model, test_cfg=config.get('test_cfg'))
    fp16_cfg = config.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    if checkpoint is not None:
        map_loc = 'cpu' if device == 'cpu' else None
        checkpoint = load_checkpoint(model, checkpoint, map_location=map_loc)
        if 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            warnings.simplefilter('once')
            warnings.warn('Class names are not saved in the checkpoint\'s '
                          'meta data, use COCO classes by default.')
            model.CLASSES = get_classes('coco')
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


def inference_detector(model, img, cfg):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        If imgs is a str, a generator will be returned, otherwise return the
        detection results directly.
    """
    device = next(model.parameters()).device  # model device
    # prepare data
    if isinstance(img, np.ndarray):
        # directly add img
        data = dict(img=img)
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'
    else:
        # add information into dict
        data = dict(img_info=dict(filename=img), img_prefix=None)
    # build the data pipeline
    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    # just get the actual data from DataContainer
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    # forward the model
    with torch.no_grad():
        t = time()
        result = model(return_loss=False, rescale=True, **data)[0]
        t2 = time()-t
    return result, t2


config_file = 'saved_models/study/faster_rcnn_r50_fpn_fp16_4x1_3e_1280x1920/faster_rcnn_r50_fpn_fp16_4x1_3e_1280x1920.py'
checkpoint_file = 'saved_models/study/faster_rcnn_r50_fpn_fp16_4x1_3e_1280x1920/latest.pth'
device = 'cuda:0'
# init a detector
model = init_detector(config_file, checkpoint_file, device=device)
cfg = model.cfg

model1 = model

config_file = 'saved_models/study/retinanet_r50_fpn_fp16_4x1_3e_1280x1920/retinanet_r50_fpn_fp16_4x1_3e_1280x1920.py'
checkpoint_file = 'saved_models/study/retinanet_r50_fpn_fp16_4x1_3e_1280x1920/latest.pth'
model2 = init_detector(config_file, checkpoint_file, device=device)

# model = MMDataParallel(model, device_ids=[0])



class Fusion(nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()
        self.name = 'fusion_layer'
        self.corner_points_feature = Sequential(
            nn.Conv2d(24,48,1),
            nn.ReLU(),
            nn.Conv2d(48,96,1),
            nn.ReLU(),
            nn.Conv2d(96,96,1),
            nn.ReLU(),
            nn.Conv2d(96,4,1),
        )
        self.fuse_2d_3d = Sequential(
            nn.Conv2d(3, 18, 1),
            nn.ReLU(),
            nn.Conv2d(18,36,1),
            nn.ReLU(),
            nn.Conv2d(36,36,1),
            nn.ReLU(),
            nn.Conv2d(36,1,1),
        )
        self.maxpool = Sequential(
            nn.MaxPool2d([1, 98], 1),
        )


    def forward(self, input_1, T_out, T_indices):
        # flag = -1
        # if tensor_index[0, 0] == -1:
        #     out_1 = torch.zeros(1,200,70400,dtype = input_1.dtype,device = input_1.device)
        #     out_1[:,:,:] = -9999999
        #     flag = 0
        # else:
        #     x = self.fuse_2d_3d(input_1)
        #     out_1 = torch.zeros(1,200,70400,dtype = input_1.dtype,device = input_1.device)
        #     out_1[:,:,:] = -9999999
        #     out_1[:,tensor_index[:,0],tensor_index[:,1]] = x[0,:,0,:]
        #     flag = 1
        # x = self.maxpool(out_1)
        # #x, _ = torch.max(out_1,1)
        # x = x.squeeze().reshape(1,-1,1)
        # return x, flag

        x = self.fuse_2d_3d(input_1)

        T_out[:, :, :] = -9999999
        T_out[:, T_indices[0], T_indices[1]] = x[0, 0, :, :]

        mp = nn.MaxPool2d([1, 98], 1)
        mp(T_out)
        x = self.maxpool(T_out)

        return x.squeeze()



class EnsembleModel(nn.Module):

    def __init__(self, models):
        super(EnsembleModel, self).__init__()
        self.models = models
        self.fusion = Fusion()

    # @auto_fp16(apply_to=('img', ))
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


    def forward_train(self, img, img_metas, proposals=None, rescale=False):


    def forward_test(self, img, img_metas, proposals=None, rescale=False):
        # for model in self.models[:-1]:
        #     x = F.relu(model(x))
        # x = self.models[-1](x) # don't use relu for last model


        # TODO: Check this
        with torch.no_grad():
            x1 = self.models[0](img, img_metas, return_loss=False, rescale=True)
            x2 = self.models[1](img, img_metas, return_loss=False, rescale=True)

        o = [[np.concatenate(r_c) for r_c in zip(*r_img)] for r_img in zip(x1, x2)]

        o_cars = torch.tensor(o[0][0], dtype=torch.float32).cuda()
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

        T = torch.tensor(T).cuda()
        non_empty_indices = torch.nonzero(T[:, :, 0])
        non_empty_indices = torch.nonzero(T[:, :, 0], as_tuple=True)

        # flat_T = T.reshape(-1, F)
        # non_empty_elements = flat_T[torch.nonzero(flat_T[:, 0], as_tuple=True)]

        non_empty_elements = T[non_empty_indices[0], non_empty_indices[1], :]

        non_empty_elements_T = non_empty_elements.permute(1, 0)
        non_empty_elements_T = non_empty_elements_T.unsqueeze(1).unsqueeze(0).cuda()    # Shape [1,3,1, #non-zero]

        T_out = torch.zeros((1, K, N)).cuda()

        new_scores = self.fusion(non_empty_elements_T, T_out, non_empty_indices)
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

        return o


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

# TODO: Put model on GPU
model = EnsembleModel([model1, model2])
model = MMDataParallel(model, device_ids=[0])


cfg.seed = None




# model.eval()

dataset = build_dataset(cfg.data.test)
batch = 1
train_detector(model, dataset, cfg)
data_loader = build_dataloader(
        dataset,
        samples_per_gpu=batch,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)


# for i, data in enumerate(data_loader):
#     y_pred = model(data)
# d = next(iter(data_loader))
# model(d)


times = []
for i, data in enumerate(data_loader):
    with torch.no_grad():
        # d, t = inference_detector(model, "data/waymococo_f0/val2020/val_00000_00000_camera1.jpg", cfg)
        t2 = time()
        result = model(**data, return_loss=False)
        t = time()-t2
        print(time()-t2)
        # print(t)
        if i>5:
            times.append(t)

    if i == 500:
        break
#
# inference the demo image

# times = []
# for i, data in enumerate(data_loader):
# # for i in range(200):
#     with torch.no_grad():
#         # d, t = inference_detector(model, "data/waymococo_f0/val2020/val_00000_00000_camera1.jpg", cfg)
#         t2 = time()
#         result = model(return_loss=False, rescale=True, **data)
#         t = time()-t2
#         print(time()-t2)
#         # print(t)
#         if i>5:
#             times.append(t)
#
#     if i == 100:
#         break
#
# # show_result_pyplot(model, "data/waymococo_f0/val2020/val_00000_00000_camera1.jpg", d)
# print(len(times)*batch/np.sum(times))
# print(np.sum(times))
