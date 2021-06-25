import os.path as osp
import torch
import mmcv
from mmcv.image import tensor2imgs
from mmdet.core import encode_mask_results
from mmdet.apis.inference import init_detector
from ensemble import ensembleDetections


def init_base_models(base_models_names):
    base_models = []
    for m in base_models_names:
        config_file = 'saved_models/study/{}/{}.py'.format(m, m)
        config = mmcv.Config.fromfile(config_file)

        # Disable NMS and thresholds of base models
        if 'rcnn' in config.model.test_cfg.keys():
            config.model.test_cfg.rcnn.score_thr = 0.0
            config.model.test_cfg.rcnn.max_per_img = -1
            config.model.test_cfg.rcnn.nms = None
        else:
            config.model.test_cfg.score_thr = 0.0
            config.model.test_cfg.max_per_img = -1
            config.model.test_cfg.nms = None


        checkpoint_file = 'saved_models/study/{}/latest.pth'.format(m)
        bm = init_detector(config, checkpoint_file, device='cuda:0')
        base_models.append(bm)
    return base_models


def load_saved_preds_base_models(base_models_names, file_name):
    saved_preds = []
    for m in base_models_names:
        sp = mmcv.load("saved_models/study/{}/{}".format(m,file_name))
        saved_preds.append(sp)
    return saved_preds


def single_gpu_test_two_outputs(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    model.eval()
    results, results2 = [], []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result, result2 = model(return_loss=False, rescale=True, two_outputs=True, **data)

        batch_size = len(result)
        if show or out_dir:
            if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result[i],
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr)

        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
        results.extend(result)
        results2.extend(result2)
        for _ in range(batch_size):
            prog_bar.update()
    return results, results2


def postprocess_detections(res, score_th, nms_cfg, max_dets_class):
    for i in range(len(res)):
        for j in range(len(res[i])):
            res[i][j] = res[i][j][res[i][j][:, 4] > score_th]


    res = [ensembleDetections([dets], nms_cfg) for dets in res]

    for i in range(len(res)):
        for j in range(len(res[i])):
            res[i][j] = res[i][j][:max_dets_class]

    return res