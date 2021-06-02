import mmcv
import torch
import wandb
import numpy as np
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from tools.train import train_detector
from mmdet.datasets import build_dataloader, build_dataset
from mean_ap import eval_map
from ensemble import ensembleDetections
from neural_ensemble import EnsembleModel
from neural_ensemble_utils import init_base_models, load_saved_preds_base_models, \
    single_gpu_test_two_outputs, postprocess_detections
from mmcv.runner import wrap_fp16_model, load_checkpoint
from curves import score_th_results

log_wandb = False
train = True
load_saved_preds = True
# ensemble_checkpoint = 'ensemble/epoch_12300.pth'
ensemble_checkpoint = None
save_dir = 'ensemble/'
pkl_path = save_dir + 'results.pkl'
base_models_names = ['retinanet_r50_fpn_fp16_4x2_1x_1280x1920', 'faster_rcnn_r50_fpn_fp16_4x2_1x_1280x1920']
# base_models_names = ['faster_rcnn_r50_fpn_fp16_4x2_1x_1280x1920', 'faster_rcnn_r50_fpn_fp16_4x2_1x_1280x1920_flip_test']

n_dets1, n_dets2 = 1000, 1000


## Load base models and ensemble model (with checkpoint if provided)
## TODO: Each base model in different GPU
base_models = init_base_models(base_models_names)
model = EnsembleModel(base_models, n_dets1, n_dets2, log_wandb)

saved_preds = load_saved_preds_base_models(base_models_names, 'results_example_all.pkl') if load_saved_preds and train else None
model.saved_preds = saved_preds
# Wrap FP16 base models
for m in model.models:
    wrap_fp16_model(m)
model = MMDataParallel(model, device_ids=[0])  # Esto mete el modelo Ensemble en la GPU

if log_wandb:
    wandb.init(project='ensemble_od', entity='minerva')
    wandb.watch(model, log='all')


cfg = mmcv.Config.fromfile('ensemble_study/neural_ensemble_cfg.py')
cfg.load_from = ensemble_checkpoint

## Get cfg of one base model and set test_mode to avoid filtering out images without gts
# TODO: Write here complete data dict
# cfg = base_models[0].cfg
# cfg.seed = None
# cfg.data.train.filter_empty_gt = False
# cfg.data.test.test_mode = True
# cfg.data.samples_per_gpu = 1
# cfg.data.train.pipeline = [d if d['type'] != 'RandomFlip' else {'type': 'RandomFlip', 'flip_ratio': 0.} for d in cfg.data.train.pipeline] # No Horizontal Flip
# cfg.runner.max_epochs = 3
# cfg.lr_config.step = []
# cfg.gpu_ids = [0]
# cfg.load_from = None
# cfg.fp16 = None
# fp16_cfg = cfg.get('fp16', None)
# if fp16_cfg is not None:
#     wrap_fp16_model(model)

if train:
    if ensemble_checkpoint:
        model.load_state_dict(torch.load(ensemble_checkpoint))

    # saved_preds = load_saved_preds_base_models(base_models_names, 'results_example_all.pkl') if load_saved_preds else None
    # model.saved_preds = saved_preds

    dataset = build_dataset(cfg.data.train)
    # Filter dataset by specific image index
    # dataset = torch.utils.data.Subset(dataset, [44201])
    # dataset = torch.utils.data.Subset(dataset, [29822])
    # dataset.flag = np.array([1], dtype=np.uint8)

    train_detector(model, dataset, cfg, validate=True)

else:
    if ensemble_checkpoint:
        checkpoint = load_checkpoint(model, ensemble_checkpoint)
    # model.load_state_dict(torch.load(ensemble_checkpoint))

    dataset = build_dataset(cfg.data.test)
    anns = [dataset.get_ann_info(n) for n in range(len(dataset))]

    # Filter dataset by specific image index
    # ind = 32
    # anns, dataset = [anns[ind]], torch.utils.data.Subset(dataset, [ind])
    # dataset.flag = np.array([1], dtype=np.uint8)

    # TODO: Allow larger batch
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=True,
        shuffle=False)

    # Get predictions
    # res = single_gpu_test(model, data_loader)
    res1, res2 = single_gpu_test_two_outputs(model, data_loader)
    #
    # # res1, res2 = mmcv.load('ensemble/results1.pkl'), mmcv.load('ensemble/results2.pkl')
    # Postprocess predictions
    score_th = 0.05
    nms_cfg = {'type': 'nms', 'iou_threshold': 0.5}
    max_dets_class = 100

    # TODO: Do NMS exactly as in MMDetection
    res1 = postprocess_detections(res1, score_th, nms_cfg, max_dets_class)
    res2 = postprocess_detections(res2, score_th, nms_cfg, max_dets_class)

    mmcv.dump(res1, 'ensemble/results1.pkl')
    mmcv.dump(res2, 'ensemble/results2.pkl')


    # Evaluate
    original_res = mmcv.load("saved_models/study/{}/results_example.pkl".format(base_models_names[0]))
    original_res2 = mmcv.load("saved_models/study/{}/results_example.pkl".format(base_models_names[1]))

    # COCO Evaluation
    # original_res2[0][0] = np.expand_dims(original_res2[0][0][5], 0)
    # # original_res2[0][1] = original_res2[0][2]
    # # original_res2[0][0] = original_res2[0][0][5:7]
    # dataset.evaluate(res1, waymo_metrics=True)
    # dataset.evaluate(res2, waymo_metrics=True)
    # dataset.evaluate(original_res, waymo_metrics=True)
    # dataset.evaluate(original_res2, waymo_metrics=True)


    mean_ap, eval_results, df_summary, recalls, precisions = eval_map(original_res, anns, nproc=4, model_name="Original Retina")
    mean_ap, eval_results, df_summary, recalls, precisions = eval_map(res1, anns, nproc=4, model_name="Ensemble 1")
    #
    mean_ap, eval_results, df_summary, recalls, precisions = eval_map(original_res2, anns, nproc=4, model_name="Original FRCNN")
    mean_ap, eval_results, df_summary, recalls, precisions = eval_map(res2, anns, nproc=4, model_name="Ensemble 2")
    #
    combined_dets_nms = [[dets[n] for dets in [original_res, original_res2]] for n in range(len(original_res))]
    cfg = {'type': 'nms', 'iou_threshold': 0.5}
    res_combined_nms = [ensembleDetections(dets, cfg) for dets in combined_dets_nms]
    mean_ap, eval_results, df_summary, recalls, precisions = eval_map(res_combined_nms, anns, nproc=4, model_name="Original NMS")

    combined_dets_wbf = [[dets[n] for dets in [original_res, original_res2]] for n in range(len(original_res))]
    cfg = {'type': 'wbf', 'iou_threshold': 0.5}
    res_combined_wbf = [ensembleDetections(dets, cfg) for dets in combined_dets_wbf]
    mean_ap, eval_results, df_summary, recalls, precisions  = eval_map(res_combined_wbf, anns, nproc=4, model_name="Original WBF")

    # score_th_results(anns, [res1, res2, original_res, original_res2, res_combined_nms, res_combined_wbf], ["Fuse Retina", "Fuse FRCNN", "RetinaNet", "FRCNN",  "NMS", "WBF"])