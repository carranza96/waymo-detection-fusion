import mmcv
import torch
import wandb
import numpy as np
from mmcv.parallel import MMDataParallel
from tools.train import train_detector
from mmdet.datasets import build_dataloader, build_dataset
from mean_ap import eval_map
from ensemble import ensembleDetections
from neural_ensemble import EnsembleModel
from neural_ensemble_utils import init_base_models, load_saved_preds_base_models, \
    single_gpu_test_two_outputs, postprocess_detections

log_wandb = False
train = True
load_saved_preds = False
ensemble_checkpoint = None
save_dir = 'ensemble1/'
pkl_path = save_dir + 'results.pkl'
base_models_names = ['faster_rcnn_r50_fpn_fp16_4x2_1x_1280x1920', 'retinanet_r50_fpn_fp16_4x2_1x_1280x1920']



## Load base models and ensemble model (with checkpoint if provided)
## TODO: Each base model in different GPU
base_models = init_base_models(base_models_names)
model = EnsembleModel(base_models)
model = MMDataParallel(model, device_ids=[0])
if ensemble_checkpoint:
    model.load_state_dict(torch.load(ensemble_checkpoint))
if log_wandb:
    wandb.init(project='ensemble_od', entity='minerva')
    wandb.watch(model, log='all')

## Get cfg of one base model and set test_mode to avoid filtering out images without gts
# TODO: Write here complete data dict
cfg = base_models[0].cfg
cfg.seed = None
cfg.data.test.test_mode = True


if train:
    saved_preds = load_saved_preds_base_models(base_models_names) if load_saved_preds else None

    dataset = build_dataset(cfg.data.train)
    # Filter dataset by specific image index
    # dataset = torch.utils.data.Subset(dataset, [44201])
    # dataset = torch.utils.data.Subset(dataset, [29822])
    # dataset.flag = np.array([1], dtype=np.uint8)

    train_detector(model, dataset, cfg)

    torch.save(model.state_dict(), save_dir + 'latest.pth')


else:
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
    mmcv.dump(pkl_path)


    # Postprocess predictions
    score_th = 0.05
    nms_cfg = {'type': 'nms', 'iou_threshold': 0.5}
    max_dets_class = 100

    # TODO: Do NMS exactly as in MMDetection
    res1 = postprocess_detections(res1, score_th, nms_cfg, max_dets_class)
    res2 = postprocess_detections(res2, score_th, nms_cfg, max_dets_class)



    # Evaluate
    original_res = mmcv.load("saved_models/study/faster_rcnn_r50_fpn_fp16_4x2_1x_1280x1920/results_sample.pkl")
    original_res2 = mmcv.load("saved_models/study/retinanet_r50_fpn_fp16_4x2_1x_1280x1920/results_sample.pkl")

    mean_ap, eval_results, df_summary, recalls, precisions = eval_map(original_res, anns, nproc=4, model_name="Original 1")
    mean_ap, eval_results, df_summary, recalls, precisions = eval_map(original_res2, anns, nproc=4, model_name="Original 1")

    mean_ap, eval_results, df_summary, recalls, precisions = eval_map(res1, anns, nproc=4, model_name="Ensemble 1")
    mean_ap, eval_results, df_summary, recalls, precisions = eval_map(res2, anns, nproc=4, model_name="Ensemble 2")

    combined_dets = [[dets[n] for dets in [original_res, original_res2]] for n in range(len(original_res))]
    cfg = {'type': 'wbf', 'iou_threshold': 0.5}
    print("WBF Original")
    res_combined = [ensembleDetections(dets, cfg) for dets in combined_dets]
    mean_ap, eval_results, df_summary = eval_map(res_combined, anns, nproc=4, model_name="Original WBF")