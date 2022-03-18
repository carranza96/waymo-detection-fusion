from torchvision.transforms.functional import to_tensor
import data
from model import ENet, LDCNet
import os
import torch
import numpy as np
import glob
from torch.utils.data import Dataset, DataLoader
from torch import optim
from torchvision import transforms, utils
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import datetime

# cmap = plt.cm.jet
# cmap2 = plt.cm.nipy_spectral

# def depth_colorize(depth):
#     depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
#     depth = 255 * cmap(depth)[:, :, :3]  # H, W, C
#     return depth.astype('uint8')

def main():
    h, w = 352, 1216
    model_type = "LDCNet"
    
    temp_start = time.gmtime()
    temp = "(" + str(temp_start[2]) + "," + str(temp_start[1]) + "," + str(temp_start[0]) + "), " + str(temp_start[3]) + ":" + str(temp_start[4]) + ":" + str(temp_start[5])
    
    to_tensor = transforms.ToTensor()
    to_float_tensor = lambda x: to_tensor(x).float()
    transform = transforms.Compose([to_float_tensor])

    val_dataset = data.KittiDataset(h, w, "/home/javgal/kitti_depth_clean/kitti_depth", '/home/javgal/kitti_depth_clean/kitti_raw', "val",transform)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_path = "/home/javgal/kitti_depth_clean/results/ENet_Simple_1216x352/ENet_Simple_Best.pth"

    model = None

    if(model_type == "LDCNet"):
        model = nn.DataParallel(LDCNet(h, w), device_ids=[0,1]).to(device)
    elif(model_type == "ENet"):
        model = nn.DataParallel(ENet(h, w), device_ids=[0,1]).to(device)

    model.load_state_dict(torch.load(model_path))
    
    model.eval().to(device)
    criterion = nn.MSELoss()

    i = 0
    a = len(val_loader)
    total_loss = 0
    gpu_total_time = 0
    for batch_features in val_loader:
        rgb = batch_features["rgb"]
        d = batch_features["d"]
        gt = torch.tensor(batch_features['gt']).float()
        gt = gt.view(-1, 1, h, w).to(device)

        features = np.zeros((d.shape[0], 4, h, w))

        features[:,0:3,:,:] = rgb
        features[:,3,:,:] = np.reshape(d, (rgb.shape[0], h, w))

        args = {"position": torch.tensor(batch_features["position"]).view(-1, 2, h, w).to(device), "K": torch.tensor(batch_features["K"]).view(-1, 3, 3).to(device)}

        num_images = len(val_loader)

        with torch.no_grad():
            batch_features = torch.tensor(features).view(-1, 4, h, w).to(device)

            batch_features = batch_features.float()
            start = time.time()

            if(model_type == "LDCNet"):
                out = model(batch_features, args)
            elif(model_type == "ENet"):
                _ , _ , out = model(batch_features, args)

            gpu_time = time.time() - start
            gpu_total_time = gpu_total_time + (gpu_time / a)
            
            # RMSE
            valid_mask = gt > 0.1

            outputs_mm = 1e3 * out[valid_mask]
            gt_mm = 1e3 * gt[valid_mask]
            abs_diff = (outputs_mm - gt_mm).abs()
            mse = float((torch.pow(abs_diff, 2)).mean())
            loss = math.sqrt(mse)

            if(i % 10 == 0):
                print(i, " / ", a)
                print("RMSE: ", loss)

            total_loss = total_loss + (float(loss) / a)
            i = i + 1

    print("Mean RMSE: ", total_loss)
    print("Mean Execution Time: ", gpu_total_time)


if __name__ == '__main__':
    main()