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

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = (diff**2).mean()
        return self.loss

def adjust_learning_rate(lr_init, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 5 epochs"""
    #lr = lr_init * (0.5**(epoch // 5))
    #'''
    lr = lr_init
    if (epoch >= 10):
        lr = lr_init * 0.5
    if (epoch >= 15):
        lr = lr_init * 0.1
    if (epoch >= 25):
        lr = lr_init * 0.01
    #'''

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

# Print iterations progress
def printProgress(iteration, total):
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filledLength = int(100 * iteration // total)
    bar = '█' * filledLength + '-' * (100 - filledLength)
    print(f'\r{""} |{bar}| {percent}% {""}', end = '\r')


def main():
    h, w = 352, 1216
    model_type = "LDCNet"
    
    temp_start = time.gmtime()
    temp = "(" + str(temp_start[2]) + "," + str(temp_start[1]) + "," + str(temp_start[0]) + "), " + str(temp_start[3]) + ":" + str(temp_start[4]) + ":" + str(temp_start[5])
    
    to_tensor = transforms.ToTensor()
    to_float_tensor = lambda x: to_tensor(x).float()
    transform = transforms.Compose([to_float_tensor])
    train_dataset = data.KittiDataset(h, w, "/home/javgal/kitti_depth_clean/kitti_depth", '/home/javgal/kitti_depth_clean/kitti_raw', "train",transform)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4, pin_memory=True)

    val_dataset = data.KittiDataset(h, w, "/home/javgal/kitti_depth_clean/kitti_depth", '/home/javgal/kitti_depth_clean/kitti_raw', "val",transform)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    absolute_loss = 9999999999999
    a = len(train_loader)
    epochs = 5

    model = None

    if(model_type == "LDCNet"):
        model = nn.DataParallel(LDCNet(h, w), device_ids=[0,1]).to(device)
    elif(model_type == "ENet"):
        model = nn.DataParallel(ENet(h, w), device_ids=[0,1]).to(device)
    

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6, betas=(0.9, 0.99))

    depth_criterion = MaskedMSELoss()

    if not os.path.exists("ldcnet/results"):
        os.makedirs("ldcnet/results")
    os.mkdir("ldcnet/results/" + temp)

    with open("ldcnet/results/" + temp + "/" + temp + ".txt", 'w') as file:
        file.write('Train ' + temp + "\n")
        file.write("Model: " + str(model_type) +  "\nEpochs: " +  str(epochs) +  "\n\n")

    for epoch in range(epochs):
        i = 0
        t_loss = 0
        v_loss = 0
        relative_time = 0
        absolute_time = 0
        
        lr = adjust_learning_rate(1e-3, optimizer, epoch)

        with open("ldcnet/results/" + temp + "/" + temp + ".txt", 'a') as file:
            for batch_features in train_loader:
                start = time.time()
                rgb = batch_features["rgb"]
                d = batch_features["d"]
                gt = batch_features["gt"]
                gt = gt.view(-1, 1, h, w).to(device)

                features = np.zeros((d.shape[0], 4, h, w))

                features[:,0:3,:,:] = rgb
                features[:,3,:,:] = np.reshape(d, (rgb.shape[0], h, w))

                args = {"position": batch_features["position"].clone().detach().view(-1, 2, h, w).to(device), "K":  batch_features["K"].clone().detach().view(-1, 3, 3).to(device)}

                batch_features = torch.tensor(features).view(-1, 4, h, w).to(device)
                optimizer.zero_grad()

                batch_features = batch_features.float()

                if(model_type == "LDCNet"):
                    out = model(batch_features, args)

                    depth_loss = depth_criterion(out, gt)
                    st1_loss = 0
                    st2_loss = 0
                elif(model_type == "ENet"):
                    st1_pred, st2_pred, out = model(batch_features, args)

                    depth_loss = depth_criterion(out, gt)
                    st1_loss = depth_criterion(st1_pred, gt)
                    st2_loss = depth_criterion(st2_pred, gt)

                # RMSE
                w_st1, w_st2 = 0, 0

                if(epoch <= 1):
                    w_st1, w_st2 = 0.2, 0.2
                elif(epoch <= 3):
                    w_st1, w_st2 = 0.05, 0.05
                else:
                    w_st1, w_st2 = 0, 0
                
                train_loss = (1 - w_st1 - w_st2) * depth_loss + w_st1 * st1_loss + w_st2 * st2_loss

                train_loss.backward()
                optimizer.step()

                relative_time = time.time() - start
                absolute_time = absolute_time + relative_time

                finish = (epochs - epoch - 1) * a * (absolute_time / (i+1)) + (absolute_time / (i+1)) * a - absolute_time

                printProgress(i, a)

                if(i%50 == 0):
                    text_to_write = "Epoch: [" +  str(epoch + 1) +  "] [" +  str(i) +  " / " +  str(a) +  "]  eta: " +  str(datetime.timedelta(seconds=int(finish))) + ", Loss: " + str(float(train_loss)) 
                    print(115 * " ", end="\r")
                    print(text_to_write + "\n", end = '\r')
                    file.write(text_to_write + "\n")

                t_loss = t_loss + float(train_loss)
                i = i + 1
            
            t_loss = t_loss / a

            absolute_time_val = 0

            for batch_features in val_loader:
                with torch.no_grad():
                    rgb = batch_features["rgb"]
                    d = batch_features["d"]
                    gt = batch_features['gt']
                    gt = gt.view(-1, 1, h, w).to(device)

                    features = np.zeros((d.shape[0], 4, h, w))

                    features[:,0:3,:,:] = rgb
                    features[:,3,:,:] = np.reshape(d,(rgb.shape[0], h, w))

                    args = {"position": batch_features["position"].clone().detach().view(-1, 2, h, w).to(device), "K":  batch_features["K"].clone().detach().view(-1, 3, 3).to(device)}

                    batch_features = torch.tensor(features).view(-1, 4, h, w).to(device)

                    batch_features = batch_features.float()
                    start = time.time()

                    if(model_type == "LDCNet"):
                        out = model(batch_features, args)
                    elif(model_type == "ENet"):
                        _ , _ , out = model(batch_features, args)

                    relative_time = time.time() - start
                    absolute_time_val = absolute_time_val + relative_time
                    
                    # RMSE

                    valid_mask = gt > 0.1

                    # convert from meters to mm
                    output_mm = 1e3 * out[valid_mask]
                    target_mm = 1e3 * gt[valid_mask]

                    abs_diff = (output_mm - target_mm).abs()

                    mse = float((torch.pow(abs_diff, 2)).mean())
                    val_loss = math.sqrt(mse)

                    v_loss = v_loss + float(val_loss)
            
            model.train()
            v_loss = v_loss / len(val_loader)
            ex_time = absolute_time_val / len(val_loader)

            if (v_loss < absolute_loss):
                torch.save(model.state_dict(),"ldcnet/results/{}/LDCNet_Best.pth".format(temp))
                absolute_loss = v_loss

            
            print(115 * " ", end="\r")
            print("epoch : {}/{}, validation loss = {:.6f} , training loss = {:.6f}, Execution time = {:.6f}".format(epoch + 1, epochs, v_loss, t_loss, ex_time))
            to_text = "epoch : {}/{}, validation loss = {:.6f} , training loss = {:.6f} \nExecution time = {:.6f}\n\n".format(epoch + 1, epochs, v_loss, t_loss, ex_time)
            file.write(to_text)

            torch.save(model.state_dict(),"ldcnet/results/{}/LDCNet_epoch_{}.pth".format(temp, epoch))

    print("Best result: ", absolute_loss)


if __name__ == '__main__':
    main()