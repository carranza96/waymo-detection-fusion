from torchvision.transforms.functional import to_tensor
import data
from model import ENet, LDCNet
import os
import torch
from skimage import io, transform
import numpy as np
import glob
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch import optim
from torchvision import transforms, utils
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from CoordConv import AddCoordsNp
import cv2

def load_calib():
    """
    Temporarily hardcoding the calibration matrix using calib file from 2011_09_26
    """
    calib = open("/home/javgal/mmdetection_clean/mmdetection/ldcnet/dataloaders/calib_cam_to_cam.txt", "r")
    lines = calib.readlines()
    P_rect_line = lines[25]

    Proj_str = P_rect_line.split(":")[1].split(" ")[1:]
    Proj = np.reshape(np.array([float(p) for p in Proj_str]),
                    (3, 4)).astype(np.float32)
    K = Proj[:3, :3]  # camera matrix

    # note: we will take the center crop of the images during augmentation
    # that changes the optical centers, but not focal lengths
    # K[0, 2] = K[0, 2] - 13  # from width = 1242 to 1216, with a 13-pixel cut on both sides
    # K[1, 2] = K[1, 2] - 11.5  # from width = 375 to 352, with a 11.5-pixel cut on both sides
    K[0, 2] = K[0, 2] - 13;
    K[1, 2] = K[1, 2] - 11.5;
    return K

def rgb_read(filename):
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    # rgb_png = np.array(img_file, dtype=float) / 255.0 # scale pixels to the range [0,1]
    rgb_png = np.array(img_file, dtype='uint8')  # in the range [0,255]
    img_file.close()
    return rgb_png

def depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    depth_png = np.array(img_file, dtype=int)
    img_file.close()

    depth = depth_png.astype(np.float) / 256.
    # depth = depth_png.astype(np.float)
    # depth[depth_png == 0] = -1.
    depth = np.expand_dims(depth, -1)
    return depth

cmap = plt.cm.jet
cmap2 = plt.cm.nipy_spectral

vmin=0
vmax=95
cmap= "viridis"

def depth_colorize(depth):
    cmap = plt.cm.jet
    depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    depth = 255 * cmap(depth)[:, :, :3]  # H, W, C
    return depth.astype('uint8')

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach1280()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = (diff**2).mean()
        return self.loss

def main():
    h, w = 352, 1216
    # Esto se hace porque el tamaño en camaras 4 y 5 no es valido con la enet, y hay que reescalar antes de meter al modelo
    h2, w2 = 1280, 1920

    to_tensor = transforms.ToTensor()
    to_float_tensor = lambda x: to_tensor(x).float()
    transform = transforms.Compose([to_float_tensor])

    rgb_directory = "....."
    depth_directory = "....."

    rgb_id = os.path.expanduser(
        '/home/manuel/Escritorio/mmdetection/data/waymococo_f0/val2020/*_camera[1,2,3].jpg')

    rgb_names = sorted(glob.glob(rgb_id))

    d_id = os.path.expanduser(
        '/home/manuel/Escritorio/mmdetection/data/waymococo_f0/val2020/*_camera[1,2,3]_lidar_u16.png')

    d_names = sorted(glob.glob(d_id))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_path = "/home/javgal/kitti_depth_clean/results/ENet_Simple_1216x352/ENet_Simple_Best.pth"

    model = None
    model_type = "LDCNet"

    if(model_type == "LDCNet"):
        model = nn.DataParallel(LDCNet(h, w), device_ids=[0,1]).to(device)
    elif(model_type == "ENet"):
        model = nn.DataParallel(ENet(h, w), device_ids=[0,1]).to(device)

    model.load_state_dict(torch.load(model_path))

    model.eval().to(device)

    if not os.path.exists("ldcnet/results/images_depth"):
        os.makedirs("ldcnet/results/images_depth")
    if not os.path.exists("ldcnet/results/images_colored"):
        os.makedirs("ldcnet/results/images_colored")

    for i in range(len(rgb_names)):
        rgb = transform(rgb_read(rgb_names[i]).astype(int))
        d = transform(depth_read(d_names[i]))

        rgb = F.interpolate(rgb.unsqueeze(0), size=(h,w), mode='nearest').squeeze(0)
        d = F.interpolate(d.unsqueeze(0), size=(h,w), mode='nearest').squeeze(0) / 80.

        features = np.zeros((d.shape[0], 4, h, w))

        features[:,0:3,:,:] = rgb
        features[:,3,:,:] = np.reshape(d,(1,h,w))

        K = load_calib()

        position = AddCoordsNp(h, w)
        position = position.call()

        args = {"position": to_float_tensor(position).view(-1, 2, h, w).to(device), "K": torch.tensor(K).view(-1, 3, 3).to(device)}

        with torch.no_grad():
            batch_features = torch.tensor(features).view(-1, 4, h, w).to(device)

            batch_features = batch_features.float()
            out = model(batch_features, args)

            print(i+1, " / ", len(rgb_names))

            out_image = out.cpu().detach().numpy()[0,0,:,:]

            image_to_write = (cv2.resize(out_image,dsize=(w2,h2), interpolation=cv2.INTER_AREA)*256).astype(np.uint16)
            res_name = (rgb_names[i].split(".")[0]).split("/")[-1] + "_LDCNet_u16.png"

            cv2.imwrite("ldcnet/results/images_depth/" + res_name, image_to_write)

            frame1 = plt.gca()
            frame1.axes.xaxis.set_ticklabels([])
            frame1.axes.yaxis.set_ticklabels([])
            plt.imshow(depth_colorize(image_to_write/ 256.), vmin=vmin, vmax=vmax, cmap=cmap)
            plt.savefig("ldcnet/results/images_colored/" + res_name, bbox_inches='tight')

            plt.show()
            plt.close()

            i = i + 1


if __name__ == '__main__':
    main()