from ldcnet.model import ENet, LDCNet
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from ldcnet.CoordConv import AddCoordsNp

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

    # depth = depth_png.astype(np.float) / 256.
    depth = depth_png.astype(np.float)
    # depth[depth_png == 0] = -1.
    depth = np.expand_dims(depth, -1)
    return depth

def output(rgb_img_path, depth_img_path):
    with torch.no_grad():
        h, w = 352, 1216
        h2, w2 = 1280, 1920

        to_tensor = transforms.ToTensor()
        to_float_tensor = lambda x: to_tensor(x).float()
        transform = transforms.Compose([to_float_tensor])

        rgb_image = rgb_img_path
        depth_image = depth_img_path

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model_path = "/home/javgal/mmdetection_clean/mmdetection/ldcnet/results/ldcnet_testing/LDCNet_Best.pth"

        model = None
        model_type = "LDCNet"

        if(model_type == "LDCNet"):
            model = nn.DataParallel(LDCNet(h, w), device_ids=[0,1]).to(device)
        elif(model_type == "ENet"):
            model = nn.DataParallel(ENet(h, w), device_ids=[0,1]).to(device)

        model.load_state_dict(torch.load(model_path))

        model.eval().to(device)

        rgb = transform(rgb_read(rgb_image).astype(int))
        d = transform(depth_read(depth_image))

        rgb = F.interpolate(rgb.unsqueeze(0), size=(h2,w2), mode='nearest').squeeze(0)
        d = F.interpolate(d.unsqueeze(0), size=(h2,w2), mode='nearest').squeeze(0) / 80.

        features = np.zeros((d.shape[0], 4, h2, w2))

        features[:,0:3,:,:] = rgb
        features[:,3,:,:] = np.reshape(d,(1,h2,w2))

        K = load_calib()

        position = AddCoordsNp(h2, w2)
        position = position.call()

        args = {"position": to_float_tensor(position).view(-1, 2, h2, w2).to(device), "K": torch.tensor(K).view(-1, 3, 3).to(device)}

        batch_features = torch.tensor(features).view(-1, 4, h2, w2).to(device)

        batch_features = batch_features.float()
        out = model(batch_features, args)

        out_image = out.cpu().detach().numpy()[0,0,:,:]

        return out_image