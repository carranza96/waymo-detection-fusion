from __future__ import print_function, division
import os
import torch
import numpy as np
import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn.functional as F
from PIL import Image
from CoordConv import AddCoordsNp

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
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert np.max(depth_png) > 255, \
        "np.max(depth_png)={}, path={}".format(np.max(depth_png), filename)

    depth = depth_png.astype(np.float) / 256.
    # depth[depth_png == 0] = -1.
    depth = np.expand_dims(depth, -1)
    return depth

def transform(rgb, sparse, target, transform_list):
    transform = transform_list
    if rgb is not None:
        rgb = transform(rgb)
    if sparse is not None:
        sparse = transform(sparse)
    if target is not None:
        target = transform(target)

    return rgb, sparse, target

to_tensor = transforms.ToTensor()
to_float_tensor = lambda x: to_tensor(x).float()

def load_calib():
    """
    Temporarily hardcoding the calibration matrix using calib file from 2011_09_26
    """
    calib = open("ldcnet/dataloaders/calib_cam_to_cam.txt", "r")
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

class KittiDataset(Dataset):
 
    def __init__(self, h, w, root_dir_depth, root_dir_raw, split, transform):
        self.root_dir_depth = root_dir_depth
        self.transform = transform
        self.h = h
        self.w = w
        
        if(split=="train"):
            glob_d = os.path.join(
                root_dir_depth,
                'data_depth_velodyne/train/*_sync/proj_depth/velodyne_raw/image_0[2,3]/*.png')
            glob_gt = os.path.join(
                root_dir_depth,
                'data_depth_annotated/train/*_sync/proj_depth/groundtruth/image_0[2,3]/*.png'
            )

            def get_rgb_paths(p):
                ps = p.split('/')
                date_liststr = []
                date_liststr.append(ps[-5][:10])
                pnew = root_dir_raw + "/" + '/'.join(date_liststr + ps[-5:-4] + ps[-2:-1] + ['data'] + ps[-1:])
                return pnew
        elif(split=="val"):
            glob_d = os.path.join(
                root_dir_depth,
                'data_depth_velodyne/val/*_sync/proj_depth/velodyne_raw/image_0[2,3]/*.png')
            glob_gt = os.path.join(
                root_dir_depth,
                'data_depth_annotated/val/*_sync/proj_depth/groundtruth/image_0[2,3]/*.png'
            )

            def get_rgb_paths(p):
                ps = p.split('/')
                date_liststr = []
                date_liststr.append(ps[-5][:10])
                pnew = root_dir_raw + "/" + '/'.join(date_liststr + ps[-5:-4] + ps[-2:-1] + ['data'] + ps[-1:])
                return pnew


        paths_d = sorted(glob.glob(glob_d))
        paths_gt = sorted(glob.glob(glob_gt))
        paths_rgb = [get_rgb_paths(p) for p in paths_gt]
        self.paths = {"rgb": paths_rgb, "d": paths_d, "gt": paths_gt}
    
    def __len__(self):
        return len(self.paths['gt'])

    def __getraw__(self, index):
        rgb= rgb_read(self.paths['rgb'][index])
        sparse= depth_read(self.paths['d'][index])
        target= depth_read(self.paths['gt'][index])

        return rgb, sparse, target
    
    def __getitem__(self, index):
        # This version normalizes RGB and sparse depth
        rgb, sparse, target = self.__getraw__(index)

        rgb, sparse, target = transform(rgb.astype(int), sparse, target, self.transform)  # Estandar

        rgb = F.interpolate(rgb.unsqueeze(0), size=(self.h,self.w), mode='nearest').squeeze(0)
        sparse = F.interpolate(sparse.unsqueeze(0), size=(self.h,self.w), mode='nearest').squeeze(0)
        target = F.interpolate(target.unsqueeze(0), size=(self.h,self.w), mode='nearest').squeeze(0)

        K = load_calib()
        position = AddCoordsNp(self.h, self.w)
        position = position.call()

        to_tensor = transforms.ToTensor()
        to_float_tensor = lambda x: to_tensor(x).float()
        
        data = {"rgb": rgb, "d": sparse/80., "gt": target, "position": to_float_tensor(position), "K": K}
    
        return data