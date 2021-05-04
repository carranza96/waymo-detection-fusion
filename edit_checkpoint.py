import torch
from copy import copy
oldck = torch.load("faster_rcnn_r50_caffe_c4_2x-71c67f27.pth")
newversion = torch.load("faster_rcnn_r50_caffe_c4.pth")

mod_state_dict = dict()
# oldck['state_dict']['backbone.conv1.weight']

for k, v in oldck['state_dict'].items():

    if 'shared_head' in k:
        k = 'roi_head.' + k

    if 'bbox_head.fc' in k:
        k = 'roi_head.' + k

    mod_state_dict[k] = v

oldck['state_dict'] = mod_state_dict

torch.save(oldck, "faster_rcnn_r50_caffe_c4_2x-71c67f27_mod.pth")

