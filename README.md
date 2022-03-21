# Multi-modal object detection for autonomous driving using transfer learning for LiDAR depth completion

## Contents
1. [Dependency](#dependency)
0. [Kitti Dataset Setup](#kitti-dataset-setup)
0. [LDCNet](#ldcnet)
0. [MMDetection](#mmdetection)
0. [Citation](#citation)

## Dependency

## Kitti Dataset Setup

## LDCNet
### Method

Our proposed LiDAR Depth Completion network (LDCNet). The network outputs a dense depth map combining a camera image and the sparse LiDAR projections. In the encoder, 3D position maps are concatenated to the feature maps to encode geometric information. The decoder upsamples the feature maps using deconvolution. The numbers below the maps indicate the number of channels. K and S indicate kernel size and stride in the convolution, respectively.

<div align=center><img src="https://github.com/carranza96/mmdetection/blob/javi/images/LDCNet-1.png" width = "100%" height = "100%" /></div>

## MMDetection

## Citation