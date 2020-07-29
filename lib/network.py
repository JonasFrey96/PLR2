import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import Image
import numpy as np
import pdb
import torch.nn.functional as F
from lib.pspnet import PSPNet

def pointwise_conv(in_features, maps, out_features):
    layers = []
    previous = in_features
    for feature_map in maps:
        layers.append(nn.Conv2d(previous, feature_map, kernel_size=1, padding=0, bias=True))
        layers.append(nn.ELU(True))
        previous = feature_map
    layers.append(nn.Conv2d(previous, out_features, kernel_size=1, padding=0, bias=True))
    return nn.Sequential(*layers)

class KeypointHead(nn.Module):
    def __init__(self, n_classes, in_features, out_features):
        super().__init__()
        self.out_features = out_features
        self.n_out = n_classes + 1
        self.conv1 = nn.Conv2d(in_features, 256, 1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 128, 1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.out = nn.Conv2d(128, self.n_out * out_features, 1, stride=1, padding=0, bias=True)

    def forward(self, x, label):
        # x: N x C x H x W
        # label: N x 1 x H x W
        x = self.bn1(self.conv1(x))
        x = F.relu(x, inplace=True)
        x = self.bn2(self.conv2(x))
        x = F.relu(x, inplace=True)
        N, C, H, W = x.shape
        x = self.out(x).view(N, self.out_features, self.n_out, H, W)
        return torch.gather(x, 2, label[:, None, None, :, :].expand(-1, self.out_features, -1, -1, -1))[:, :, 0]

class Conv(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3, padding=1, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size, padding=padding, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_features)
        self.act = nn.ReLU(True)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return self.act(x)

class DenseBlock(nn.Module):
    def __init__(self, in_features, layers=4, k=8):
        super().__init__()
        self.convolutions = []
        for i in range(layers):
            conv = Conv(in_features + i * k, k)
            self.add_module(f'conv_{i}', conv)
            self.convolutions.append(conv)

    def forward(self, inputs):
        outputs = [inputs]
        for conv in self.convolutions:
            out = conv(torch.cat(outputs, dim=1))
            outputs.append(out)
        return torch.cat(outputs, dim=1)

class Downsample(nn.Sequential):
    def __init__(self, in_features):
        super().__init__()
        self.add_module('conv', nn.Conv2d(in_features, in_features, kernel_size=3, stride=2, padding=1, bias=False))
        self.add_module('bn', nn.BatchNorm2d(in_features))
        self.add_module('act', nn.ReLU(True))

class Upsample(nn.Sequential):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.add_module('conv', nn.ConvTranspose2d(in_features, out_features, kernel_size=4, stride=2, padding=1, bias=False))
        self.add_module('bn', nn.BatchNorm2d(out_features))
        self.add_module('act', nn.ReLU(True))

class KeypointNet(nn.Module):
    def __init__(self, growth_rate=16, num_keypoints=8, num_classes=22):
        super().__init__()
        self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=2, bias=True),
                nn.ELU(True))
        features1 = 70
        self.conv1 = DenseBlock(features1, layers=4, k=growth_rate)
        features2 = features1 + 4 * growth_rate
        self.downsample1 = Downsample(features2)
        self.conv2 = DenseBlock(features2, layers=5, k=growth_rate)
        features3 = features2 + 5 * growth_rate
        self.downsample2 = Downsample(features3)
        self.conv3 = DenseBlock(features3, layers=7, k=growth_rate)
        features4 = features3 + 7 * growth_rate
        self.downsample3 = Downsample(features4)
        features5 = features4 + 9 * growth_rate
        self.conv4 = DenseBlock(features4, layers=9, k=growth_rate)

        self.upsample1 = Upsample(features5, 256)
        features = 256 + features4
        self.conv5 = DenseBlock(features, layers=7, k=growth_rate)
        features += 7 * growth_rate
        self.upsample2 = Upsample(features, 128)
        features = 128 + features3
        self.conv6 = DenseBlock(features, layers=5, k=growth_rate)
        self.upsample3 = Upsample(features + 5 * growth_rate, 64)
        features = features1 + 64
        self.conv7 = DenseBlock(features, layers=4, k=growth_rate)
        out_features = features + 4 * growth_rate + 6

        self.keypoints_out = num_keypoints * 3
        self.num_classes = num_classes

        self.keypoint_head = KeypointHead(num_classes, out_features, self.keypoints_out)
        self.center_head = pointwise_conv(out_features, [128, 64], 3)
        self.segmentation_head = pointwise_conv(out_features, [128, 64], num_classes)

    def forward(self, img, points, vertmap, label=None):
        N, C, H, W = img.shape
        features = self.features(img) # 240 x 320
        features = torch.cat([features, points, vertmap], dim=1)
        x = self.conv1(features)
        x = self.downsample1(x) # 120 x 160
        x1 = self.conv2(x)
        x = self.downsample2(x1) # 60 x 80
        x2 = self.conv3(x)
        x = self.downsample3(x2) # 30 x 40
        x = self.conv4(x)

        x = self.upsample1(x) # 60 x 80
        x = self.conv5(torch.cat([x, x2], dim=1))
        x = self.upsample2(x) # 120 x 160
        x = self.conv6(torch.cat([x, x1], dim=1))
        x = self.upsample3(x) # 240 x 320
        x = self.conv7(torch.cat([x, features], dim=1))

        x = torch.cat([x, points, vertmap], dim=1)

        segmentation = self.segmentation_head(x)

        if label is None:
            label = segmentation.argmax(dim=1)

        keypoints = self.keypoint_head(x, label)
        centers = self.center_head(x)

        return keypoints, centers, segmentation

