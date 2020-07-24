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

psp_models = {
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}

class ModifiedResnet(nn.Module):

    def __init__(self, usegpu=True):
        super(ModifiedResnet, self).__init__()

        self.model = psp_models['resnet18'.lower()]()
        self.model = nn.DataParallel(self.model)

    def forward(self, x):
        x = self.model(x)
        return x

class PoseNetFeat(nn.Module):
    def __init__(self, num_points):
        super(PoseNetFeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)

        self.e_conv1 = torch.nn.Conv1d(32, 64, 1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)

        self.conv5 = torch.nn.Conv1d(256, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, 1024, 1)

        self.ap1 = torch.nn.AvgPool1d(num_points)
        self.num_points = num_points

    def forward(self, x, emb):
        x = F.relu(self.conv1(x))
        emb = F.relu(self.e_conv1(emb))
        pointfeat_1 = torch.cat((x, emb), dim=1)

        x = F.relu(self.conv2(x))
        emb = F.relu(self.e_conv2(emb))
        pointfeat_2 = torch.cat((x, emb), dim=1)

        x = F.relu(self.conv5(pointfeat_2))
        x = F.relu(self.conv6(x))

        ap_x = self.ap1(x)

        ap_x = ap_x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
        return torch.cat([pointfeat_1, pointfeat_2, ap_x], 1) #128 + 256 + 1024

class PoseNet(nn.Module):
    def __init__(self, num_points, num_obj):
        super(PoseNet, self).__init__()
        self.num_points = num_points
        self.cnn = ModifiedResnet()
        self.feat = PoseNetFeat(num_points)

        self.conv1_r = torch.nn.Conv1d(1408, 640, 1)
        self.conv1_t = torch.nn.Conv1d(1408, 640, 1)
        self.conv1_c = torch.nn.Conv1d(1408, 640, 1)

        self.conv2_r = torch.nn.Conv1d(640, 256, 1)
        self.conv2_t = torch.nn.Conv1d(640, 256, 1)
        self.conv2_c = torch.nn.Conv1d(640, 256, 1)

        self.conv3_r = torch.nn.Conv1d(256, 128, 1)
        self.conv3_t = torch.nn.Conv1d(256, 128, 1)
        self.conv3_c = torch.nn.Conv1d(256, 128, 1)

        self.conv4_r = torch.nn.Conv1d(128, num_obj*4, 1) #quaternion
        self.conv4_t = torch.nn.Conv1d(128, num_obj*3, 1) #translation
        self.conv4_c = torch.nn.Conv1d(128, num_obj*1, 1) #confidence

        self.num_obj = num_obj

    def forward(self, img, x, choose, obj):
        out_img = self.cnn(img)

        bs, di, _, _ = out_img.size()

        emb = out_img.view(bs, di, -1)
        choose = choose.repeat(1, di, 1)
        emb = torch.gather(emb, 2, choose).contiguous()

        x = x.transpose(2, 1).contiguous()
        ap_x = self.feat(x, emb)

        rx = F.relu(self.conv1_r(ap_x))
        tx = F.relu(self.conv1_t(ap_x))
        cx = F.relu(self.conv1_c(ap_x))

        rx = F.relu(self.conv2_r(rx))
        tx = F.relu(self.conv2_t(tx))
        cx = F.relu(self.conv2_c(cx))

        rx = F.relu(self.conv3_r(rx))
        tx = F.relu(self.conv3_t(tx))
        cx = F.relu(self.conv3_c(cx))

        rx = self.conv4_r(rx).view(bs, self.num_obj, 4, self.num_points)
        tx = self.conv4_t(tx).view(bs, self.num_obj, 3, self.num_points)
        cx = torch.sigmoid(self.conv4_c(cx)).view(bs, self.num_obj, 1, self.num_points)

        b = 0
        out_rx = torch.index_select(rx[b], 0, obj[b])
        out_tx = torch.index_select(tx[b], 0, obj[b])
        out_cx = torch.index_select(cx[b], 0, obj[b])

        out_rx = out_rx.contiguous().transpose(2, 1).contiguous()
        out_cx = out_cx.contiguous().transpose(2, 1).contiguous()
        out_tx = out_tx.contiguous().transpose(2, 1).contiguous()

        return out_rx, out_tx, out_cx, emb.detach()

def pointwise_conv(in_features, maps, out_features):
    layers = []
    previous = in_features
    for feature_map in maps:
        layers.append(nn.BatchNorm2d(previous))
        layers.append(nn.ReLU(True))
        layers.append(nn.Conv2d(previous, feature_map, kernel_size=1, padding=0, bias=False))
        previous = feature_map
    layers.append(nn.BatchNorm2d(previous))
    layers.append(nn.ReLU(True))
    layers.append(nn.Conv2d(previous, out_features, kernel_size=1, padding=0, bias=True))
    return nn.Sequential(*layers)

class Conv(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3, padding=1, **kwargs):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(in_features)
        self.relu = nn.ReLU(True)
        self.conv = nn.Conv2d(in_features, out_features, kernel_size, padding=padding, bias=False, **kwargs)
        self.dropout = nn.Dropout2d(0.2)

    def forward(self, x):
        x = self.batch_norm(x)
        x = self.relu(x)
        return self.dropout(self.conv(x))

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
        self.add_module('batch_norm', nn.BatchNorm2d(in_features))
        self.add_module('relu', nn.ReLU(True))
        self.add_module('conv', nn.Conv2d(in_features, in_features, kernel_size=1, stride=1, padding=0, bias=False))
        self.add_module('dropout', nn.Dropout2d(0.2))
        self.add_module('pool', nn.MaxPool2d(2, stride=2, padding=0))

class Upsample(nn.Sequential):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.add_module('conv', nn.ConvTranspose2d(in_features, out_features, kernel_size=4, stride=2, padding=1, bias=False))

class KeypointNet(nn.Module):
    def __init__(self, growth_rate=16, num_keypoints=8, num_classes=22):
        super().__init__()
        self.features = nn.Conv2d(6, 64, kernel_size=7, padding=3, stride=2)
        features = 64
        self.conv1 = DenseBlock(features, layers=4, k=growth_rate)
        features2 = features + 4 * growth_rate
        self.downsample1 = Downsample(features2)
        self.conv2 = DenseBlock(features2, layers=5, k=growth_rate)
        features3 = features2 + 5 * growth_rate
        self.downsample2 = Downsample(features3)
        self.conv3 = DenseBlock(features3, layers=7, k=growth_rate)
        features4 = features3 + 7 * growth_rate
        self.downsample3 = Downsample(features4)
        features5 = features4 + 5 * growth_rate
        self.conv4 = DenseBlock(features4, layers=5, k=growth_rate)

        self.upsample1 = Upsample(features5, 128)
        features = 128 + features4
        self.conv5 = DenseBlock(features, layers=7, k=growth_rate)
        features += 7 * growth_rate
        self.upsample2 = Upsample(features, 128)
        features = 128 + features3
        self.conv6 = DenseBlock(features, layers=5, k=growth_rate)
        self.upsample3 = Upsample(features + 5 * growth_rate, 64)
        self.conv7 = DenseBlock(128, layers=4, k=growth_rate)
        out_features = 128 + 4 * growth_rate + 6

        self.keypoints_out = num_keypoints * 3
        self.num_classes = num_classes
        self.keypoint_head = pointwise_conv(out_features, [128, 64], self.keypoints_out)
        self.center_head = pointwise_conv(out_features, [128, 64], 3)
        self.segmentation_head = pointwise_conv(out_features, [128, 64], num_classes)

    def forward(self, img, points):
        N, C, H, W = img.shape
        inputs = torch.cat([img, points], dim=1)
        features = self.features(inputs) # 240 x 320
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

        inputs_small = F.interpolate(inputs, size=[240, 320], mode='bilinear')

        x = torch.cat([x, inputs_small], dim=1)

        keypoints = self.keypoint_head(x)
        centers = self.center_head(x)
        segmentation = self.segmentation_head(x)

        keypoints = F.interpolate(keypoints, size=[480, 640], mode='bilinear')
        centers = F.interpolate(centers, size=[480, 640], mode='bilinear')
        segmentation = F.interpolate(segmentation, size=[480, 640], mode='bilinear')

        return keypoints, centers, segmentation

