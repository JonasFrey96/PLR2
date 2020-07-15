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
    return nn.Sequential(
        nn.Conv2d(in_features, maps, kernel_size=1, padding=0, bias=False),
        nn.BatchNorm2d(maps),
        nn.ReLU(),
        nn.Conv2d(maps, out_features, kernel_size=1, padding=0, bias=True)
        )

class Conv(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3, padding=1, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size, padding=padding, bias=False, **kwargs)
        self.batch_norm = nn.BatchNorm2d(out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        return self.relu(x)

class BasicBlock(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        features = in_features // 4
        self.conv1x1 = Conv(in_features, features, kernel_size=1, stride=1, padding=0)
        self.conv1 = Conv(features, features, kernel_size=3, dilation=1, padding=1)
        self.conv2 = Conv(features, features, kernel_size=3, dilation=2, padding=2)
        self.conv3 = Conv(features, features, kernel_size=3, dilation=3, padding=3)
        self.conv4 = Conv(features, features, kernel_size=3, dilation=4, padding=4)
        self.out = Conv(in_features, in_features, kernel_size=1, stride=1, padding=0)

    def forward(self, inputs):
        x = self.conv1x1(inputs)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x12 = x1 + x2
        x123 = x1 + x2 + x3
        x1234 = x1 + x2 + x3 + x4

        concat = torch.cat([x1, x12, x123, x1234], dim=1)
        return inputs + self.out(concat)

class Downsample(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.conv = Conv(in_features, out_features, kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        features = in_features // 4
        self.conv = nn.Sequential(
                Conv(in_features, features, kernel_size=1, stride=1, padding=0),
                nn.ConvTranspose2d(features, out_features, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.BatchNorm2d(out_features))

    def forward(self, x):
        return self.conv(x)

class KeypointNet(nn.Module):
    def __init__(self, num_points, num_obj, num_keypoints=8, num_classes=22):
        super().__init__()
        self.features = Conv(6, 64, kernel_size=7, padding=3, stride=2)
        self.conv1 = BasicBlock(64)
        self.downsample1 = Downsample(64, 64)
        self.conv2 = BasicBlock(64)
        self.downsample2 = Downsample(64, 128)
        self.conv3 = BasicBlock(128)
        self.downsample3 = Downsample(128, 256)
        self.conv4 = nn.Sequential(BasicBlock(256), BasicBlock(256))

        self.upsample1 = Upsample(512, 128)
        self.conv5 = BasicBlock(128)
        self.upsample2 = Upsample(256, 64)
        self.conv6 = BasicBlock(64)
        self.upsample3 = Upsample(128, 64)
        self.conv7 = BasicBlock(64)

        self.keypoints_out = num_keypoints * 3
        self.num_classes = num_classes
        self.keypoint_head = pointwise_conv(70, 32, self.keypoints_out)
        self.center_head = pointwise_conv(70, 32, 3)
        self.segmentation_head = pointwise_conv(70, 64, num_classes)

    def forward(self, img, points):
        N, C, H, W = img.shape
        inputs = torch.cat([img, points], dim=1)
        features = self.features(inputs) # 240 x 320
        x = self.conv1(features)
        x1 = self.downsample1(x) # 120 x 160
        x = self.conv2(x1)
        x2 = self.downsample2(x) # 60 x 80
        x = self.conv3(x2)
        x3 = self.downsample3(x) # 30 x 40
        x = self.conv4(x3)

        x = self.upsample1(torch.cat([x, x3], dim=1)) # 60 x 80
        x = self.conv5(x)
        x = self.upsample2(torch.cat([x, x2], dim=1)) # 120 x 160
        x = self.conv6(x)
        x = self.upsample3(torch.cat([x, x1], dim=1)) # 240 x 320
        x = self.conv7(x)

        inputs_small = F.interpolate(inputs, size=[240, 320], mode='bilinear')

        x = torch.cat([x, inputs_small], dim=1)

        keypoints = self.keypoint_head(x)
        centers = self.center_head(x)
        segmentation = self.segmentation_head(x)

        keypoints = F.interpolate(keypoints, size=[480, 640], mode='bilinear')
        centers = F.interpolate(centers, size=[480, 640], mode='bilinear')
        segmentation = F.interpolate(segmentation, size=[480, 640], mode='bilinear')

        return keypoints, centers, segmentation


class PoseRefineNetFeat(nn.Module):
    def __init__(self, num_points):
        super(PoseRefineNetFeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)

        self.e_conv1 = torch.nn.Conv1d(32, 64, 1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)

        self.conv5 = torch.nn.Conv1d(384, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, 1024, 1)

        self.ap1 = torch.nn.AvgPool1d(num_points)
        self.num_points = num_points

    def forward(self, x, emb):
        x = F.relu(self.conv1(x))
        emb = F.relu(self.e_conv1(emb))
        pointfeat_1 = torch.cat([x, emb], dim=1)

        x = F.relu(self.conv2(x))
        emb = F.relu(self.e_conv2(emb))
        pointfeat_2 = torch.cat([x, emb], dim=1)

        pointfeat_3 = torch.cat([pointfeat_1, pointfeat_2], dim=1)

        x = F.relu(self.conv5(pointfeat_3))
        x = F.relu(self.conv6(x))

        ap_x = self.ap1(x)

        ap_x = ap_x.view(-1, 1024)
        return ap_x

class PoseRefineNet(nn.Module):
    def __init__(self, num_points, num_obj):
        super(PoseRefineNet, self).__init__()
        self.num_points = num_points
        self.feat = PoseRefineNetFeat(num_points)

        self.conv1_r = torch.nn.Linear(1024, 512)
        self.conv1_t = torch.nn.Linear(1024, 512)

        self.conv2_r = torch.nn.Linear(512, 128)
        self.conv2_t = torch.nn.Linear(512, 128)

        self.conv3_r = torch.nn.Linear(128, num_obj*4) #quaternion
        self.conv3_t = torch.nn.Linear(128, num_obj*3) #translation

        self.num_obj = num_obj

    def forward(self, x, emb, obj):
        bs = x.size()[0]

        x = x.transpose(2, 1).contiguous()
        ap_x = self.feat(x, emb)

        rx = F.relu(self.conv1_r(ap_x))
        tx = F.relu(self.conv1_t(ap_x))

        rx = F.relu(self.conv2_r(rx))
        tx = F.relu(self.conv2_t(tx))

        rx = self.conv3_r(rx).view(bs, self.num_obj, 4)
        tx = self.conv3_t(tx).view(bs, self.num_obj, 3)

        b = 0
        out_rx = torch.index_select(rx[b], 0, obj[b])
        out_tx = torch.index_select(tx[b], 0, obj[b])

        return out_rx, out_tx
