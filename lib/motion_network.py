import torch
import torch.nn as nn
from lib.knn.__init__ import KNearestNeighbor
import torch.nn.functional as F
# first idea show that you can predict the motion with a simple cor


class NearestTranslationEmbStack(nn.Module):

    def __init__(self):
        super(NearestTranslationEmbStack, self).__init__()

    def forward(self, emb1, emb2, t1, t2):
        bs, num_feat, num_points = emb1.shape

        t1 = t1.view(bs, 3, -1)
        t2 = t2.view(bs, 3, -1)

        # t1 shape 1,3,1000  1,3,1000
        inds = KNearestNeighbor.apply(t1, t2, 1)
        t1 = torch.index_select(t1, 2, inds.view(-1).detach() - 1)
        emb1 = torch.index_select(emb1, 2, inds.view(-1).detach() - 1)
        emb = torch.stack([emb1, emb2], dim=1).view(
            bs, num_feat * 2, num_points)

        return emb


class LearnedEmbStack(nn.Module):

    def __init__(self):
        super(LearnedEmbStack, self).__init__()
        # Frame 1  - 1000 points * num_feat
        # Frame 2  - 1000 points * num_feat

        # idea use num

    def forward(self, emb1, emb2, t1, t2):
        bs, num_feat, num_points = emb1.shape

        t1 = t1.view(bs, 3, -1)
        t2 = t2.view(bs, 3, -1)

        # t1 shape 1,3,1000  1,3,1000
        inds = KNearestNeighbor.apply(t1, t2, 1)
        t1 = torch.index_select(t1, 2, inds.view(-1).detach() - 1)
        emb1 = torch.index_select(emb1, 2, inds.view(-1).detach() - 1)
        emb = torch.stack([emb1, emb2], dim=1).view(
            bs, num_feat * 2, num_points)

        return emb


class MotionNetwork(nn.Module):

    def __init__(self, num_points, num_obj, num_feat, mode):
        super(MotionNetwork, self).__init__()
        self.num_obj = num_obj
        self.num_points = num_points
        l1 = 640
        l2 = 256
        l3 = 128
        self.conv1_r = torch.nn.Conv1d(2 * num_feat, l1, 1)
        self.conv1_t = torch.nn.Conv1d(2 * num_feat, l1, 1)

        self.conv2_r = torch.nn.Conv1d(l1, l2, 1)
        self.conv2_t = torch.nn.Conv1d(l1, l2, 1)

        self.conv3_r = torch.nn.Conv1d(l2, l3, 1)
        self.conv3_t = torch.nn.Conv1d(l2, l3, 1)

        self.conv4_r = torch.nn.Conv1d(l3, num_obj * 4, 1)
        self.conv4_t = torch.nn.Conv1d(l3, num_obj * 3, 1)
        if mode == 'NearestTranslationEmbStack':
            self.stack_net = NearestTranslationEmbStack()
        elif mode == 'LearnedEmbStack':
            self.stack_net = LearnedEmbStack()
        else:
            print("ERROR")
            raise Exception

    def forward(self, emb1, emb2, t1, t2, obj):
        # stack emb1 and emb2 in a usefull manner
        bs, num_feat, num_points = emb1.shape
        emb = self.stack_net(emb1, emb2, t1, t2)

        rx = F.relu(self.conv1_r(emb))
        tx = F.relu(self.conv1_t(emb))

        rx = F.relu(self.conv2_r(rx))
        tx = F.relu(self.conv2_t(tx))

        rx = F.relu(self.conv3_r(rx))
        tx = F.relu(self.conv3_t(tx))

        rx = self.conv4_r(rx).view(bs, self.num_obj, 4, self.num_points)
        tx = self.conv4_t(tx).view(bs, self.num_obj, 3, self.num_points)

        out_rx = torch.index_select(rx[0], 0, obj[0]).transpose(2, 1)
        out_tx = torch.index_select(tx[0], 0, obj[0]).transpose(2, 1)

        return out_rx, out_tx
