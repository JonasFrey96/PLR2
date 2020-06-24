import torch
import torch.nn as nn


class MotionNetwork(nn.Module):

    def __init__(self, num_points, num_obj):
        super(MotionNetwork, self).__init__()
        emb_size = 1408
        l1 = 640
        l2 = 256
        l3 = 128
        l4 = num_obj * 21
        self.conv1_r = torch.nn.Conv1d(3 * emb_size, l1, 1)
        self.conv1_t = torch.nn.Conv1d(3 * emb_size, l1, 1)

        self.conv1_r = torch.nn.Conv1d(l1, l2, 1)
        self.conv1_t = torch.nn.Conv1d(l1, l2, 1)

        self.conv1_r = torch.nn.Conv1d(l2, l3, 1)
        self.conv1_t = torch.nn.Conv1d(3 * emb_size, 640, 1)

        self.conv1_r = torch.nn.Conv1d(3 * emb_size, 640, 1)
        self.conv1_t = torch.nn.Conv1d(3 * emb_size, 640, 1)

        self.conv1_r = torch.nn.Conv1d(3 * emb_size, 640, 1)
        self.conv1_t = torch.nn.Conv1d(3 * emb_size, 640, 1)

    def forward(self, emb1, emb2, emb3):
        emb = torch.stack([emb1, emb2, emb3])
