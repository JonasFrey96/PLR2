from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import torch
import time
import numpy as np
import torch.nn as nn
import random
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
from lib.knn.__init__ import KNearestNeighbor
#from lib.knn import KNearestNeighbor

def loss_calculation(pred_r, pred_t, pred_c, target, model_points, idx, points, w, refine, num_point_mesh, sym_list):
    bs, num_p, _ = pred_c.size()

    pred_r = pred_r / (torch.norm(pred_r, dim=2).view(bs, num_p, 1))

    base = torch.cat(((1.0 - 2.0*(pred_r[:, :, 2]**2 + pred_r[:, :, 3]**2)).view(bs, num_p, 1),\
                      (2.0*pred_r[:, :, 1]*pred_r[:, :, 2] - 2.0*pred_r[:, :, 0]*pred_r[:, :, 3]).view(bs, num_p, 1), \
                      (2.0*pred_r[:, :, 0]*pred_r[:, :, 2] + 2.0*pred_r[:, :, 1]*pred_r[:, :, 3]).view(bs, num_p, 1), \
                      (2.0*pred_r[:, :, 1]*pred_r[:, :, 2] + 2.0*pred_r[:, :, 3]*pred_r[:, :, 0]).view(bs, num_p, 1), \
                      (1.0 - 2.0*(pred_r[:, :, 1]**2 + pred_r[:, :, 3]**2)).view(bs, num_p, 1), \
                      (-2.0*pred_r[:, :, 0]*pred_r[:, :, 1] + 2.0*pred_r[:, :, 2]*pred_r[:, :, 3]).view(bs, num_p, 1), \
                      (-2.0*pred_r[:, :, 0]*pred_r[:, :, 2] + 2.0*pred_r[:, :, 1]*pred_r[:, :, 3]).view(bs, num_p, 1), \
                      (2.0*pred_r[:, :, 0]*pred_r[:, :, 1] + 2.0*pred_r[:, :, 2]*pred_r[:, :, 3]).view(bs, num_p, 1), \
                      (1.0 - 2.0*(pred_r[:, :, 1]**2 + pred_r[:, :, 2]**2)).view(bs, num_p, 1)), dim=2).contiguous().view(bs * num_p, 3, 3)

    ori_base = base
    base = base.contiguous().transpose(2, 1).contiguous()
    model_points = model_points.view(bs, 1, num_point_mesh, 3).repeat(1, num_p, 1, 1).view(bs * num_p, num_point_mesh, 3)
    target = target.view(bs, 1, num_point_mesh, 3).repeat(1, num_p, 1, 1).view(bs * num_p, num_point_mesh, 3)
    ori_target = target
    pred_t = pred_t.contiguous().view(bs * num_p, 1, 3)
    ori_t = pred_t
    points = points.contiguous().view(bs * num_p, 1, 3)
    pred_c = pred_c.contiguous().view(bs * num_p)

    pred = torch.add(torch.bmm(model_points, base), points + pred_t)

    if not refine:
        if idx[0].item() in sym_list:
            target = target[0].transpose(1, 0).contiguous().view(3, -1)
            pred = pred.permute(2, 0, 1).contiguous().view(3, -1)
            inds = KNearestNeighbor.apply(target.unsqueeze(0), pred.unsqueeze(0), 1)
            target = torch.index_select(target, 1, inds.view(-1).detach() - 1)
            target = target.view(3, bs * num_p, num_point_mesh).permute(1, 2, 0).contiguous()
            pred = pred.view(3, bs * num_p, num_point_mesh).permute(1, 2, 0).contiguous()

    dis = torch.mean(torch.norm((pred - target), dim=2), dim=1)
    loss = torch.mean((dis * pred_c - w * torch.log(pred_c)), dim=0)


    pred_c = pred_c.view(bs, num_p)
    how_max, which_max = torch.max(pred_c, 1)
    dis = dis.view(bs, num_p)


    t = ori_t[which_max[0]] + points[which_max[0]]
    points = points.view(1, bs * num_p, 3)

    ori_base = ori_base[which_max[0]].view(1, 3, 3).contiguous()
    ori_t = t.repeat(bs * num_p, 1).contiguous().view(1, bs * num_p, 3)
    new_points = torch.bmm((points - ori_t), ori_base).contiguous()

    new_target = ori_target[0].view(1, num_point_mesh, 3).contiguous()
    ori_t = t.repeat(num_point_mesh, 1).contiguous().view(1, num_point_mesh, 3)
    new_target = torch.bmm((new_target - ori_t), ori_base).contiguous()

    # print('------------> ', dis[0][which_max[0]].item(), pred_c[0][which_max[0]].item(), idx[0].item())
    return loss, dis[0][which_max[0]], new_points.detach(), new_target.detach()


class Loss(_Loss):

    def __init__(self, num_points_mesh, sym_list):
        super(Loss, self).__init__(True)
        self.num_pt_mesh = num_points_mesh
        self.sym_list = sym_list

    def forward(self, pred_r, pred_t, pred_c, target, model_points, idx, points, w, refine):

        return loss_calculation(pred_r, pred_t, pred_c, target, model_points, idx, points, w, refine, self.num_pt_mesh, self.sym_list)

class KeypointLoss(_Loss):
    def __init__(self, num_points_mesh, num_keypoints=8):
        super().__init__(True)
        self.num_points_mesh = num_points_mesh
        self.num_keypoints = num_keypoints

    def forward(self, predicted_keypoints, gt_keypoints, points, gt_trans):
        # For each point and keypoint, calculate the displacement vector from the
        # point to the keypoint. Return the average mean squared error between the displacement
        # and the actual keypoints.
        # predicted: N x P x K x 3 or N x P x K * 3
        # gt_keypoints: N x K x 3
        # points: N x P x 3
        # gt_trans: N x 3
        shape = predicted_keypoints.shape
        N, P = (shape[0], shape[1])
        predicted_keypoints = predicted_keypoints.reshape(N, P, self.num_keypoints, 3)
        points = points[:, :, None, :].expand(-1, -1, self.num_keypoints, -1)

        # Keypoints are relative to object center.
        labels = gt_trans[:, None, :] + gt_keypoints # N x K x 3
        # Predicted keypoints are relative to the point from which it is predicted.
        y_hat = points + predicted_keypoints
        labels = labels[:, None, :, :].expand(-1, P, -1, -1)

        loss = torch.pow(y_hat - labels, 2).sum(dim=3).mean()
        distances = torch.norm(y_hat - labels, 2, dim=3).mean()
        return loss, distances

