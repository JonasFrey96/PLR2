if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.getcwd())
    sys.path.append(os.path.join(os.getcwd() + '/src'))
    sys.path.append(os.path.join(os.getcwd() + '/lib'))

from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import torch
import time
import numpy as np
import torch.nn as nn
import random
import torch.backends.cudnn as cudnn
from lib.knn.__init__ import KNearestNeighbor


def loss_calculation(pred_r, pred_t, target, model_points, idx, points, num_point_mesh, sym_list):
    """ADD loss calculation

    Args:
        pred_r ([type]): BS * 3
        pred_t ([type]): BS * 4
        idx ([type]): BS * 1

        model_points ([type]): BS * num_points * 3 : randomly selected points of the CAD model
        target ([type]): BS * num_points * 3 : model_points rotated and translated according to the regression goal (not ground truth because of data augmentation)

        sym_list ([list of integers]):
    Returns:
        [type]: [description]
    """
    bs, nr_tar_points, = target.size()

    num_input_points = len(points[0])

    pred_r = pred_r / (torch.norm(pred_r, dim=2).view(bs, num_p, 1))

    base = torch.cat(((1.0 - 2.0 * (pred_r[:, :, 2]**2 + pred_r[:, :, 3]**2)).view(bs, num_p, 1),
                      (2.0 * pred_r[:, :, 1] * pred_r[:, :, 2] - 2.0 *
                       pred_r[:, :, 0] * pred_r[:, :, 3]).view(bs, num_p, 1),
                      (2.0 * pred_r[:, :, 0] * pred_r[:, :, 2] + 2.0 *
                       pred_r[:, :, 1] * pred_r[:, :, 3]).view(bs, num_p, 1),
                      (2.0 * pred_r[:, :, 1] * pred_r[:, :, 2] + 2.0 *
                       pred_r[:, :, 3] * pred_r[:, :, 0]).view(bs, num_p, 1),
                      (1.0 - 2.0 * (pred_r[:, :, 1]**2 +
                                    pred_r[:, :, 3]**2)).view(bs, num_p, 1),
                      (-2.0 * pred_r[:, :, 0] * pred_r[:, :, 1] + 2.0 *
                       pred_r[:, :, 2] * pred_r[:, :, 3]).view(bs, num_p, 1),
                      (-2.0 * pred_r[:, :, 0] * pred_r[:, :, 2] + 2.0 *
                       pred_r[:, :, 1] * pred_r[:, :, 3]).view(bs, num_p, 1),
                      (2.0 * pred_r[:, :, 0] * pred_r[:, :, 1] + 2.0 *
                       pred_r[:, :, 2] * pred_r[:, :, 3]).view(bs, num_p, 1),
                      (1.0 - 2.0 * (pred_r[:, :, 1]**2 + pred_r[:, :, 2]**2)).view(bs, num_p, 1)), dim=2).contiguous().view(bs * num_p, 3, 3)

    ori_base = base
    base = base.contiguous().transpose(2, 1).contiguous()
    model_points = model_points.view(bs, 1, num_point_mesh, 3).repeat(
        1, num_p, 1, 1).view(bs * num_p, num_point_mesh, 3)
    target = target.view(bs, 1, num_point_mesh, 3).repeat(
        1, num_p, 1, 1).view(bs * num_p, num_point_mesh, 3)
    ori_target = target
    pred_t = pred_t.contiguous().view(bs * num_p, 1, 3)
    ori_t = pred_t

    pred = torch.add(torch.bmm(model_points, base), pred_t)

    if idx[0].item() in sym_list:
        target = target[0].transpose(1, 0).contiguous().view(3, -1)
        pred = pred.permute(2, 0, 1).contiguous().view(3, -1)
        inds = KNearestNeighbor.apply(
            target.unsqueeze(0), pred.unsqueeze(0), k=1)
        target = torch.index_select(target, 1, inds.view(-1) - 1)
        target = target.view(
            3, bs * num_p, num_point_mesh).permute(1, 2, 0).contiguous()
        pred = pred.view(
            3, bs * num_p, num_point_mesh).permute(1, 2, 0).contiguous()

    dis = torch.mean(torch.norm((pred - target), dim=2), dim=1)

    t = ori_t[0]
    points = points.view(1, num_input_points, 3)

    ori_base = ori_base[0].view(1, 3, 3).contiguous()
    ori_t = t.repeat(bs * num_input_points,
                     1).contiguous().view(1, bs * num_input_points, 3)
    new_points = torch.bmm((points - ori_t), ori_base).contiguous()

    new_target = ori_target[0].view(1, num_point_mesh, 3).contiguous()
    ori_t = t.repeat(num_point_mesh, 1).contiguous().view(1, num_point_mesh, 3)
    new_target = torch.bmm((new_target - ori_t), ori_base).contiguous()

    # print('------------> ', dis.item(), idx[0].item())
    return dis, new_points.detach(), new_target.detach()


class Loss_add(nn.Module):

    def __init__(self, sym_list):
        super(Loss_add, self).__init__()
        self.sym_list = sym_list

    def forward(self, pred_r, pred_t, target, model_points, idx, points):
        return loss_calculation(pred_r, pred_t, target, model_points, idx, points, self.num_pt_mesh, self.sym_list)


if __name__ == "__main__":
    from scipy.spatial.transform import Rotation as R
    import numpy as np
    from scipy.stats import special_ortho_group
    from helper import re_quat
    from deep_im import RearangeQuat

    print('test loss')
    bs = 10
    re_q = RearangeQuat(bs)
    mat = special_ortho_group.rvs(dim=3, size=bs)
    quat = R.from_matrix(mat).as_quat()

    q = torch.from_numpy(quat)
    re_q(q, input_format='xyzw')

    loss_add = Loss_add(sym_list=[1, 2, 3, 10])

    # nr_tar_points = 3000

    # tar = torch.ones((bs, nr_tar_points, 3))
    # a
    # (pred_r, pred_t
