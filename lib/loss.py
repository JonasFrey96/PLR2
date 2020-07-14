from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from lib.knn.__init__ import KNearestNeighbor
from lib import keypoint_helper

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


class ADDLoss(_Loss):
    def __init__(self, num_points_mesh, sym_list):
        super().__init__(True)
        self.num_pt_mesh = num_points_mesh
        self.sym_list = sym_list

    def forward(self, pred_r, pred_t, pred_c, target, model_points, idx, points, w, refine):

        return loss_calculation(pred_r, pred_t, pred_c, target, model_points, idx, points, w, refine, self.num_pt_mesh, self.sym_list)

class KeypointLoss(_Loss):
    def __init__(self, keypoint_weight=1.0, center_weight=1.0, semantic_weight=1.0):
        super().__init__()
        self.keypoint_weight = keypoint_weight
        self.center_weight = center_weight
        self.semantic_weight = semantic_weight
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, p_keypoints, p_centers, p_semantic, gt_keypoints, gt_centers, gt_semantic):
        """
        keypoints: N x H x W x K x 3
        centers: N x H x W x 3
        semantic: N x H x W x C
        object_ids: N
        """
        loss_mask = gt_semantic != 0
        keypoint_diff = torch.pow(p_keypoints - gt_keypoints, 2).sum(dim=1)
        N = float(p_keypoints.shape[0])
        keypoint_loss = keypoint_diff[loss_mask].mean()
        center_diff = torch.pow(p_centers - gt_centers, 2).sum(dim=1)
        center_loss = center_diff[loss_mask].mean()
        semantic_loss = self.cross_entropy(p_semantic, gt_semantic)
        return (self.keypoint_weight * keypoint_loss +
                self.center_weight * center_loss +
                self.semantic_weight * semantic_loss)

class FocalLoss(_Loss):
    """Adapted from implementation by He et. al in PVN3D. Original code available
    at https://github.com/ethnhe/PVN3D.
    """
    def __init__(self, gamma=0, alpha=0.25, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        """
        semantic: N x H x W x C
        object_ids: N
        """ 
        input = input.view(input.size(0), input.size(1) * input.size(2), -1) # N, H, W, C => N, H*W, C
        input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim=1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            a_t_mask = np.tile(np.arange(input.size(1)), (input.size(0),1))
            target_mask = np.tile(target.numpy(), (1, input.size(1)))
            a_t = self.alpha[0].repeat(input.size(0),input.size(1))
            a_t_mask = torch.BoolTensor(a_t_mask == target_mask)
            a_t[a_t_mask] = self.alpha[1]
            loss = -1 * a_t*(1-pt)**self.gamma * logpt
        else:
            loss = -1 * (1-pt)**self.gamma * logpt

        loss = -1 * a_t*(1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

class MultiObjectADDLoss:
    def __init__(self, sym_list):
        self.sym_list = sym_list

    def __call__(self, points, p_keypoints, gt_keypoints, gt_label, model_keypoints,
            object_models, objects_in_scene, losses):
        """
        p_keypoints: N x K x 3 x H x W
        gt_keypoints: N x K x 3 x H x W
        model_keypoints: M x K x 3
        object_models: M x P x 3
        objects_in_scene: N x M
        returns dictionary of losses by object
        """
        gt_keypoints = points[:, None] + gt_keypoints
        p_keypoints = points[:, None] + p_keypoints
        N = p_keypoints.shape[0]
        for i in range(N):
            objects = torch.nonzero(objects_in_scene[i])[0]
            for object_index in objects:
                object_id = (object_index + 1).item()
                object_mask = gt_label[i] == object_id

                object_keypoints = p_keypoints[i, :, :, object_mask]
                gt_object_keypoints = gt_keypoints[i, :, :, object_mask]
                keypoints = model_keypoints[object_index]
                gt_object_keypoints = keypoint_helper.vote(gt_object_keypoints[None])
                object_keypoints = keypoint_helper.vote(object_keypoints[None])

                gt_T = keypoint_helper.solve_transform(gt_object_keypoints,
                        keypoints)[0]
                T_hat = keypoint_helper.solve_transform(object_keypoints,
                        keypoints)[0]
                if object_index in self.sym_list:
                    add = self._compute_add_symmetric(gt_T, T_hat, object_models[object_index])
                else:
                    add = self._compute_add(gt_T, T_hat, object_models[object_index])
                if object_id not in losses:
                    losses[object_id] = []
                losses[object_id].append(add.item())
        return losses


    def _compute_add(self, gt_T, T_hat, model_points):
        ones = torch.ones(model_points.shape[0], 1, dtype=model_points.dtype).to(gt_T.device)
        points = torch.cat([model_points, ones], dim=1)[:, :, None]
        ground_truth = (gt_T @ points)[:, :3, 0]
        predicted = (T_hat @ points)[:, :3, 0]
        return (ground_truth - predicted).norm(dim=1).mean()

    def _compute_add_symmetric(self, gt_T, T_hat, model_points):
        ones = torch.ones(model_points.shape[0], 1, dtype=model_points.dtype).to(gt_T.device)
        points = torch.cat([model_points, ones], dim=1)[:, :, None]
        ground_truth = (gt_T @ points)[:, :3, 0]
        predicted = (T_hat @ points)[:, :3, 0]
        dima = (ground_truth[None] - predicted[:, None]).norm(dim=2)
        min_values, _ = dima.min(dim=1)
        return min_values.mean(dim=0)


