import torch


def motion_loss(out_rx, out_tx, gt_rot_wxyz_ls, gt_trans_ls):
    gt_delta = gt_trans_ls[1] - gt_trans_ls[0]
    gt_delta = gt_delta.view(-1, 1, 3).repeat(1, 1000, 1).type(torch.float32)

    loss = torch.dist(gt_delta, out_tx)

    return loss
