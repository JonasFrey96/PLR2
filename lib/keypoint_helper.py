import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)
import torch
import numpy as np
from lib.meanshift_pytorch import MeanShiftTorch

class Mean:
    def fit(self, x):
        return x.mean(dim=0)[None]

class VotingModule:
    def __init__(self, method='mean_shift', **kwargs):
        if method == 'mean':
            self.cluster = Mean()
        elif method == 'mean_shift':
            self.cluster = MeanShiftTorch(**kwargs)
        else:
            raise NotImplementedError("This clustering voting method has not been implemented")
        self.n_points = 1000

    def __call__(self, object_keypoints):
        """
        object_keypoints: K x 3 x P
        """
        obj_kp_means = []
        selected_points = np.random.choice(np.arange(object_keypoints.shape[2]), self.n_points)
        for k in range(object_keypoints.shape[0]):
            obj_kp_means.append(self.cluster.fit(object_keypoints[k, :, selected_points].T)[0])
        return torch.stack(obj_kp_means, dim=0)

def solve_transform(keypoints, gt_keypoints):
    """
    keypoints: N x K x 3
    gt_keypoints: K x 3
    return: N x 4 x 4 transformation matrix
    """
    keypoints = keypoints.clone()
    gt_keypoints = gt_keypoints.clone()
    N, K, _ = keypoints.shape
    center = keypoints.mean(dim=1)
    gt_center = gt_keypoints.mean(dim=0)
    keypoints -= center[:, None, :]
    gt_keypoints -= gt_center[None]
    matrix = keypoints.transpose(2, 1) @ gt_keypoints[None]
    U, S, V = torch.svd(matrix)
    Vt = V.transpose(2, 1)
    Ut = U.transpose(2, 1)

    d = (V @ Ut).det()
    I = torch.eye(3, 3, dtype=gt_center.dtype)[None].repeat(N, 1, 1).to(keypoints.device)
    I[:, 2, 2] = d.clone()

    R = U @ I @ Vt
    T = torch.zeros(N, 4, 4, dtype=gt_center.dtype).to(keypoints.device)
    T[:, 0:3, 0:3] = R
    T[:, 0:3, 3] = center[None] - (R @ gt_center[None :, None])[:, :, 0]
    T[:, 3, 3] = 1.0

    return T


