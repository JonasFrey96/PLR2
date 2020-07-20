import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)
import torch
import numpy as np
from . import mean_shift as ms

def vote(keypoints):
    """
    keypoints: N x K x 3 x P
    return: N x K x 3 aggregated keypoints
    """
    return keypoints.mean(dim=3) # For now, just average over the points.

def solve_transform(keypoints, gt_keypoints):
    """
    keypoints: N x K x 3
    gt_keypoints: K x 3
    return: N x 4 x 4 transformation matrix
    """
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

def mean_shift_gaussian(keypoints, kernel_bandwidth=[0.1, 0.1, 0.1]):
    """
    keypoints: N x K x 3
    return: K x 3 averaged keypoints
    """
    if type(keypoints) != np.ndarray:
        keypoints = keypoints.numpy()

    mean_shifter = ms.MeanShift(kernel='multivariate_gaussian')

    kp_ms = []

    for k in range(keypoints.shape[1]): # do for K keypoints
        kp = keypoints[:,k,:]
        result = mean_shifter.cluster(kp, kernel_bandwidth)
        clusters, counts = np.unique(result.cluster_ids, return_counts=True)
        kp_mean = np.mean( result.shifted_points[ \
            result.cluster_ids == clusters[ np.argmax(counts) ] ], axis=0)
        kp_ms.append(kp_mean)
    
    return np.stack(kp_ms)

    