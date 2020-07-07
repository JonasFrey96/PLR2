import torch

def vote(keypoints):
    """
    keypoints: N x P x K x 3
    return: N x K x 3 aggregated keypoints
    """
    return keypoints.mean(dim=1) # For now, just average over the points.

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
    I = torch.eye(3, 3)[None].repeat(N, 1, 1)
    I[:, 2, 2] = d.clone()

    R = U @ I @ Vt
    T = torch.zeros(N, 4, 4)
    T[:, 0:3, 0:3] = R
    T[:, 0:3, 3] = center[None] - (R @ gt_center[None :, None])[:, :, 0]
    T[:, 3, 3] = 1.0

    return T

def compute_points(points, predicted_keypoints):
    """
    points: N x P x 3 depth points
    predicted_keypoints: N x P x K x 3 keypoint predictions - one for each point and keypoint.
    return: N x P x K x 3 keypoints in world space
    """
    N, P, K, _ = predicted_keypoints.shape
    points = points[:, :, None, :].expand(-1, -1, K, -1)
    return points + predicted_keypoints
