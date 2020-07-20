import unittest
import numpy as np
import torch
from scipy.spatial.transform.rotation import Rotation
import os
import sys
sys.path.append(os.getcwd())
from lib import keypoint_helper as kh

class KeypointHelperTest(unittest.TestCase):
    def test_identity_one(self):
        points = torch.ones(1, 8, 3)
        gt_points = torch.ones(8, 3)
        T = kh.solve_transform(points, gt_points)
        self.assertEqual(T.shape[0], 1)
        self.assertEqual(T.shape[1], 4)
        self.assertEqual(T.shape[2], 4)
        np.testing.assert_allclose(T.numpy(), np.eye(4, dtype=np.float32)[None])

    def test_random_translation(self):
        gt_keypoints = torch.randn(8, 3)
        random_translation = torch.tensor(np.random.uniform(-1, 1, size=(2, 3)).astype(np.float32))
        keypoints = gt_keypoints[None].expand(2, -1, -1) + random_translation[:, None, :]
        T = kh.solve_transform(keypoints, gt_keypoints)
        np.testing.assert_allclose(random_translation, T[:, 0:3, 3].numpy(), 1e-5)

    def test_random_rotation(self):
        random_rotation = Rotation.random().as_matrix().astype(np.float32)
        random_rotation2 = Rotation.random().as_matrix().astype(np.float32)
        gt_keypoints = torch.randn(8, 3)
        keypoints = torch.zeros(2, 8, 3)
        keypoints[0] = (torch.tensor(random_rotation) @ gt_keypoints[:, :, None])[:, :, 0]
        keypoints[1] = (torch.tensor(random_rotation2) @ gt_keypoints[:, :, None])[:, :, 0]
        T = kh.solve_transform(keypoints, gt_keypoints)
        R1 = T[0, 0:3, 0:3].numpy()
        R2 = T[1, 0:3, 0:3].numpy()
        np.testing.assert_allclose(R1, random_rotation, 1e-4)
        np.testing.assert_allclose(R2, random_rotation2, 1e-4)

    def test_random_rotation_translation(self):
        random_rotation = Rotation.random().as_matrix().astype(np.float32)
        random_rotation2 = Rotation.random().as_matrix().astype(np.float32)
        random_t = np.random.uniform(-1, 1, size=3).astype(np.float32)
        random_t2 = np.random.uniform(-1, 1, size=3).astype(np.float32)
        gt_keypoints = torch.randn(8, 3)
        keypoints = torch.zeros(2, 8, 3)
        keypoints[0] = (torch.tensor(random_rotation) @ gt_keypoints[:, :, None])[:, :, 0] + random_t
        keypoints[1] = (torch.tensor(random_rotation2) @ gt_keypoints[:, :, None])[:, :, 0] + random_t2
        T = kh.solve_transform(keypoints, gt_keypoints)
        R1 = T[0, 0:3, 0:3].numpy()
        R2 = T[1, 0:3, 0:3].numpy()
        t1 = T[0, 0:3, 3].numpy()
        t2 = T[1, 0:3, 3].numpy()
        np.testing.assert_allclose(R1, random_rotation, 1e-3)
        np.testing.assert_allclose(R2, random_rotation2, 1e-3)
        np.testing.assert_allclose(t1, random_t, 1e-3)
        np.testing.assert_allclose(t2, random_t2, 1e-3)

class MeanShiftGaussianTest(unittest.TestCase):
    def test_one_cluster(self):
        kp = np.array([[0.0, 0.0, 0.0],
                       [-0.1, 0.0, -0.1],
                       [0.1, 0.0,  0.1],
                       [0.0, -0.1, 0.0],
                       [0.0, 0.1,  0.0],
                       [0.5, 0.3, 0.5],
                       [-0.5,- 0.3, -0.5]])
        keypoints = np.stack([kp, kp], axis=1)
        kernel = [0.1, 0.1, 0.1]
        result = kh.mean_shift_gaussian(keypoints, kernel)
        np.testing.assert_allclose(result, \
            np.array([ [ 0.0, 0.0, 0.0 ],
                       [ 0.0, 0.0, 0.0 ]]), 1e-6, 1e-10)

    def test_two_clusters(self):
        kp = np.array([[0.0, 0.0, 0.0],
                       [-0.1, 0.0, -0.1],
                       [0.1, 0.0,  0.1],
                       [0.1, 1.01, 0.0],
                       [-0.1, 0.99,  0.0]])
        keypoints = np.stack([kp, kp], axis=1)
        kernel = [0.1, 0.1, 0.1]
        result = kh.mean_shift_gaussian(keypoints, kernel)
        np.testing.assert_allclose(result, \
            np.array([ [ 0.0, 0.0, 0.0 ],
                       [ 0.0, 0.0, 0.0 ]]), 1e-6, 1e-10)


if __name__ == "__main__":
    unittest.main()

