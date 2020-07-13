import os
import sys
import unittest
import torch
import torch.nn.functional as F
import numpy as np
sys.path.append(os.getcwd())
from lib.loss import KeypointLoss, FocalLoss

def build_cube():
    x, y, z = [np.zeros(3) for _ in range(3)]
    x[0] = 1.0
    y[1] = 1.0
    z[2] = 1.0
    cube_vertices = np.zeros((8, 3))
    multipliers = (-1, 1)
    for i in range(2):
        i_m = multipliers[i]
        for j in range(2):
            j_m = multipliers[j]
            for k in range(2):
                k_m = multipliers[k]
                index = i * 4 + j * 2 + k
                cube_vertices[index] = i_m * x + j_m * y + k_m * z
    return torch.tensor(cube_vertices)

class KeypointLossTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.keypoint_count = 8
        cls.loss = KeypointLoss(None, cls.keypoint_count)

    # def test_single_zero(self):
    #     point_count = 100
    #     cube = build_cube()[None]
    #     predictions = torch.tensor(np.zeros((1, point_count, self.keypoint_count, 3)))
    #     points = torch.randn(1, point_count, 3)
    #     for i in range(point_count):
    #         point = points[0, i, :]
    #         for k in range(self.keypoint_count):
    #             keypoint = cube[0, k, :]
    #             predictions[0, i, k, :] = keypoint - point
    #     translation = torch.zeros(1, 3)
    #     value, _ = self.loss(predictions, cube, points, translation)
    #     self.assertEqual(0.0, value.item())

    # def test_batch_size_two(self):
    #     point_count = 5
    #     cube = build_cube()[None].repeat(2, 1, 1)
    #     predictions = torch.tensor(np.zeros((2, point_count, self.keypoint_count * 3)))
    #     translation = torch.zeros(2, 3)
    #     points = cube[0, 0][None].expand(2, point_count, -1)
    #     gt_points = cube[0, 0][None].expand(2, 8, -1)
    #     value, _ = self.loss(predictions, gt_points, points, translation)
    #     self.assertEqual(0.0, value)

    # def test_error_batch(self):
    #     point_count = 10
    #     batch_size = 4
    #     cube = build_cube()[None].expand(batch_size, -1, -1)
    #     predictions = torch.tensor(np.zeros((batch_size, point_count, self.keypoint_count, 3)))
    #     points = torch.randn(batch_size, point_count, 3)
    #     for b in range(batch_size):
    #         error = torch.zeros(3)
    #         error[b % 3] = 1.0
    #         for i in range(point_count):
    #             point = points[b, i, :]
    #             for k in range(self.keypoint_count):
    #                 keypoint = cube[b, k, :]
    #                 predictions[b, i, k, :] = keypoint - point + error
    #     translation = torch.zeros(batch_size, 3)
    #     value, _ = self.loss(predictions, cube, points, translation)
    #     self.assertEqual(1.0, value.item())

class FocalLossTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # make 3x3 ground truth label image
        target = np.zeros((1, 2, 2)) # create 2 x 2 image with 1 object and background (2 labels)
        target[0,1,0] = 1
        target[0,1,1] = 1
        target[0,0,1] = 2
        self.target = torch.tensor(target, dtype=torch.int64)
        
        inputs = np.ones((1, 2, 2, 3))*0.025 # N, H, W, C, default confidence 2.5%
        # set confidence 0.95 for all true positives
        inputs[0,0,0,0] = 0.95
        inputs[0,0,1,2] = 0.95
        inputs[0,1,0,1] = 0.95
        inputs[0,1,1,1] = 0.95

        self.inputs = torch.Tensor(inputs)

        self.loss = FocalLoss(gamma=1, alpha=0.25, size_average=False)

    def test_loss(self):
        val = self.loss.forward(self.inputs, self.target)
        self.assertLessEqual(val.detach().numpy() - 3.1252615, 1e-6)

if __name__ == "__main__":
    unittest.main()