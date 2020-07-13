import os
import sys
import unittest
import torch
import torch.nn.functional as F
import numpy as np
sys.path.append(os.getcwd())
from lib.loss import KeypointLoss, FocalLoss, MultiObjectADDLoss

def build_cube():
    x, y, z = np.eye(3)
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

def sample_cube_points(n=100):
    out = np.zeros((n, 3))
    I = np.eye(3)
    for i in range(n):
        face = np.random.randint(3)
        u = I[face, :] * np.random.choice([-1, 1])
        v = I[(face + 1) % 3, :]
        w = I[(face + 2) % 3, :]
        out[i] = u + np.random.uniform(-1, 1) * v + np.random.uniform(-1, 1) * w

    return torch.tensor(out)

class ADDLossTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.keypoint_count = 8
        cls.loss = MultiObjectADDLoss([])

    def test_zero(self):
        point_count = 100
        cube = build_cube()[None]
        H = 100
        W = 200
        predictions = torch.tensor(np.zeros((1, self.keypoint_count, 3, H, W)))
        points = torch.randn(1, 3, H, W, dtype=torch.float64)
        for h in range(H):
            for w in range(W):
                point = points[0, :, h, w]
                for k in range(self.keypoint_count):
                    keypoint = cube[0, k, :]
                    predictions[0, k, :, h, w] = keypoint - point
        translation = torch.zeros(1, 3)
        label = torch.tensor(np.zeros((1, 100, 200), dtype=np.int64))
        label[0, 25:75, 50:150] = 1
        cube_points = sample_cube_points()
        objects_in_scene = torch.tensor(np.ones((1, 1), dtype=np.int64))
        losses = {}
        value = self.loss(points, predictions, predictions, label, cube,
                cube_points[None], objects_in_scene, losses)
        self.assertEqual(value[1][0], 0.0)

    def test_translation(self):
        point_count = 100
        H = 100
        W = 200
        translation = np.zeros((1, self.keypoint_count, 3, H, W))
        diff = np.random.randn(3)
        translation[:, :, :, :, :] = diff[None, None, :, None, None]
        cube = build_cube()[None]
        predictions = torch.tensor(np.zeros((1, self.keypoint_count, 3, H, W)))
        points = torch.randn(1, 3, H, W, dtype=torch.float64)
        for h in range(H):
            for w in range(W):
                point = points[0, :, h, w]
                for k in range(self.keypoint_count):
                    keypoint = cube[0, k, :]
                    predictions[0, k, :, h, w] = keypoint - point
        label = torch.tensor(np.zeros((1, 100, 200), dtype=np.int64))
        label[0, 25:75, 50:150] = 1
        cube_points = sample_cube_points()
        objects_in_scene = torch.tensor(np.ones((1, 1), dtype=np.int64))
        losses = {}
        gt_keypoints = predictions + translation
        value = self.loss(points, predictions, gt_keypoints, label, cube,
                cube_points[None], objects_in_scene, losses)
        self.assertAlmostEqual(value[1][0], np.linalg.norm(diff, 2))



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