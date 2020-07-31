import os
import sys
import unittest
import torch
import torch.nn.functional as F
import numpy as np
sys.path.append(os.getcwd())
from lib.loss import KeypointLoss, FocalLoss, MultiObjectADDLoss
from torch import optim

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
        cls.loss = MultiObjectADDLoss()

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
        value, _ = self.loss(points, predictions, predictions, label, cube,
                cube_points[None], objects_in_scene, losses, {})
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
        add, add_s = self.loss(points, predictions, gt_keypoints, label, cube,
                cube_points[None], objects_in_scene, losses, {})
        self.assertAlmostEqual(add[1][0], np.linalg.norm(diff, 2))

    def test_mean_shift(self):
        loss = MultiObjectADDLoss(0.5, 100, 'mean_shift')
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
        value = loss(points, predictions, predictions, label, cube,
                cube_points[None], objects_in_scene, losses, {})
        self.assertEqual(value[0][1][0], 0.0)



class FocalLossTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.alpha = 0.25
        self.gamma = 1.0
        self.loss = FocalLoss(gamma=self.gamma, alpha=self.alpha, size_average=False)

    def test_loss(self):
        target = np.zeros((1, 2, 2)) # create 2 x 2 image with 2 objects and background (2 labels)
        target[0,1,0] = 1
        target[0,1,1] = 1
        target[0,0,1] = 2
        target = torch.tensor(target, dtype=torch.int64)

        inputs = np.ones((1, 3, 2, 2))*0.025 # N, H, W, C, default confidence 2.5%
        # set confidence 0.95 for all true positives
        inputs[0,0,0,0] = 0.95
        inputs[0,1,1,0] = 0.95
        inputs[0,1,1,1] = 0.95
        inputs[0,2,0,1] = 0.95

        inputs = torch.tensor(inputs)
        logits = torch.log(inputs) + torch.log(torch.exp(inputs).sum(dim=1))[:, None, :, :]

        val = self.loss(logits, target)

        loss = 3.0 * -self.alpha * (1.0 - 0.95) ** self.gamma * np.log(0.95)
        loss += -(1.0 - self.alpha) * (1.0 - 0.95) ** self.gamma * np.log(0.95)

        self.assertAlmostEqual(val.item(), loss)

    def test_batch(self):
        target = np.zeros((2, 2, 2))
        target[0, 1, 0] = 1
        target[0, 1, 1] = 1
        target[0, 0, 1] = 2
        target[1, 0, 0] = 1
        target[1, 0, 1] = 1
        target = torch.tensor(target, dtype=torch.int64)

        inputs = np.ones((2, 3, 2, 2)) * 0.025 # N, H, W, C, default confidence 2.5%
        # set confidence 0.95 for all true positives
        inputs[0, 0, 0, 0] = 0.95
        inputs[0, 1, 1, 0] = 0.95
        inputs[0, 1, 1, 1] = 0.95
        inputs[0, 2, 0, 1] = 0.95
        inputs[1, 1, 0, 0] = 0.95
        inputs[1, 1, 0, 1] = 0.95
        inputs[1, 0, 1, 0] = 0.95
        inputs[1, 0, 1, 1] = 0.95

        inputs = torch.tensor(inputs)
        logits = torch.log(inputs) + torch.log(torch.exp(inputs).sum(dim=1))[:, None, :, :]

        val = self.loss(logits, target)

        loss = 5.0 * -self.alpha * (1.0 - 0.95) ** self.gamma * np.log(0.95)
        loss += 3.0 * -(1.0 - self.alpha) * (1.0 - 0.95) ** self.gamma * np.log(0.95)

        self.assertAlmostEqual(val.item(), loss)

    def test_optimal(self):
        target = torch.zeros(1, 2, 2).to(torch.int64)
        target[0, 0, 1] = 1

        inputs = np.zeros((1, 3, 2, 2))
        inputs[0, 0, 0, 0] = 1.0
        inputs[0, 1, 0, 1] = 1.0
        inputs[0, 0, 1, 0] = 1.0
        inputs[0, 0, 1, 1] = 1.0
        inputs = torch.tensor(inputs, dtype=torch.float32, requires_grad=True)
        logits = torch.log(inputs + 1e-16) + torch.log(torch.exp(inputs).sum(dim=1))[:, None, :, :]

        loss = self.loss(logits, target)
        self.assertAlmostEqual(loss.item(), 0.0, 5)

        optimizer = optim.SGD([inputs], lr=1e-1)
        for _ in range(10):
            logits = torch.log(inputs + 1e-16) + torch.log(torch.exp(inputs).sum(dim=1))[:, None, :, :]
            loss = self.loss(logits, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            self.assertAlmostEqual(loss.item(), 0.0, 5)




if __name__ == "__main__":
    unittest.main()
