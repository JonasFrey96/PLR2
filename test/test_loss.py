import unittest
import torch
import numpy as np
from lib.loss import KeypointLoss, MultiObjectADDLoss

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




if __name__ == "__main__":
    unittest.main()

