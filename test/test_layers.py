import unittest
import torch
import numpy as np
from lib.layers import SelectionConv2d

class TestLayers(unittest.TestCase):
    def test_selection(self):
        layer = SelectionConv2d(5, 3, 3)
        classes = torch.tensor([0, 2, 0])
        features = torch.ones(3, 3, 10, 10)
        with torch.no_grad():
            weight = torch.zeros(5, 3, 3)
            weight[2, :, :, :, :] = torch.ones(3, 3)
            layer.weight.set_(weight)
            layer.bias.set_(torch.zeros(5, 3))
            layer.weight[0]
        out = layer(features, classes).detach()[:, :, 5, 5]
        np.testing.assert_equal(out[0].numpy(), np.zeros(3))
        np.testing.assert_equal(out[1].numpy(), np.ones(3) * 3)
        np.testing.assert_equal(out[2].numpy(), np.zeros(3))

if __name__ == "__main__":
    unittest.main()


