import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

class SelectionConv2d(nn.Module):
    def __init__(self, class_count, in_features, out_features):
        super().__init__()
        self.out_channels = out_features
        self.weight = nn.Parameter(torch.Tensor(class_count, out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(class_count, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x, classes):
        """
        x: N x C x H x W - input features
        classes: N - integer class selector
        """
        N, C, H, W = x.shape
        weights = self.weight[classes]
        bias = self.bias[classes]
        x = x.view(N, C, H*W).transpose(1, 2) # -> N x H*W x C

        import ipdb; ipdb.set_trace()
        weights = weights.reshape(self.out_channels, N * C, 1, 1)
        return F.conv2d(x, weights, bias, stride=1, groups=N)

