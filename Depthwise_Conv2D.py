import torch
from torch import nn

class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size = 5, padding = 2, bias=False):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = torch.nn.Conv2d(nin, nin, kernel_size=kernel_size, padding=padding, groups=nin, bias=bias)
        self.pointwise = torch.nn.Conv2d(nin, nout, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out