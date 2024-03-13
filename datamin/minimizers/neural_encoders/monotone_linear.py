import torch
import torch.nn as nn
import torch.nn.functional as F


class MonotoneLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, scale: float):
        super(MonotoneLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.weight = nn.Parameter(torch.randn(self.out_features, self.in_features))
        self.bias = nn.Parameter(torch.randn(self.out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.square(self.weight)
        x = F.linear(x, w, None)
        x = x / self.scale
        x = x + (self.bias)  # tanh?
        return x
