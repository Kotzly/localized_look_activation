from torch import nn
from torch.distributions import Normal
import torch

class LoLoInvSquare(nn.Module):
    
    def __init__(self, scale=1):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return 1/(1 + self.scale*x**2)

class LoLoGaussian(nn.Module):
    
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return Normal(0, 1).cdf(x)

class LoLoSigmoid(nn.Module):
    
    def __init__(self):
        super().__init__()

    def sigmoid(self, x):
        return 1/(1 + torch.exp(-x))

    def forward(self, x):
        sigmoid = self.sigmoid(x)
        return sigmoid * (1 - sigmoid)
