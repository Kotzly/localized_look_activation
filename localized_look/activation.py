from torch import nn
from torch.distributions import Normal
import torch

class LoLoInvSquare(nn.Module):
    
    def __init__(self, scale=1):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return 1/(1 + (self.scale*x)**2)

class LoLoInv4(nn.Module):
    
    def __init__(self, scale=1):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return 1/(1 + (self.scale*x)**4)

class AdaptativeSigmoid(nn.Module):
    def __init__(self, scale=1, a=1, b=1):
        super().__init__()
        self.scale = scale
        self.a = nn.Parameter(torch.Tensor(a))
        self.b = nn.Parameter(torch.Tensor(b))
    
    def sigmoid(self, x):
        return 1/(1 + torch.exp(-x))

    def forward(self, x):
        pred = self.sigmoid(x + self.a) * (1 - self.sigmoid(x - self.b))
        return pred

class LoLoInvLogSquared(nn.Module):
    
    def __init__(self, scale=1):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return 1/(1 + torch.log((self.scale*x)**2 + 1))

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

class SinX(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)/x

class CosOverCosh(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.cos(x)/(torch.exp(x) + torch.exp(-x))

class InvCosh(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 1/torch.cosh(x)

class TanhX(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.tanh(x)/x

class OneMinusTanhSquared(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 1 - torch.tanh(x)**2







ACTIVATION_DICT = {
    "invsquare": LoLoInvSquare(),
    "gaussian": LoLoInvSquare(),
    "sigmoid": LoLoSigmoid(),
    "inv4": LoLoInv4(),
    "logsquared": LoLoInvLogSquared(),
    "adaptative": AdaptativeSigmoid(),
    "sinx": SinX(),
    "cosovercosh": CosOverCosh(),
    "invcosh": InvCosh(),
    "tanhx": TanhX(),
    "omtq": OneMinusTanhSquared()
}
