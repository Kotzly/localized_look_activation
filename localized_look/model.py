import torch
from torch import nn
import numpy as np
from . import activation as act

class LoLoLayer(nn.Module):
    
    def __init__(self, n_channels=3, n_kernels=3, fusion="channels", activation="invsquare"):
        """ Localized Look layer
        If fusion=="channels", all channels are fused and the output has n_kernel features.
        If fusion=="kernels", all kernels are fused and the output has n_channels features.
        
        Parameters
        ----------
        n_channels : int
        n_kernels : int
            Number of kernels
        fusion : [None, str]
            If None, no fusion is made and the output shape will be (-1, n_channels*n_kernels)
        
        
        """
        
        super().__init__()
        self.nc = n_channels
        self.nk = n_kernels
        self.fusion = fusion

        if isinstance(activation, str):
            activation = act.ACTIVATION_DICT[activation]

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, n_kernels),
                activation,
            ) for i in range(n_channels)
        ])
        if self.fusion:
            if self.fusion == "channels":
                size = n_kernels
            elif self.fusion == "kernels":
                size = n_channels
            self.fusion_layer_w = nn.Parameter(torch.Tensor(np.random.rand(1, n_kernels, n_channels)*2 - 1))
            self.fusion_layer_b = nn.Parameter(torch.Tensor(np.random.rand(1, size)*2 - 1))

    def forward(self, x):
        channels = [x[:, [i]] for i in range(self.nc)]
        looked = [layer(channel) for layer, channel in zip(self.layers, channels)]
        if self.fusion:
            looked = [tensor.unsqueeze(2) for tensor in looked]
            merged = torch.cat(looked, axis=2)
            if self.fusion == "channels":
                axis = 2
            if self.fusion == "kernels":
                axis = 1
            output = (merged*self.fusion_layer_w).sum(axis=axis) + self.fusion_layer_b
        else:
            output = torch.cat(looked, axis=1)
        return output


class ModelLoLo(nn.Module):
    def __init__(self, activation="invsquare"):
        super().__init__()
        self.layer = nn.Sequential(
            LoLoLayer(n_kernels=100, n_channels=2, fusion="channels", activation=activation),
            nn.LeakyReLU(.05),
            nn.Linear(100, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.layer(x)

class ModelReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(2, 26),
            nn.LeakyReLU(.05),
            nn.Linear(26, 26),
            nn.LeakyReLU(.05),
            nn.Linear(26, 1),
            nn.Sigmoid()
        )
