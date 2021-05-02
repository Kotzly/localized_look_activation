import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from .model import LoLoLayer

def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, LoLoLayer):
        for param in m.layers.parameters():
            init.uniform_(param.data, a=-1., b=1.)
        if m.fusion is not None:
            init.uniform_(m.fusion_layer_w, a=-1., b=1.)
            if hasattr(m, "fusion_layer_b"):
                init.uniform_(m.fusion_layer_b, a=-1., b=1.)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)

def create_labels_checker(x, n=3, size=1, stride=2, start=0, size_factor=1, stride_factor=1):
    assert stride >= 2*size
    allcond = np.ones(x.shape[0]).astype(bool)
    for dim in range(x.shape[1]):
        stepcond = np.zeros(x.shape[0]).astype(bool)
        size_ = size
        stride_ = stride
        start_ = start
        end_ = start + size
        for i in range(n):
            cond = (x[:, dim] > start_) & (x[:, dim] < end_)
            stepcond = stepcond | cond
            
            size_ = size_ * size_factor
            stride_ = stride_ * stride_factor
            start_ = end_ + stride_
            end_ = start_ + size_
        allcond = stepcond & allcond
    return allcond
