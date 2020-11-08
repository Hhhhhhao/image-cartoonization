import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable

__all__ = ['TVLoss']


class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self,x):
        b, c, h, w = x.size()
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h-1,:]),2)
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w-1]),2)
        return h_tv / (c * h * w).mean() + w_tv / (c * h * w).mean()