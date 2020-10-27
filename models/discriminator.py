import torch.nn as nn
import torch.nn.functional as F
from .utils import InstanceNormalization

__all__ = ['Discriminator']


class Discriminator(nn.Module):
    """
    CartoonGAN Discriminator
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv_1 = nn.Conv2d(3, 32, 3, 1)
        # leak_relu

        self.conv_2_1 = nn.Conv2d(32, 64, 3, 2, 1)
        # leak_relu
        self.conv_2_2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.in_2 = InstanceNormalization(128)
        # leak_relu

        self.conv_3_1 = nn.Conv2d(128, 128, 3, 2, 1)
        # leak_relu
        self.conv_3_2 = nn.Conv2d(128, 256, 3, 1, 1)
        self.in_3 = InstanceNormalization(256)
        # leak_relu

        self.conv_4 = nn.Conv2d(256, 256, 3, 1, 1)
        self.in_4 = InstanceNormalization(256)
        # leak_relu

        self.conv5 = nn.Conv2d(256, 1, 3, 1, 1)

    def forward(self, x):
        x = F.leaky_relu(self.conv_1(x), negative_slope=0.2)

        x = F.leaky_relu(self.conv_2_1(x), negative_slope=0.2)
        x = F.leaky_relu(self.in_2(self.conv_2_2(x)), negative_slope=0.2)

        x = F.leaky_relu(self.conv_3_1(x), negative_slope=0.2)
        x = F.leaky_relu(self.in_3(self.conv_3_2(x)), negative_slope=0.2)

        x = F.leaky_relu(self.in_4(self.conv_4(x)), negative_slope=0.2)

        x = self.conv5(x)

        return x