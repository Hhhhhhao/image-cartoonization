import torch.nn as nn
import math
from .utils import InstanceNorm
from torch.nn.utils import spectral_norm

__all__ = ['Discriminator']


class Discriminator(nn.Module):
    """
    CartoonGAN Discriminator
    """
    def __init__(self, image_size=256, down_size=64):
        super(Discriminator, self).__init__()

        self.image_size = image_size
        self.down_size = down_size
        self.num_down = int(math.log2(self.image_size // self.down_size))

        self.conv_in = nn.Sequential(
            spectral_norm(nn.Conv2d(3, 32, kernel_size=3, padding=1)),
            InstanceNorm(32),
            nn.LeakyReLU(negative_slope=0.1),
        )

        feat_dim = 32
        down_layers = []
        for i in range(self.num_down):
            down_layers.append(nn.Sequential(
                spectral_norm(nn.Conv2d(feat_dim, feat_dim * 2, kernel_size=3, stride=2, padding=1)),
                InstanceNorm(feat_dim * 2),
                nn.LeakyReLU(negative_slope=0.1),
                spectral_norm(nn.Conv2d(feat_dim * 2, feat_dim * 4, kernel_size=3, stride=1, padding=1)),
                InstanceNorm(feat_dim * 4),
                nn.LeakyReLU(negative_slope=0.1)
            ))
            feat_dim = feat_dim * 4
        self.conv_down = nn.Sequential(*down_layers)

        self.conv_out = nn.Sequential(
            spectral_norm(nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1)),
            InstanceNorm(feat_dim),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(feat_dim, 1, kernel_size=1, padding=0)
        )

    def forward(self, x):
        out = self.conv_in(x)
        out = self.conv_down(out)
        out = self.conv_out(out)
        return out


if __name__ == '__main__':
    import torch
    model = Discriminator(128, down_size=32)
    a = torch.randn((4, 3, 128, 128))
    out = model(a)
    print(out.shape)