import torch
import torch.nn as nn
import math
from .utils import InstanceNorm, AdaInstanceNorm


class ResConv(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size):
        super(ResConv, self).__init__()
        padding = kernel_size // 2

        self.pad1 = nn.ReflectionPad2d(padding)
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size)
        self.norm1 = InstanceNorm(out_dim)
        self.act1 = nn.ReLU(inplace=True)

        self.pad2 = nn.ReflectionPad2d(padding)
        self.conv2 = nn.Conv2d(in_dim, out_dim, kernel_size)
        self.norm2 = InstanceNorm(out_dim)
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out)
        out = self.act1(out)

        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out)

        out += identity
        out = self.act2(out)
        return out


class Generator(nn.Module):
    def __init__(self, image_size=256, down_size=64, num_res=8, skip_conn=False):
        super(Generator, self).__init__()
        self.image_size = image_size
        self.down_size = down_size
        self.num_res = num_res
        self.num_down = int(math.log2(self.image_size // self.down_size))
        self.skip_conn = skip_conn

        # input layer
        self.conv_in = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(3, 64, kernel_size=5),
            InstanceNorm(64),
            nn.ReLU(inplace=True))

        # downsample layers
        feat_dim = 64
        self.down_layers = nn.ModuleList()
        for i in range(self.num_down):
            next_feat_dim = feat_dim * 2
            self.down_layers.append(nn.Sequential(
                nn.Conv2d(feat_dim, next_feat_dim, kernel_size=3, stride=2, padding=1),
                InstanceNorm(next_feat_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(next_feat_dim, next_feat_dim, kernel_size=3, stride=1, padding=1),
                InstanceNorm(next_feat_dim),
                nn.ReLU(inplace=True)
            ))
            feat_dim = next_feat_dim

        # residual layers
        res_layers = []
        for i in range(self.num_res):
            res_layers.append(ResConv(feat_dim, feat_dim, 3))
        self.res_layers = nn.Sequential(*res_layers)

        # upsample layers
        self.up_layers = nn.ModuleList()
        for i in range(self.num_down):
            next_feat_dim = feat_dim // 2
            self.up_layers.append(nn.Sequential(
                nn.ConvTranspose2d(feat_dim * 2 if self.skip_conn else feat_dim, next_feat_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                InstanceNorm(next_feat_dim),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(next_feat_dim, next_feat_dim, kernel_size=3, stride=1, padding=1),
                InstanceNorm(next_feat_dim),
                nn.ReLU(inplace=True)
            ))
            feat_dim = next_feat_dim

        # output layer
        self.conv_out = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(64, 3, kernel_size=5),
            nn.Tanh())

    def forward(self, x):
        out, down = self.forward_encoder(x)

        for i, up_layer in enumerate(self.up_layers):
            if self.skip_conn:
                en = down[i]
                out = torch.cat([out, en], dim=1)
            out = up_layer(out)

        out = self.conv_out(out)
        return out

    def forward_encoder(self, x):
        out = self.conv_in(x)

        down = []
        for down_layer in self.down_layers:
            out = down_layer(out)
            down.append(out)
        down = down[::-1]

        out = self.res_layers(out)
        return out, down



class UpBlock(nn.Module):
    def __init__(self, in_dim, out_dim, style_dim):
        super(UpBlock, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.norm1 = AdaInstanceNorm(style_dim, out_dim)
        self.act1 = nn.ReLU(inplace=False)
        self.conv2 = nn.ConvTranspose2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.norm2 = AdaInstanceNorm(style_dim, out_dim)
        self.act2 = nn.ReLU(inplace=False)

    def forward(self, x, s):
        out = self.conv1(x)
        out = self.norm1(out, s)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.norm2(out, s)
        out = self.act2(out)
        return out


class StarGenerator(nn.Module):
    def __init__(self, image_size=256, down_size=64, num_res=8, skip_conn=False, style_dim=64):
        super(StarGenerator, self).__init__()
        self.image_size = image_size
        self.down_size = down_size
        self.num_res = num_res
        self.num_down = int(math.log2(self.image_size // self.down_size))
        self.skip_conn = skip_conn
        self.style_dim = style_dim

        # input layer
        self.conv_in = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(3, 64, kernel_size=5),
            InstanceNorm(64),
            nn.ReLU(inplace=False))

        # downsample layers
        feat_dim = 64
        self.down_layers = nn.ModuleList()
        for i in range(self.num_down):
            next_feat_dim = feat_dim * 2
            self.down_layers.append(nn.Sequential(
                nn.Conv2d(feat_dim, next_feat_dim, kernel_size=3, stride=2, padding=1),
                InstanceNorm(next_feat_dim),
                nn.ReLU(inplace=False),
                nn.Conv2d(next_feat_dim, next_feat_dim, kernel_size=3, stride=1, padding=1),
                InstanceNorm(next_feat_dim),
                nn.ReLU(inplace=False)
            ))
            feat_dim = next_feat_dim

        # residual layers
        res_layers = []
        for i in range(self.num_res):
            res_layers.append(ResConv(feat_dim, feat_dim, 3))
        self.res_layers = nn.ModuleList(res_layers)

        # upsample layers
        self.up_layers = nn.ModuleList()
        for i in range(self.num_down):
            next_feat_dim = feat_dim // 2
            self.up_layers.append(UpBlock(feat_dim * 2 if self.skip_conn else feat_dim, next_feat_dim, self.style_dim))
            feat_dim = next_feat_dim

        # output layer
        self.conv_out = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(64, 3, kernel_size=5),
            nn.Tanh())

    def forward(self, x, s):

        out, down = self.forward_encoder(x)

        for i, up_layer in enumerate(self.up_layers):
            if self.skip_conn:
                en = down[i]
                out = torch.cat([out, en], dim=1)
            out = up_layer(out, s)

        out = self.conv_out(out)
        return out

    def forward_encoder(self, x):
        out = self.conv_in(x)

        feat_list = []
        for i, down_layer in enumerate(self.down_layers):
            out = down_layer(out)
            if i % 2 == 0:
                feat_list.append(out)

        for res_layer in self.res_layers:
            out = res_layer(out)
            feat_list.append(out)
        return feat_list



if __name__ == '__main__':
    model = Generator(skip_conn=True)
    a = torch.randn((4, 3, 256, 256))
    out = model(a)





        







