import torch
import torch.nn as nn


class ResNet(nn.Module):
    def __init__(self, num_feature, num_class):
        super(ResNet, self).__init__()

        conv_layers = []

        conv_layers.append(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=3, bias=False))
        conv_layers.append(nn.BatchNorm2d(64))
        conv_layers.append(nn.ReLU(inplace=True))
        conv_layers.append(nn.MaxPool2d(kernel_size=3, stride=2))

        Stage_1 = [64, 64]
        conv_layers.append(Conv_Block(stage=Stage_1, stride_size=1))
        for i in range(1,3):
            conv_layers.append(Identity_Block(channel_size=64, stride_size=1))
        conv_layers.append(nn.Dropout(0.2))

        Stage_2 = [64, 128]
        conv_layers.append(Conv_Block(stage=Stage_2, stride_size=2))
        for i in range(1,4):
            conv_layers.append(Identity_Block(channel_size=128, stride_size=1))
        conv_layers.append(nn.Dropout(0.2))

        Stage_3 = [128, 256]
        conv_layers.append(Conv_Block(stage=Stage_3, stride_size=2))
        for i in range(1,6):
            conv_layers.append(Identity_Block(channel_size=256, stride_size=1))
        conv_layers.append(nn.Dropout(0.2))

        Stage_4 = [256, 512]
        conv_layers.append(Conv_Block(stage=Stage_4, stride_size=2))
        for i in range(1,3):
            conv_layers.append(Identity_Block(channel_size=512, stride_size=1))
        conv_layers.append(nn.Dropout(0.2))

        conv_layers.append(nn.AdaptiveAvgPool2d((1,1)))
        conv_layers.append(nn.Dropout(0.2))
        self.conv = nn.Sequential(*conv_layers)

        # self.linear = nn.Linear(512, num_class)
        linear_layers_1 = []
        linear_layers_1.append(nn.Linear(512, num_feature))
        linear_layers_1.append(nn.BatchNorm1d(num_features=num_feature))
        self.linear_1 = nn.Sequential(*linear_layers_1)

        linear_layers_2 = []
        linear_layers_2.append(nn.ReLU(inplace=True))
        linear_layers_2.append(nn.Dropout(0.2))
        linear_layers_2.append(nn.Linear(num_feature, num_class))
        self.linear_2 = nn.Sequential(*linear_layers_2)

    def forward(self, x):
        out = self.conv(x)
        out = torch.flatten(out, 1)
        out = self.linear_1(out)
        out = self.linear_2(out)
        return out


class Identity_Block(nn.Module):
    def __init__(self, channel_size, stride_size):
        super(Identity_Block, self).__init__()
        layers = []
        
        layers.append(nn.Conv2d(in_channels=channel_size, out_channels=channel_size, kernel_size=3, stride=stride_size, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(channel_size))
        layers.append(nn.ReLU())

        layers.append(nn.Conv2d(in_channels=channel_size, out_channels=channel_size, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(channel_size))
        layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x_shortcut = x
        out = self.net(x)
        out += x_shortcut
        return out


class Conv_Block(nn.Module):
    def __init__(self, stage, stride_size):
        super(Conv_Block, self).__init__()
        self.stride_size = stride_size
        self.downsample = False
        in_size, out_size = stage[0], stage[1]

        layers_x = []

        layers_x.append(nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=3, stride=stride_size, padding=1, bias=False))
        layers_x.append(nn.BatchNorm2d(out_size))
        layers_x.append(nn.ReLU())

        layers_x.append(nn.Conv2d(in_channels=out_size, out_channels=out_size, kernel_size=3, stride=1, padding=1, bias=False))
        layers_x.append(nn.BatchNorm2d(out_size))
        layers_x.append(nn.ReLU())

        self.net_x = nn.Sequential(*layers_x)

        if in_size != out_size:
            layers_x_shortcut = []
            layers_x_shortcut.append(nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=1, stride=stride_size, bias=False))
            layers_x_shortcut.append(nn.BatchNorm2d(out_size))
            layers_x_shortcut.append(nn.ReLU())
            self.net_x_shortcut = nn.Sequential(*layers_x_shortcut)
            self.downsample = True
        
    def forward(self, x):
        if self.downsample is True:
            out_x_shortcut = self.net_x_shortcut(x)
        else:
            out_x_shortcut = x

        out_x = self.net_x(x)
        out = out_x + out_x_shortcut
        return out