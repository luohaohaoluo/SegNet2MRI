import torch
import torch.nn as nn

from .block1 import *
from .block2 import *


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()

        self.mpconv = nn.Sequential(
            # Extraction(in_ch, in_ch),
            nn.MaxPool3d(2, 2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        # x = self.sigmoid(x)
        return x


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

        # self.conv2 = nn.Conv3d(in_ch, out_ch, 1)
        # self.identity = nn.Identity()

    def forward(self, x):
        # identity = self.identity(x)
        # identity = self.conv2(identity)

        x = self.conv1(x)
        # x = x + identity

        return x


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch, kernel_size=2, stride=2)
        # self.attention = Attention(in_ch, skip_ch, out_ch)
        self.conv = DoubleConv(in_ch+skip_ch, out_ch)
        # self.extra = Extraction(out_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        # x = self.attention(x1, x2)
        x = self.conv(x)
        # x = self.extra(x)
        return x


class Baseline(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Baseline, self).__init__()
        features = [16, 32, 64, 128, 256]  # [16, 32, 64, 64, 128]

        self.inc = InConv(in_channels, features[0])
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])
        self.down4 = Down(features[3], features[4])

        self.up1 = Up(features[4], features[3], features[3])
        self.up2 = Up(features[3], features[2], features[2])
        self.up3 = Up(features[2], features[1], features[1])
        self.up4 = Up(features[1], features[0], features[0])
        self.outc = OutConv(features[0], num_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


if __name__ == '__main__':
    x = torch.randn(1, 4, 160, 160, 128)
    net = Baseline(in_channels=4, num_classes=4)
    y = net(x)
    print("params: ", sum(p.numel() for p in net.parameters()))
    print(y.shape)
