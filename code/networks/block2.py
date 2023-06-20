import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                               bias=bias)
        self.gn1 = nn.GroupNorm(out_channels//2, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                               bias=bias)
        self.gn2 = nn.GroupNorm(out_channels//2, out_channels)

        # Adjust the input size to match the output size if necessary
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(out_channels//2, out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.gn1(out)

        out = self.conv2(out)
        out = self.relu(out)
        out = self.gn2(out)

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class Attention_Block(nn.Module):
    def __init__(self, in_ch, ou_ch):
        super(Attention_Block, self).__init__()
        self.ou_ch = ou_ch

        self.conv = nn.Conv3d(in_ch, ou_ch, 3, padding=1)

        self.fc1 = nn.Linear(ou_ch, ou_ch)
        self.fc2 = nn.Linear(ou_ch, ou_ch)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.gn = nn.BatchNorm3d(ou_ch)
        self.drop = nn.Dropout(0.4)

    def forward(self, x):
        # x.shape --> B, C, D, H, W

        x = self.conv(x)
        x = self.relu(x)
        x = self.gn(x)

        x = x.permute(0, 4, 2, 3, 1)
        x = F.adaptive_avg_pool3d(x, (1, 1, self.ou_ch))

        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.sigmoid(x)
        x = x.permute(0, 4, 2, 3, 1)

        return x


class Attention(nn.Module):
    def __init__(self, in_ch, ou_ch):
        super(Attention, self).__init__()

        self.path = Attention_Block(in_ch, ou_ch)

        self.conv1 = ResidualBlock(in_ch, ou_ch)

    def forward(self, x):

        x1 = self.path(x)

        x2 = self.conv1(x)

        output = torch.mul(x1, x2)

        return output


if __name__ == "__main__":
    x = torch.rand((2, 8, 128, 128, 64))
    net = Attention(8, 4)
    y = net(x)
    print("params: ", sum(p.numel() for p in net.parameters()))
    print(y.shape)
