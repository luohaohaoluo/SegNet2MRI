import torch
import torch.nn as nn
import torch.nn.functional as F


class Extraction(nn.Module):
    def __init__(self, in_ch, ou_ch):
        super(Extraction, self).__init__()

        self.step1_1 = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(in_ch // 2, in_ch)
        )

        self.step1_2 = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(in_ch // 2, in_ch)
        )

        self.step1_3 = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(in_ch // 2, in_ch)
        )

        self.step1_4 = nn.Sequential(
            nn.Conv3d(in_ch, ou_ch, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(ou_ch // 2, ou_ch)
        )

        self.step2 = nn.Sequential(
            nn.Conv3d(ou_ch, ou_ch, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.GroupNorm(ou_ch // 2, ou_ch)
        )

        self.step3 = nn.Sequential(
            nn.Conv3d(ou_ch, ou_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(ou_ch // 2, ou_ch)
        )

        self.step4 = nn.Sequential(
            nn.Conv3d(ou_ch*4, ou_ch, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(ou_ch // 2, ou_ch)
        )

    def forward(self, x):
        x1 = self.step1_1(x)
        x2 = self.step1_2(x)
        x3 = self.step1_3(x)
        x4 = self.step1_4(x)

        x1_2 = self.step2(x1+x2)
        x1_2_3 = self.step3(x1_2 + x3)
        x1_2_3_4 = x1_2_3 + x4

        out = torch.cat([x1, x1_2, x1_2_3, x1_2_3_4], dim=1)
        out = self.step4(out)

        return out


if __name__ == "__main__":
    x = torch.rand((1, 4, 128, 128, 64))
    net = Extraction(4, 4)
    y = net(x)
    print("params: ", sum(p.numel() for p in net.parameters()))
    print(y.shape)
