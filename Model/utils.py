import torch
from torch import nn


class DoubleConv(nn.Module):
    """
        Double Convolution layer with both 2 BN and Activation Layer in between
        Conv2d==>BN==>Activation==>Conv2d==>BN==>Activation
    """

    def __init__(self, in_channel, out_channel):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channel)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class DownConv(nn.Module):
    """
      A maxpool layer followed by a Double Convolution.
      MaxPool2d==>double_conv.
    """

    def __init__(self, in_channel, out_channel):
        super(DownConv, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channel, out_channel)
        )

    def forward(self, x):
        x = self.down(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UpSample, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.double_conv = DoubleConv(in_channel, out_channel)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        x = self.double_conv(x)
        return x
