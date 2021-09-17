from Model.utils import *


class UNet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UNet, self).__init__()

        # DownSampling Block
        self.down_block1 = DoubleConv(in_channel, 16)
        self.down_block2 = DownConv(16, 32)
        self.down_block3 = DownConv(32, 64)
        self.down_block4 = DownConv(64, 128)
        self.down_block5 = DownConv(128, 256)
        self.down_block6 = DownConv(256, 512)
        self.down_block7 = DownConv(512, 1024)

        # UpSampling Block
        self.up_block1 = UpSample(1024 + 512, 512)
        self.up_block2 = UpSample(512 + 256, 256)
        self.up_block3 = UpSample(256 + 128, 128)
        self.up_block4 = UpSample(128 + 64, 64)
        self.up_block5 = UpSample(64 + 32, 32)
        self.up_block6 = UpSample(32 + 16, 16)
        self.up_block7 = nn.Conv2d(16, out_channel, 1)

    def forward(self, x):
        # Down
        x1 = self.down_block1(x)
        x2 = self.down_block2(x1)
        x3 = self.down_block3(x2)
        x4 = self.down_block4(x3)
        x5 = self.down_block5(x4)
        x6 = self.down_block6(x5)
        x7 = self.down_block7(x6)

        # Up
        x8 = self.up_block1(x7, x6)
        x9 = self.up_block2(x8, x5)
        x10 = self.up_block3(x9, x4)
        x11 = self.up_block4(x10, x3)
        x12 = self.up_block5(x11, x2)
        x13 = self.up_block6(x12, x1)
        x14 = self.up_block7(x13)
        out = torch.sigmoid(x14)
        return out
