import torch
import torch.nn as nn

# 定义通道注意力模块 (CAM)
class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=64):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x

# 定义空间注意力模块 (SAM)
class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat_out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(concat_out)
        return self.sigmoid(out) * x

# 定义主网络结构
class CSAUNet(nn.Module):
    def __init__(self, input_channels):
        super(CSAUNet, self).__init__()
        # 下采样路径
        self.enc1 = self._conv_block(input_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)

        self.dec3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        # 上采样路径
        self.up3 = self._up_block(512, 256)
        self.up2 = self._up_block(256, 128)
        self.up1 = self._up_block(128, 64)

        # CAM 和 SAM
        self.cam1 = ChannelAttentionModule(64)
        self.sam1 = SpatialAttentionModule()
        self.cam2 = ChannelAttentionModule(128)
        self.sam2 = SpatialAttentionModule()
        self.cam3 = ChannelAttentionModule(256)
        self.sam3 = SpatialAttentionModule()

        # 最终卷积
        self.final_conv = nn.Conv2d(64, 2, kernel_size=1)

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2,   mode='bilinear', align_corners=True),
            # nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        )

    def forward(self, x):
        # 编码
        x1 = self.enc1(x)
        x2 = self.enc2(nn.MaxPool2d(2)(x1))
        x3 = self.enc3(nn.MaxPool2d(2)(x2))
        x4 = self.enc4(nn.MaxPool2d(2)(x3))

        # 解码
        d3 = self.up3(x4)
        d3 = torch.cat([self.sam3(self.cam3(x3)), d3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([self.sam2(self.cam2(x2)), d2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([self.sam1(self.cam1(x1)), d1], dim=1)
        d1 = self.dec1(d1)

        # 输出
        out = self.final_conv(d1)
        return torch.unsqueeze(out[:,-1,:,:],dim=1)
