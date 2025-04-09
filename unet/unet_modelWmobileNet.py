from .unet_parts import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, backbone='mobilenetv3_large_100'):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.backbone = timm.create_model(backbone, pretrained=True, features_only=True, in_chans=n_channels)
        encoder_channels = self.backbone.feature_info.channels()  # [16, 24, 40, 112, 160] for MobileNetV3
        self.encoder_stages = self.backbone

        factor = 2 if bilinear else 1
        self.bottleneck = nn.Conv2d(encoder_channels[-1], 1024 // factor, kernel_size=3, padding=1)

        self.up1 = Up(1024 // factor, encoder_channels[-1], encoder_channels[-2] // factor, bilinear)
        self.up2 = Up(encoder_channels[-2] // factor, encoder_channels[-3], encoder_channels[-3] // factor, bilinear)
        self.up3 = Up(encoder_channels[-3] // factor, encoder_channels[-4], encoder_channels[-4], bilinear)
        self.up4 = Up(encoder_channels[-4], encoder_channels[0], encoder_channels[0], bilinear)

        self.outc = OutConv(encoder_channels[0], n_classes)

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.encoder_stages(x)  # Low to high level features
        x_bottleneck = self.bottleneck(x5)

        x = self.up1(x_bottleneck, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)

        logits = self.outc(x)
        logits = F.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=False)
        return logits

    def use_checkpointing(self):
        self.bottleneck = torch.utils.checkpoint(self.bottleneck)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)