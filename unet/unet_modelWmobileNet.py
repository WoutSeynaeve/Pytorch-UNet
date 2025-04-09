from .unet_parts import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Use MobileNetV3 from torchvision as backbone
        backbone = models.mobilenet_v3_large(pretrained=True)
        if n_channels != 3:
            backbone.features[0][0] = nn.Conv2d(n_channels, 16, kernel_size=3, stride=2, padding=1, bias=False)

        self.encoder_stages = nn.ModuleList([
            nn.Sequential(backbone.features[0]),   # x1: 16
            nn.Sequential(backbone.features[1], backbone.features[2]),   # x2: 24
            nn.Sequential(backbone.features[3], backbone.features[4], backbone.features[5]),   # x3: 40
            nn.Sequential(*backbone.features[6:12]),  # x4: 112
            nn.Sequential(*backbone.features[12:]),   # x5: 160
        ])

        encoder_channels = [16, 24, 40, 112, 160]
        factor = 2 if bilinear else 1
        self.bottleneck = nn.Conv2d(encoder_channels[-1], 1024 // factor, kernel_size=3, padding=1)

        self.up1 = Up(1024 // factor, encoder_channels[-1], bilinear)
        self.up2 = Up(encoder_channels[-1], encoder_channels[-2], bilinear)
        self.up3 = Up(encoder_channels[-2], encoder_channels[-3], bilinear)
        self.up4 = Up(encoder_channels[-3], encoder_channels[0], bilinear)

        self.outc = OutConv(encoder_channels[0], n_classes)

    def forward(self, x):
        x1 = self.encoder_stages[0](x)
        x2 = self.encoder_stages[1](x1)
        x3 = self.encoder_stages[2](x2)
        x4 = self.encoder_stages[3](x3)
        x5 = self.encoder_stages[4](x4)

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
