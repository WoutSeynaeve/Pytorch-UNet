import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, backbone='resnet34', freeze_backbone=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Load a pretrained ResNet model
        if backbone == 'resnet34':
            resnet = models.resnet34(pretrained=True)
            filters = [64, 128, 256, 512]  # Feature sizes of ResNet34
        elif backbone == 'resnet50':
            resnet = models.resnet50(pretrained=True)
            filters = [256, 512, 1024, 2048]  # Feature sizes of ResNet50
        else:
            raise ValueError("Only 'resnet34' and 'resnet50' are supported.")

        if freeze_backbone:
            for param in resnet.parameters():
                param.requires_grad = False  # Freeze ResNet parameters

        # Replace first convolution layer to match input channels
        if n_channels != 3:
            self.encoder0 = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.encoder0 = resnet.conv1

        # Encoder layers from ResNet
        self.encoder1 = nn.Sequential(resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1)  # 64 filters
        self.encoder2 = resnet.layer2  # 128 filters
        self.encoder3 = resnet.layer3  # 256 filters
        self.encoder4 = resnet.layer4  # 512 filters (or 2048 in ResNet50)

        # Bottleneck layer
        factor = 2 if bilinear else 1
        self.bottleneck = nn.Conv2d(filters[-1], 1024 // factor, kernel_size=3, padding=1)

        # Decoder (upsampling path with skip connections)
        self.up1 = Up(1024, filters[-1] // factor, bilinear)
        self.up2 = Up(filters[-1], filters[-2] // factor, bilinear)
        self.up3 = Up(filters[-2], filters[-3] // factor, bilinear)
        self.up4 = Up(filters[-3], filters[-4], bilinear)

        # Output segmentation head
        self.outc = OutConv(filters[-4], n_classes)

    def forward(self, x):
        original_size = x.shape[2:]  # Store original H, W

        # Encoder (ResNet Backbone)
        x0 = self.encoder0(x)  # Initial Conv
        x1 = self.encoder1(x0)  # First ResNet Block
        x2 = self.encoder2(x1)  # Second ResNet Block
        x3 = self.encoder3(x2)  # Third ResNet Block
        x4 = self.encoder4(x3)  # Fourth ResNet Block (Bottleneck)

        # Bottleneck
        x5 = self.bottleneck(x4)

        # Ensure spatial alignment (upsample skip connections if needed)
        def upsample_if_needed(enc_feat, dec_feat):
            if enc_feat.shape[2:] != dec_feat.shape[2:]:
                enc_feat = F.interpolate(enc_feat, size=dec_feat.shape[2:], mode="bilinear", align_corners=False)
            return enc_feat

        # Decoder with corrected skip connections
        x = self.up1(x5, upsample_if_needed(x4, x5))  # Skip from encoder4
        x = self.up2(x, upsample_if_needed(x3, x))   # Skip from encoder3
        x = self.up3(x, upsample_if_needed(x2, x))   # Skip from encoder2
        x = self.up4(x, upsample_if_needed(x1, x))   # Skip from encoder1

        logits = self.outc(x)  # Final output

        # Ensure final output is same spatial size as input
        logits = F.interpolate(logits, size=original_size, mode="bilinear", align_corners=False)

        return logits

    def use_checkpointing(self):
        self.encoder0 = torch.utils.checkpoint(self.encoder0)
        self.encoder1 = torch.utils.checkpoint(self.encoder1)
        self.encoder2 = torch.utils.checkpoint(self.encoder2)
        self.encoder3 = torch.utils.checkpoint(self.encoder3)
        self.encoder4 = torch.utils.checkpoint(self.encoder4)
        self.bottleneck = torch.utils.checkpoint(self.bottleneck)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
