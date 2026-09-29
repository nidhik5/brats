"""3D U-Net (same topology as the original Keras notebook, ported to PyTorch).

Encoder 16-32-64-128-256 filters, two 3x3x3 convs per level with dropout
between them, 2x2x2 max-pool down, 2x2x2 transposed conv up, skip
concatenation, 1x1x1 conv to class logits. `norm="instance"` adds
InstanceNorm after each conv (not in the original); `norm="none"` reproduces
the original architecture.
"""
import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, dropout, norm):
        super().__init__()
        layers = []
        for i, c in enumerate((cin, cout)):
            layers.append(nn.Conv3d(c, cout, 3, padding=1, bias=norm == "none"))
            if norm == "instance":
                layers.append(nn.InstanceNorm3d(cout, affine=True))
            layers.append(nn.ReLU(inplace=True))
            if i == 0:
                layers.append(nn.Dropout3d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels=3, num_classes=4, base_filters=16, norm="instance"):
        super().__init__()
        f = [base_filters * 2**i for i in range(5)]
        drop = [0.1, 0.1, 0.2, 0.2, 0.3]
        self.enc = nn.ModuleList()
        c = in_channels
        for fi, d in zip(f, drop):
            self.enc.append(ConvBlock(c, fi, d, norm))
            c = fi
        self.pool = nn.MaxPool3d(2)
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in (3, 2, 1, 0):
            self.up.append(nn.ConvTranspose3d(f[i + 1], f[i], 2, stride=2))
            self.dec.append(ConvBlock(2 * f[i], f[i], drop[i], norm))
        self.head = nn.Conv3d(f[0], num_classes, 1)
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")  # he_uniform
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        skips = []
        for i, block in enumerate(self.enc):
            x = block(x)
            if i < 4:
                skips.append(x)
                x = self.pool(x)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            x = dec(torch.cat([up(x), skip], dim=1))
        return self.head(x)  # logits; softmax lives in the loss / at inference
