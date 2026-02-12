from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.ReLU(inplace=True),
    )


class TinyUNet(nn.Module):
    """
    Very small U-Net:
      input:  [B,1,N,N]
      output: [B,1,N,N] logits (apply sigmoid to get probabilities)
    """

    def __init__(self, base: int = 16):
        super().__init__()
        self.enc1 = conv_block(1, base)           # -> base
        self.enc2 = conv_block(base, base * 2)    # -> 2base
        self.enc3 = conv_block(base * 2, base * 4)# -> 4base

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = conv_block(base * 4, base * 8)

        self.dec3 = conv_block(base * 8 + base * 4, base * 4)
        self.dec2 = conv_block(base * 4 + base * 2, base * 2)
        self.dec1 = conv_block(base * 2 + base, base)

        self.out = nn.Conv2d(base, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)            # [B,base,N,N]
        p1 = self.pool(e1)           # [B,base,N/2,N/2]

        e2 = self.enc2(p1)           # [B,2base,N/2,N/2]
        p2 = self.pool(e2)           # [B,2base,N/4,N/4]

        e3 = self.enc3(p2)           # [B,4base,N/4,N/4]
        p3 = self.pool(e3)           # [B,4base,N/8,N/8]

        b = self.bottleneck(p3)      # [B,8base,N/8,N/8]

        # Decoder (upsample + skip connections)
        u3 = F.interpolate(b, scale_factor=2, mode="nearest")   # -> N/4
        d3 = self.dec3(torch.cat([u3, e3], dim=1))

        u2 = F.interpolate(d3, scale_factor=2, mode="nearest")  # -> N/2
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = F.interpolate(d2, scale_factor=2, mode="nearest")  # -> N
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        logits = self.out(d1)         # [B,1,N,N]
        return logits
