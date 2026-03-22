# Ultralytics YOLO 🚀, AGPL-3.0 license
"""EMA attention module."""

import math

import torch
import torch.nn as nn

__all__ = ("EMA",)


class EMA(nn.Module):
    """
    Efficient Multi-Scale Attention.

    This module keeps the channel count unchanged so it can be inserted as a standalone
    layer in YAML model definitions, e.g. after SPPF in YOLO11.
    """

    def __init__(self, c1, c2=None, factor=32):
        """Initialize EMA with equal input/output channels and grouped spatial attention."""
        super().__init__()
        c2 = c1 if c2 is None else c2
        if c1 != c2:
            raise ValueError(f"EMA expects equal input/output channels, but got c1={c1}, c2={c2}.")

        self.groups = math.gcd(c1, factor) or 1
        self.group_channels = c1 // self.groups
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.pool_hw = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(self.group_channels, self.group_channels, 1, 1, 0)
        self.conv3 = nn.Conv2d(self.group_channels, self.group_channels, 3, 1, 1)
        self.gn = nn.GroupNorm(self.group_channels, self.group_channels)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """Apply EMA attention while preserving the input tensor shape."""
        b, c, h, w = x.shape
        gx = x.reshape(b * self.groups, self.group_channels, h, w)
        x_h = self.pool_h(gx)
        x_w = self.pool_w(gx).permute(0, 1, 3, 2)

        hw = self.conv1(torch.cat((x_h, x_w), dim=2))
        x_h, x_w = torch.split(hw, (h, w), dim=2)
        x1 = self.gn(gx * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3(gx)

        x11 = self.softmax(self.pool_hw(x1).flatten(1).unsqueeze(1))
        x12 = x2.reshape(b * self.groups, self.group_channels, -1)
        x21 = self.softmax(self.pool_hw(x2).flatten(1).unsqueeze(1))
        x22 = x1.reshape(b * self.groups, self.group_channels, -1)

        weights = (torch.bmm(x11, x12) + torch.bmm(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (gx * weights.sigmoid()).reshape(b, c, h, w)
