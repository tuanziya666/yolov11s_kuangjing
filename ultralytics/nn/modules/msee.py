"""Multi-scale edge enhancement modules for YOLO11 ablations."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import C3k2
from .conv import Conv

__all__ = ("EdgeEnhancer", "MSEE", "C3k2MSEE")


class EdgeEnhancer(nn.Module):
    """Lightweight edge emphasis block inspired by the paper's EE module."""

    def __init__(self, c):
        """Initialize an edge enhancement block that reweights high-frequency responses."""
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.edge_conv = Conv(c, c, 3, 1)
        self.weight = nn.Sequential(nn.Conv2d(c, c, 1, bias=True), nn.Sigmoid())

    def forward(self, x):
        """Enhance residual edge responses and add them back to the input feature."""
        edge = x - self.pool(x)
        edge = self.edge_conv(edge)
        return x + edge * self.weight(edge)


class MSEE(nn.Module):
    """Paper-inspired multi-scale edge enhancement module."""

    def __init__(self, c, pool_sizes=(3, 6, 9, 12), branch_channels=None):
        """Build parallel pooled branches, edge enhancement, and final feature fusion."""
        super().__init__()
        branch_channels = branch_channels or max(c // 4, 16)
        self.local = Conv(c, branch_channels, 1, 1)
        self.branches = nn.ModuleList(
            nn.ModuleDict(
                {
                    "pool": nn.AdaptiveAvgPool2d((size, size)),
                    "proj": nn.Sequential(Conv(c, branch_channels, 1, 1), Conv(branch_channels, branch_channels, 3, 1)),
                    "edge": EdgeEnhancer(branch_channels),
                }
            )
            for size in pool_sizes
        )
        self.fuse = Conv(branch_channels * (len(pool_sizes) + 1), c, 1, 1)

    def forward(self, x):
        """Fuse local detail with pooled multi-scale edge-enhanced context."""
        size = x.shape[-2:]
        feats = [self.local(x)]
        for branch in self.branches:
            pooled = branch["pool"](x)
            pooled = branch["proj"](pooled)
            pooled = F.interpolate(pooled, size=size, mode="bilinear", align_corners=False)
            feats.append(branch["edge"](pooled))
        return self.fuse(F.relu(torch.cat(feats, 1), inplace=False)) + x


class C3k2MSEE(nn.Module):
    """Map the paper's C2f+MSEE idea onto YOLO11's C3k2 backbone block."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True, pool_sizes=(3, 6, 9, 12)):
        """Apply a standard C3k2 block followed by MSEE without changing the YAML interface."""
        super().__init__()
        self.c3k2 = C3k2(c1, c2, n=n, c3k=c3k, e=e, g=g, shortcut=shortcut)
        self.msee = MSEE(c2, pool_sizes=pool_sizes)

    def forward(self, x):
        """Run backbone feature extraction first, then refine with multi-scale edge enhancement."""
        return self.msee(self.c3k2(x))
