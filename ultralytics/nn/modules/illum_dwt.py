"""Lightweight illumination enhancement and DWT downsampling modules for YOLO11 ablations."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv

__all__ = ("AIE", "DWTDown")


class AIE(nn.Module):
    """Adaptive illumination enhancement block for shallow backbone features."""

    def __init__(self, c1, c2, reduction=4, kernel_size=5):
        """Build a residual illumination-aware calibration block."""
        super().__init__()
        if c1 != c2:
            raise ValueError(f"AIE expects equal input/output channels, got c1={c1}, c2={c2}")

        hidden = max(c1 // reduction, 8)
        gate_hidden = max(hidden // 2, 4)
        padding = kernel_size // 2

        self.reduce = Conv(c1, hidden, 1, 1)
        self.detail = Conv(hidden, c1, 1, 1, act=False)
        self.spatial = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size, 1, padding, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, 1, 1, bias=True),
        )
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden, gate_hidden, 1, 1, bias=True),
            nn.SiLU(),
            nn.Conv2d(gate_hidden, c1, 1, 1, bias=True),
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """Enhance low-illumination responses while preserving bright stable regions."""
        feat = self.reduce(x)
        detail = self.detail(feat)

        spatial_map = torch.sigmoid(self.spatial(feat))
        darkness = 1.0 - spatial_map
        channel_gate = torch.sigmoid(self.channel(feat))
        gain = darkness * channel_gate

        return x + self.gamma * detail * gain


class DWTDown(nn.Module):
    """Haar DWT downsampling that preserves low/high-frequency cues before channel fusion."""

    def __init__(self, c1, c2):
        """Fuse four Haar subbands with a 1x1 projection."""
        super().__init__()
        self.fuse = Conv(c1 * 4, c2, 1, 1)

    def forward(self, x):
        """Downsample by Haar decomposition and concatenate four subbands."""
        h, w = x.shape[-2:]
        pad_h = h % 2
        pad_w = w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        a = x[..., 0::2, 0::2]
        b = x[..., 0::2, 1::2]
        c = x[..., 1::2, 0::2]
        d = x[..., 1::2, 1::2]

        ll = (a + b + c + d) * 0.5
        lh = (a - b + c - d) * 0.5
        hl = (a + b - c - d) * 0.5
        hh = (a - b - c + d) * 0.5

        return self.fuse(torch.cat((ll, lh, hl, hh), 1))
