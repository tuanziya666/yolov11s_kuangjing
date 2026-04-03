"""CARAFE upsampling module adapted for the local YOLO11s experiments."""

from __future__ import annotations

import torch
import torch.nn as nn

from .conv import Conv

__all__ = ("CARAFE",)


class CARAFE(nn.Module):
    """Content-aware reassembly of features for lightweight upsampling."""

    def __init__(self, c: int, k_enc: int = 3, k_up: int = 5, c_mid: int = 64, scale: int = 2):
        super().__init__()
        self.scale = scale
        self.comp = Conv(c, c_mid)
        self.enc = Conv(c_mid, (scale * k_up) ** 2, k=k_enc, act=False)
        self.pix_shf = nn.PixelShuffle(scale)
        self.upsmp = nn.Upsample(scale_factor=scale, mode="nearest")
        self.unfold = nn.Unfold(kernel_size=k_up, dilation=scale, padding=k_up // 2 * scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        h_up, w_up = h * self.scale, w * self.scale

        weights = self.comp(x)
        weights = self.enc(weights)
        weights = self.pix_shf(weights)
        weights = torch.softmax(weights, dim=1)

        x = self.upsmp(x)
        x = self.unfold(x)
        x = x.view(b, c, -1, h_up, w_up)
        x = torch.einsum("bkhw,bckhw->bchw", [weights, x])
        return x
