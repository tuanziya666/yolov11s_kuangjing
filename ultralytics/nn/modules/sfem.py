"""Spectral-frequency enhancement modules for isolated YOLO11 ablations."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv

__all__ = ("HaarDWT", "HaarIDWT", "SFEM", "SFEMStem")


class HaarDWT(nn.Module):
    """Differentiable 2D Haar wavelet decomposition."""

    def forward(self, x):
        """Split the input into LL, LH, HL, and HH subbands."""
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
        return ll, lh, hl, hh, (h, w)


class HaarIDWT(nn.Module):
    """Differentiable 2D Haar wavelet reconstruction."""

    def forward(self, ll, lh, hl, hh, out_hw):
        """Reconstruct a full-resolution feature map from four subbands."""
        a = (ll + lh + hl + hh) * 0.5
        b = (ll - lh + hl - hh) * 0.5
        c = (ll + lh - hl - hh) * 0.5
        d = (ll - lh - hl + hh) * 0.5

        bsz, ch, h, w = ll.shape
        out = ll.new_zeros((bsz, ch, h * 2, w * 2))
        out[..., 0::2, 0::2] = a
        out[..., 0::2, 1::2] = b
        out[..., 1::2, 0::2] = c
        out[..., 1::2, 1::2] = d
        return out[..., : out_hw[0], : out_hw[1]]


class SFEM(nn.Module):
    """Spatial-frequency enhancement module placed at the YOLO input."""

    def __init__(self, c1):
        """Build a lightweight spatial branch, frequency branch, and gated residual fusion."""
        super().__init__()
        hidden = max(c1 * 2, 8)

        self.spatial = nn.Sequential(
            Conv(c1, hidden, 3, 1),
            Conv(hidden, hidden, 3, 1),
            Conv(hidden, c1, 1, 1, act=False),
        )

        self.dwt = HaarDWT()
        self.idwt = HaarIDWT()
        self.ll_branch = nn.Sequential(
            Conv(c1, hidden, 3, 1),
            Conv(hidden, c1, 1, 1, act=False),
        )
        self.hf_branch = nn.Sequential(
            Conv(c1 * 3, hidden, 3, 1),
            Conv(hidden, c1 * 3, 1, 1, act=False),
        )

        self.gate = nn.Sequential(nn.Conv2d(c1 * 2, c1, 1, 1, bias=True), nn.Sigmoid())

    def forward(self, x):
        """Enhance illumination/detail in spatial and frequency domains, then add back residually."""
        spatial_out = self.spatial(x)

        ll, lh, hl, hh, out_hw = self.dwt(x)
        ll = self.ll_branch(ll)
        hf = self.hf_branch(torch.cat((lh, hl, hh), 1))
        lh, hl, hh = hf.chunk(3, 1)
        freq_out = self.idwt(ll, lh, hl, hh, out_hw)

        fusion = torch.cat((spatial_out, freq_out), 1)
        gate = self.gate(fusion)
        enhanced = gate * (spatial_out + freq_out)
        return x + enhanced


class SFEMStem(nn.Module):
    """Input stem that keeps SFEM at the front while preserving downstream layer numbering."""

    def __init__(self, c1, c2, k=3, s=2):
        """Run SFEM on RGB input, then apply the original stem downsampling convolution."""
        super().__init__()
        self.sfem = SFEM(c1)
        self.conv = Conv(c1, c2, k, s)

    def forward(self, x):
        """Enhance input illumination/noise before the first stem convolution."""
        return self.conv(self.sfem(x))
