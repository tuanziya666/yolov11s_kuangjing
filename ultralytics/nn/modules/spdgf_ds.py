"""Hybrid SPD-Conv, GFNet, and DSConv modules for isolated YOLO11 ablations."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import C3k2
from .conv import Conv, DWConv

__all__ = (
    "SpaceToDepth",
    "SPDConv",
    "GlobalFilter",
    "GFBlock",
    "SPDGFDown",
    "DSConv",
    "C3k2DSConv",
)


class SpaceToDepth(nn.Module):
    """Rearrange spatial detail into channel space before convolution."""

    def __init__(self, block_size=2):
        """Initialize the space-to-depth factor."""
        super().__init__()
        self.block_size = block_size

    def forward(self, x):
        """Convert spatial neighborhoods into channel groups."""
        bs = self.block_size
        return torch.cat([x[..., i::bs, j::bs] for i in range(bs) for j in range(bs)], 1)


class SPDConv(nn.Module):
    """SPD-Conv downsampling block that avoids strided convolutions."""

    def __init__(self, c1, c2, k=3, block_size=2, act=True):
        """Initialize space-to-depth followed by a stride-1 convolution."""
        super().__init__()
        self.spd = SpaceToDepth(block_size)
        self.conv = Conv(c1 * block_size * block_size, c2, k, 1, act=act)

    def forward(self, x):
        """Preserve fine detail during downsampling."""
        return self.conv(self.spd(x))


class GlobalFilter(nn.Module):
    """A lightweight GFNet-style global filter in the Fourier domain."""

    def __init__(self, dim, h=16, w=9):
        """Initialize learnable complex filters and interpolate them per feature-map size."""
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(dim, h, w, 2) * 0.02)

    def forward(self, x):
        """Apply global frequency filtering and project back to the spatial domain."""
        dtype = x.dtype
        b, c, h, w = x.shape
        x_fft = torch.fft.rfft2(x.float(), dim=(2, 3), norm="ortho")
        target_size = (h, w // 2 + 1)
        weight = self.complex_weight.permute(0, 3, 1, 2).reshape(1, c * 2, *self.complex_weight.shape[1:3])
        weight = F.interpolate(weight, size=target_size, mode="bilinear", align_corners=False)
        weight = weight.reshape(c, 2, *target_size).permute(0, 2, 3, 1).contiguous()
        weight = torch.view_as_complex(weight)
        x = torch.fft.irfft2(x_fft * weight.unsqueeze(0), s=(h, w), dim=(2, 3), norm="ortho")
        return x.to(dtype=dtype)


class GFBlock(nn.Module):
    """GFNet-style residual block with frequency filtering and channel mixing."""

    def __init__(self, c):
        """Initialize the spectral filter and lightweight feed-forward network."""
        super().__init__()
        hidden = max(c * 2, 32)
        self.norm1 = nn.BatchNorm2d(c)
        self.filter = GlobalFilter(c)
        self.norm2 = nn.BatchNorm2d(c)
        self.ffn = nn.Sequential(Conv(c, hidden, 1, 1), Conv(hidden, c, 1, 1, act=False))

    def forward(self, x):
        """Refine features with global frequency filtering."""
        x = x + self.filter(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class SPDGFDown(nn.Module):
    """Backbone downsampling block with SPD-Conv and a parallel GFNet branch."""

    def __init__(self, c1, c2, k=3):
        """Initialize the detail-preserving path and the frequency-denoising path."""
        super().__init__()
        self.spd_path = SPDConv(c1, c2, k=k)
        self.gf_path = nn.Sequential(Conv(c1, c2, 1, 1), nn.AvgPool2d(2, 2), GFBlock(c2))
        self.fuse = Conv(c2 * 2, c2, 1, 1)

    def forward(self, x):
        """Fuse spatial-detail and frequency-domain branches."""
        return self.fuse(torch.cat((self.spd_path(x), self.gf_path(x)), 1))


class DSConv(nn.Module):
    """Directional separable convolution for elongated target morphology."""

    def __init__(self, c):
        """Initialize dynamic directional depthwise branches and a residual fuse."""
        super().__init__()
        hidden = max(c // 4, 16)
        self.pre = Conv(c, c, 1, 1)
        self.square = DWConv(c, c, 3, 1)
        self.horizontal = Conv(c, c, (1, 9), 1, g=c)
        self.vertical = Conv(c, c, (9, 1), 1, g=c)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, hidden, 1, 1),
            nn.SiLU(),
            nn.Conv2d(hidden, 3, 1, 1),
        )
        self.fuse = Conv(c, c, 1, 1)

    def forward(self, x):
        """Adaptively emphasize square, horizontal, and vertical responses."""
        x_in = self.pre(x)
        branches = torch.stack((self.square(x_in), self.horizontal(x_in), self.vertical(x_in)), 1)
        weights = self.gate(x_in).softmax(1).unsqueeze(2)
        out = (branches * weights).sum(1)
        return self.fuse(out) + x


class C3k2DSConv(nn.Module):
    """Use a standard C3k2 fusion block followed by directional DSConv refinement."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Keep the C3k2 interface so the module can be dropped into YAML safely."""
        super().__init__()
        self.c3k2 = C3k2(c1, c2, n=n, c3k=c3k, e=e, g=g, shortcut=shortcut)
        self.dsconv = DSConv(c2)

    def forward(self, x):
        """Refine fused neck features for elongated targets like drill pipes."""
        return self.dsconv(self.c3k2(x))
