import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv


class LSCCM(nn.Module):
    """Lightweight Spatial-Channel Correlation Module."""

    def __init__(self, c1, c2, reduction=4, pool_kernel=4):
        super().__init__()
        hidden = max(c2 // max(int(reduction), 1), 1)
        self.pool_kernel = max(int(pool_kernel), 1)

        self.reduce = Conv(c1, hidden, k=1, s=1)

        self.spatial_q = nn.Conv2d(hidden, hidden, kernel_size=1, bias=False)
        self.spatial_k = nn.Conv2d(hidden, hidden, kernel_size=1, bias=False)
        self.spatial_v = nn.Conv2d(hidden, hidden, kernel_size=1, bias=False)

        self.channel_proj = nn.Conv2d(hidden, hidden, kernel_size=1, bias=False)
        self.restore = Conv(hidden * 2, c2, k=1, s=1, act=False)
        self.shortcut = nn.Identity() if c1 == c2 else Conv(c1, c2, k=1, s=1, act=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def _spatial_branch(self, x):
        b, c, h, w = x.shape
        q = self.spatial_q(x).flatten(2).transpose(1, 2)  # B, HW, C

        pool = min(self.pool_kernel, h, w)
        pooled = F.avg_pool2d(x, kernel_size=pool, stride=pool, ceil_mode=True) if pool > 1 else x
        k = self.spatial_k(pooled).flatten(2)  # B, C, HWp
        v = self.spatial_v(pooled).flatten(2).transpose(1, 2)  # B, HWp, C

        attn = torch.matmul(q, k) / math.sqrt(max(c, 1))
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(b, c, h, w)
        return out

    def _channel_branch(self, x):
        b, c, h, w = x.shape
        feat = self.channel_proj(x).flatten(2)  # B, C, HW
        affinity = torch.matmul(feat, feat.transpose(1, 2)) / math.sqrt(max(h * w, 1))
        affinity = affinity.softmax(dim=-1)
        out = torch.matmul(affinity, feat).reshape(b, c, h, w)
        return out

    def forward(self, x):
        xr = self.reduce(x)
        spatial = self._spatial_branch(xr)
        channel = self._channel_branch(xr)
        fused = self.restore(torch.cat((spatial, channel), dim=1))
        return self.shortcut(x) + self.gamma * fused
