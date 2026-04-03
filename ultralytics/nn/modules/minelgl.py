"""MineLGL: a lightweight local-global-local detection block."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv

__all__ = ("MineLGL",)


class _LocalBranch(nn.Module):
    """Lightweight depthwise local feature extractor."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=5, stride=1, padding=2, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=(1, 7), stride=1, padding=(0, 3), groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=(7, 1), stride=1, padding=(3, 0), groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _GlobalBranch(nn.Module):
    """Pooled lightweight self-attention branch."""

    def __init__(self, channels: int, num_heads: int = 4, pool_ratio: int = 4):
        super().__init__()
        inner_channels = max(channels // 2, num_heads)
        if inner_channels % num_heads != 0:
            inner_channels = math.ceil(inner_channels / num_heads) * num_heads

        self.num_heads = num_heads
        self.pool_ratio = max(int(pool_ratio), 1)
        self.inner_channels = inner_channels
        self.head_dim = inner_channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.reduce = Conv(channels, inner_channels, k=1, s=1)
        self.qkv = nn.Conv2d(inner_channels, inner_channels * 3, kernel_size=1, stride=1, padding=0, bias=False)
        self.proj = Conv(inner_channels, channels, k=1, s=1)

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        out_h = max(h // self.pool_ratio, 1)
        out_w = max(w // self.pool_ratio, 1)
        return F.adaptive_avg_pool2d(x, (out_h, out_w))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        pooled = self._pool(x)
        pooled = self.reduce(pooled)

        b, c, hp, wp = pooled.shape
        n = hp * wp
        qkv = self.qkv(pooled).reshape(b, 3, self.num_heads, self.head_dim, n)
        q, k, v = qkv.unbind(dim=1)

        q = q.permute(0, 1, 3, 2).contiguous()  # B, heads, N, dim
        k = k.permute(0, 1, 3, 2).contiguous()
        v = v.permute(0, 1, 3, 2).contiguous()

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)

        out = out.permute(0, 1, 3, 2).contiguous().reshape(b, c, hp, wp)
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
        return self.proj(out)


class MineLGL(nn.Module):
    """A lightweight Local-Global-Local block for mine detection."""

    def __init__(
        self,
        c1: int,
        c2: int,
        num_heads: int = 4,
        pool_ratio: int = 4,
        gate: bool = False,
    ):
        super().__init__()
        self.proj_in = Conv(c1, c2, k=1, s=1) if c1 != c2 else nn.Identity()
        self.local_branch = _LocalBranch(c2)
        self.global_branch = _GlobalBranch(c2, num_heads=num_heads, pool_ratio=pool_ratio)
        self.fuse = Conv(c2 * 3, c2, k=1, s=1)
        self.use_gate = gate
        self.gate = nn.Sequential(nn.Conv2d(c2, c2, kernel_size=1, stride=1, padding=0, bias=True), nn.Sigmoid()) if gate else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj_in(x)
        local_feat = self.local_branch(residual)
        global_feat = self.global_branch(residual)
        fused = self.fuse(torch.cat((residual, local_feat, global_feat), dim=1))
        if self.use_gate:
            fused = fused * self.gate(fused)
        return residual + fused
