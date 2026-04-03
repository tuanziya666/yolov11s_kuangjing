"""DECM: Directional-Edge Context Module for lightweight mine detection."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv

__all__ = ("DECM",)


def _dw_block(channels: int, kernel_size, padding):
    return nn.Sequential(
        nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=1, padding=padding, groups=channels, bias=False),
        nn.BatchNorm2d(channels),
        nn.SiLU(inplace=True),
    )


class _ContextBranch(nn.Module):
    """Lightweight pooled context branch."""

    def __init__(self, channels: int, pool_kernel: int = 4):
        super().__init__()
        self.pool_kernel = max(int(pool_kernel), 1)
        self.reduce = Conv(channels, channels, k=1, s=1)
        self.dw = _dw_block(channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        pooled_h = max(h // self.pool_kernel, 1)
        pooled_w = max(w // self.pool_kernel, 1)
        x = F.adaptive_avg_pool2d(x, (pooled_h, pooled_w))
        x = self.reduce(x)
        x = self.dw(x)
        return F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)


class DECM(nn.Module):
    """Directional-Edge Context Module with three lightweight variants.

    Variants:
    - L:   Local branch only
    - LD:  Local + Direction
    - LDC: Local + Direction + Context
    """

    def __init__(self, c1: int, c2: int, variant: str = "LDC", pool_kernel: int = 4):
        super().__init__()
        self.variant = variant.upper()
        if self.variant not in {"L", "LD", "LDC"}:
            raise ValueError(f"DECM variant must be one of L/LD/LDC, got {variant}")

        self.shortcut = Conv(c1, c2, k=1, s=1) if c1 != c2 else nn.Identity()
        hidden = max(c2 // 2, 1)
        hidden = int(math.ceil(hidden / 8) * 8) if hidden >= 8 else hidden

        self.pre = Conv(c1, hidden, k=1, s=1)

        self.local = nn.Sequential(
            _dw_block(hidden, kernel_size=3, padding=1),
            _dw_block(hidden, kernel_size=5, padding=2),
        )

        self.direction = (
            nn.Sequential(
                _dw_block(hidden, kernel_size=(1, 7), padding=(0, 3)),
                _dw_block(hidden, kernel_size=(7, 1), padding=(3, 0)),
            )
            if self.variant in {"LD", "LDC"}
            else None
        )

        self.context = _ContextBranch(hidden, pool_kernel=pool_kernel) if self.variant == "LDC" else None

        branch_count = 1 + int(self.direction is not None) + int(self.context is not None)
        self.fuse = Conv(hidden * branch_count, c2, k=1, s=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x0 = self.pre(x)

        feats = [self.local(x0)]
        if self.direction is not None:
            feats.append(self.direction(x0))
        if self.context is not None:
            feats.append(self.context(x0))

        fused = self.fuse(torch.cat(feats, dim=1))
        return residual + fused
