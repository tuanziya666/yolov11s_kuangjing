"""Minimal HSPAN/HSFPN helpers for the HSPAN-DySample experiment."""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ("ChannelAttention_HSFPN", "Multiply", "Add")


class ChannelAttention_HSFPN(nn.Module):
    """Channel attention block used by the HSPAN/HSFPN neck variants."""

    def __init__(self, in_planes: int, ratio: int = 4, flag: bool = True):
        super().__init__()
        hidden = max(in_planes // ratio, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv1 = nn.Conv2d(in_planes, hidden, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden, in_planes, 1, bias=False)
        self.flag = flag
        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.conv2(self.relu(self.conv1(self.avg_pool(x))))
        max_out = self.conv2(self.relu(self.conv1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        return out * x if self.flag else out


class Multiply(nn.Module):
    """Elementwise multiply for two feature maps."""

    def forward(self, x: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return x[0] * x[1]


class Add(nn.Module):
    """Elementwise add for two feature maps."""

    def forward(self, x: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return x[0] + x[1]
