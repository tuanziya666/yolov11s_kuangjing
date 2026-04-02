"""DySample upsampling module adapted from the ICCV 2023 implementation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ("DySample",)


class DySample(nn.Module):
    """Dynamic point-sampling upsampler."""

    def __init__(self, in_channels: int, scale: int = 2, style: str = "lp", groups: int = 4, dyscope: bool = False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        if style not in {"lp", "pl"}:
            raise ValueError(f"DySample style must be 'lp' or 'pl', got {style}")
        if style == "pl":
            if in_channels < scale**2 or in_channels % scale**2 != 0:
                raise ValueError("DySample(pl) requires in_channels divisible by scale^2")
        if in_channels < groups or in_channels % groups != 0:
            raise ValueError("DySample requires in_channels divisible by groups")

        if style == "pl":
            inner_channels = in_channels // scale**2
            out_channels = 2 * groups
        else:
            inner_channels = in_channels
            out_channels = 2 * groups * scale**2

        self.offset = nn.Conv2d(inner_channels, out_channels, 1)
        self._normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(inner_channels, out_channels, 1)
            self._constant_init(self.scope, val=0.0)

        self.register_buffer("init_pos", self._init_pos())

    @staticmethod
    def _normal_init(module, mean: float = 0.0, std: float = 1.0, bias: float = 0.0):
        if getattr(module, "weight", None) is not None:
            nn.init.normal_(module.weight, mean, std)
        if getattr(module, "bias", None) is not None:
            nn.init.constant_(module.bias, bias)

    @staticmethod
    def _constant_init(module, val: float, bias: float = 0.0):
        if getattr(module, "weight", None) is not None:
            nn.init.constant_(module.weight, val)
        if getattr(module, "bias", None) is not None:
            nn.init.constant_(module.bias, bias)

    def _init_pos(self) -> torch.Tensor:
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        grid = torch.stack(torch.meshgrid(h, h, indexing="ij")).transpose(1, 2)
        return grid.repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        b, _, h, w = offset.shape
        offset = offset.view(b, 2, -1, h, w)
        coords_h = torch.arange(h, device=x.device, dtype=x.dtype) + 0.5
        coords_w = torch.arange(w, device=x.device, dtype=x.dtype) + 0.5
        # Keep the same layout as the original DySample implementation:
        # stack(meshgrid([w, h])) -> (2, w, h), then transpose -> (2, h, w).
        coords = torch.stack(torch.meshgrid(coords_w, coords_h, indexing="ij")).transpose(1, 2)
        coords = coords.unsqueeze(1).unsqueeze(0)
        normalizer = torch.tensor([w, h], device=x.device, dtype=x.dtype).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(b, -1, h, w), self.scale)
        coords = coords.view(b, 2, -1, self.scale * h, self.scale * w)
        coords = coords.permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        sampled = F.grid_sample(
            x.reshape(b * self.groups, -1, h, w),
            coords,
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        )
        return sampled.reshape(b, -1, self.scale * h, self.scale * w)

    def forward_lp(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "scope"):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x: torch.Tensor) -> torch.Tensor:
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, "scope"):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_pl(x) if self.style == "pl" else self.forward_lp(x)
