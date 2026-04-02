"""MaSA modules adapted from the official RMT implementation for YOLO11 ablations."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv

__all__ = ("MaSA",)


def rotate_every_two(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def theta_shift(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rotate_every_two(x) * sin)


class DWConv2dNHWC(nn.Module):
    """Depthwise conv that accepts NHWC tensors and returns NHWC tensors."""

    def __init__(self, dim: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv(x)
        return x.permute(0, 2, 3, 1).contiguous()


class RetNetRelPos2d(nn.Module):
    """2D relative-position builder with Manhattan-distance decay."""

    def __init__(self, embed_dim: int, num_heads: int, initial_value: float = 2.0, heads_range: float = 4.0):
        super().__init__()
        key_dim = embed_dim // num_heads
        if key_dim % 2 != 0:
            raise ValueError(f"MaSA requires an even head dimension, got embed_dim={embed_dim}, num_heads={num_heads}")
        angle = 1.0 / (10000 ** torch.linspace(0, 1, key_dim // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        decay = torch.log(1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads, dtype=torch.float32) / num_heads))
        self.register_buffer("angle", angle)
        self.register_buffer("decay", decay)

    def generate_2d_decay(self, h: int, w: int) -> torch.Tensor:
        index_h = torch.arange(h, device=self.decay.device, dtype=self.decay.dtype)
        index_w = torch.arange(w, device=self.decay.device, dtype=self.decay.dtype)
        grid = torch.meshgrid(index_h, index_w, indexing="ij")
        grid = torch.stack(grid, dim=-1).reshape(h * w, 2)
        mask = grid[:, None, :] - grid[None, :, :]
        mask = mask.abs().sum(dim=-1)
        return mask * self.decay[:, None, None]

    def generate_1d_decay(self, length: int) -> torch.Tensor:
        index = torch.arange(length, device=self.decay.device, dtype=self.decay.dtype)
        mask = (index[:, None] - index[None, :]).abs()
        return mask * self.decay[:, None, None]

    def forward(self, spatial_shape: tuple[int, int], chunkwise_recurrent: bool = True):
        h, w = spatial_shape
        index = torch.arange(h * w, device=self.decay.device, dtype=self.decay.dtype)
        sin = torch.sin(index[:, None] * self.angle[None, :]).reshape(h, w, -1)
        cos = torch.cos(index[:, None] * self.angle[None, :]).reshape(h, w, -1)

        if chunkwise_recurrent:
            mask_h = self.generate_1d_decay(h)
            mask_w = self.generate_1d_decay(w)
            return (sin, cos), (mask_h, mask_w)

        mask = self.generate_2d_decay(h, w)
        return (sin, cos), mask


class VisionRetentionChunk(nn.Module):
    """Chunkwise MaSA from the official RMT block, adapted to NHWC feature maps."""

    def __init__(self, embed_dim: int, num_heads: int, value_factor: int = 1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.value_factor = value_factor
        self.key_dim = embed_dim // num_heads
        self.value_dim = embed_dim * value_factor // num_heads
        self.scaling = self.key_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * value_factor, bias=True)
        self.lepe = DWConv2dNHWC(embed_dim * value_factor, 5, 1, 2)
        self.out_proj = nn.Linear(embed_dim * value_factor, embed_dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x: torch.Tensor, rel_pos):
        b, h, w, _ = x.shape
        (sin, cos), (mask_h, mask_w) = rel_pos
        sin = sin.to(device=x.device, dtype=x.dtype)
        cos = cos.to(device=x.device, dtype=x.dtype)
        mask_h = mask_h.to(device=x.device, dtype=x.dtype)
        mask_w = mask_w.to(device=x.device, dtype=x.dtype)

        q = self.q_proj(x)
        k = self.k_proj(x) * self.scaling
        v = self.v_proj(x)
        lepe = self.lepe(v)

        q = q.view(b, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)
        k = k.view(b, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)
        qr = theta_shift(q, sin, cos)
        kr = theta_shift(k, sin, cos)

        qr_w = qr.transpose(1, 2)
        kr_w = kr.transpose(1, 2)
        v = v.view(b, h, w, self.num_heads, self.value_dim).permute(0, 1, 3, 2, 4)

        qk_mat_w = qr_w @ kr_w.transpose(-1, -2)
        qk_mat_w = torch.softmax(qk_mat_w + mask_w, dim=-1)
        v = torch.matmul(qk_mat_w, v)

        qr_h = qr.permute(0, 3, 1, 2, 4)
        kr_h = kr.permute(0, 3, 1, 2, 4)
        v = v.permute(0, 3, 2, 1, 4)

        qk_mat_h = qr_h @ kr_h.transpose(-1, -2)
        qk_mat_h = torch.softmax(qk_mat_h + mask_h, dim=-1)
        out = torch.matmul(qk_mat_h, v)

        out = out.permute(0, 3, 1, 2, 4).flatten(-2, -1)
        out = out + lepe
        return self.out_proj(out)


class FeedForwardNetwork(nn.Module):
    """RMT-style FFN with a depthwise conv residual branch."""

    def __init__(self, embed_dim: int, ffn_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.dwconv = DWConv2dNHWC(ffn_dim, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        residual = x
        x = self.dwconv(x)
        x = x + residual
        return self.fc2(x)


class MaSA(nn.Module):
    """Lightweight Manhattan Self-Attention block for YOLO feature maps."""

    def __init__(
        self,
        c1: int,
        c2: int,
        num_heads: int = 8,
        ffn_ratio: float = 2.0,
        initial_value: float = 2.0,
        heads_range: float = 4.0,
        chunkwise_recurrent: bool = True,
        layerscale: bool = False,
        layer_init_value: float = 1e-5,
    ):
        super().__init__()
        if c2 % num_heads != 0:
            raise ValueError(f"MaSA requires channels divisible by num_heads, got c2={c2}, num_heads={num_heads}")
        if (c2 // num_heads) % 2 != 0:
            raise ValueError(
                f"MaSA requires an even per-head dimension, got c2={c2}, num_heads={num_heads}, head_dim={c2 // num_heads}"
            )

        self.chunkwise_recurrent = chunkwise_recurrent
        self.in_proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()
        self.rel_pos = RetNetRelPos2d(c2, num_heads, initial_value=initial_value, heads_range=heads_range)
        self.norm1 = nn.LayerNorm(c2, eps=1e-6)
        self.attn = VisionRetentionChunk(c2, num_heads)
        self.norm2 = nn.LayerNorm(c2, eps=1e-6)
        self.ffn = FeedForwardNetwork(c2, int(c2 * ffn_ratio))
        self.pos = DWConv2dNHWC(c2, 3, 1, 1)
        self.layerscale = layerscale

        if layerscale:
            self.gamma1 = nn.Parameter(layer_init_value * torch.ones(1, 1, 1, c2), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_init_value * torch.ones(1, 1, 1, c2), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        h, w = x.shape[1:3]
        rel_pos = self.rel_pos((h, w), chunkwise_recurrent=self.chunkwise_recurrent)

        x = x + self.pos(x)
        if self.layerscale:
            x = x + self.gamma1 * self.attn(self.norm1(x), rel_pos)
            x = x + self.gamma2 * self.ffn(self.norm2(x))
        else:
            x = x + self.attn(self.norm1(x), rel_pos)
            x = x + self.ffn(self.norm2(x))
        return x.permute(0, 3, 1, 2).contiguous()
