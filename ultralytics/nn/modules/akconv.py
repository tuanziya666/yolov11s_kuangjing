"""AKConv modules for YOLO11 ablations."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .block import Bottleneck, C3k, C3k2

__all__ = ("AKConv", "Bottleneck_AKConv", "C3k_AKConv", "C3k2_AKConv")


class AKConv(nn.Module):
    """Adaptive kernel convolution with learned sampling offsets."""

    def __init__(self, c1, c2, num_param=5, stride=1, bias=None):
        super().__init__()
        self.num_param = num_param
        self.stride = stride
        self.conv = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=(num_param, 1), stride=(num_param, 1), bias=bias),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
        )
        self.p_conv = nn.Conv2d(c1, 2 * num_param, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        nn.init.constant_(self.p_conv.bias, 0)
        self.p_conv.register_full_backward_hook(self._scale_grad)

    @staticmethod
    def _scale_grad(module, grad_input, grad_output):
        scaled_grad_input = tuple(g * 0.1 if g is not None else None for g in grad_input)
        scaled_grad_output = tuple(g * 0.1 if g is not None else None for g in grad_output)
        return scaled_grad_input

    def forward(self, x):
        offset = self.p_conv(x)
        dtype = offset.dtype
        n = offset.size(1) // 2
        p = self._get_p(offset, dtype)

        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat(
            [torch.clamp(q_lt[..., :n], 0, x.size(2) - 1), torch.clamp(q_lt[..., n:], 0, x.size(3) - 1)], dim=-1
        ).long()
        q_rb = torch.cat(
            [torch.clamp(q_rb[..., :n], 0, x.size(2) - 1), torch.clamp(q_rb[..., n:], 0, x.size(3) - 1)], dim=-1
        ).long()
        q_lb = torch.cat([q_lt[..., :n], q_rb[..., n:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :n], q_lt[..., n:]], dim=-1)

        p = torch.cat([torch.clamp(p[..., :n], 0, x.size(2) - 1), torch.clamp(p[..., n:], 0, x.size(3) - 1)], dim=-1)

        g_lt = (1 + (q_lt[..., :n].type_as(p) - p[..., :n])) * (1 + (q_lt[..., n:].type_as(p) - p[..., n:]))
        g_rb = (1 - (q_rb[..., :n].type_as(p) - p[..., :n])) * (1 - (q_rb[..., n:].type_as(p) - p[..., n:]))
        g_lb = (1 + (q_lb[..., :n].type_as(p) - p[..., :n])) * (1 - (q_lb[..., n:].type_as(p) - p[..., n:]))
        g_rt = (1 - (q_rt[..., :n].type_as(p) - p[..., :n])) * (1 + (q_rt[..., n:].type_as(p) - p[..., n:]))

        x_q_lt = self._get_x_q(x, q_lt, n)
        x_q_rb = self._get_x_q(x, q_rb, n)
        x_q_lb = self._get_x_q(x, q_lb, n)
        x_q_rt = self._get_x_q(x, q_rt, n)

        x_offset = (
            g_lt.unsqueeze(1) * x_q_lt
            + g_rb.unsqueeze(1) * x_q_rb
            + g_lb.unsqueeze(1) * x_q_lb
            + g_rt.unsqueeze(1) * x_q_rt
        )

        x_offset = self._reshape_x_offset(x_offset)
        return self.conv(x_offset)

    def _get_p_n(self, n, dtype, device):
        base_int = round(math.sqrt(self.num_param))
        row_number = self.num_param // base_int
        mod_number = self.num_param % base_int

        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(0, row_number, device=device),
            torch.arange(0, base_int, device=device),
            indexing="ij",
        )
        p_n_x = torch.flatten(p_n_x)
        p_n_y = torch.flatten(p_n_y)

        if mod_number > 0:
            mod_p_n_x, mod_p_n_y = torch.meshgrid(
                torch.arange(row_number, row_number + 1, device=device),
                torch.arange(0, mod_number, device=device),
                indexing="ij",
            )
            p_n_x = torch.cat((p_n_x, torch.flatten(mod_p_n_x)))
            p_n_y = torch.cat((p_n_y, torch.flatten(mod_p_n_y)))

        p_n = torch.cat([p_n_x, p_n_y], 0)
        return p_n.view(1, 2 * n, 1, 1).type(dtype)

    def _get_p_0(self, h, w, n, dtype, device):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(0, h * self.stride, self.stride, device=device),
            torch.arange(0, w * self.stride, self.stride, device=device),
            indexing="ij",
        )
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, n, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, n, 1, 1)
        return torch.cat([p_0_x, p_0_y], 1).type(dtype)

    def _get_p(self, offset, dtype):
        n, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)
        device = offset.device
        p_n = self._get_p_n(n, dtype, device)
        p_0 = self._get_p_0(h, w, n, dtype, device)
        return p_0 + p_n + offset

    def _get_x_q(self, x, q, n):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        x = x.contiguous().view(b, c, -1)
        index = q[..., :n] * padded_w + q[..., n:]
        index = index.contiguous().unsqueeze(1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
        return x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, n)

    @staticmethod
    def _reshape_x_offset(x_offset):
        b, c, h, w, n = x_offset.size()
        return x_offset.permute(0, 1, 2, 4, 3).contiguous().view(b, c, h * n, w)


class Bottleneck_AKConv(Bottleneck):
    """Bottleneck block using AKConv in place of standard convolutions."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        if k[0] == 3:
            self.cv1 = AKConv(c1, c2, k[0])
        self.cv2 = AKConv(c2, c2, k[1])


class C3k_AKConv(C3k):
    """C3k block built from AKConv bottlenecks."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck_AKConv(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class C3k2_AKConv(C3k2):
    """C3k2 variant that swaps its inner bottlenecks for AKConv bottlenecks."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(
            C3k_AKConv(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_AKConv(self.c, self.c, shortcut, g)
            for _ in range(n)
        )
