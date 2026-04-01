import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv


class HaarWavelet(nn.Module):
    def __init__(self, in_channels, grad=False):
        super().__init__()
        self.in_channels = in_channels

        haar_weights = torch.ones(4, 1, 2, 2)
        haar_weights[1, 0, 0, 1] = -1
        haar_weights[1, 0, 1, 1] = -1
        haar_weights[2, 0, 1, 0] = -1
        haar_weights[2, 0, 1, 1] = -1
        haar_weights[3, 0, 1, 0] = -1
        haar_weights[3, 0, 0, 1] = -1

        haar_weights = torch.cat([haar_weights] * self.in_channels, 0)
        if grad:
            self.haar_weights = nn.Parameter(haar_weights, requires_grad=True)
        else:
            self.register_buffer("haar_weights", haar_weights)

    def forward(self, x, rev=False):
        if not rev:
            out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.in_channels) / 4.0
            out = out.reshape([x.shape[0], self.in_channels, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = torch.transpose(out, 1, 2)
            return out.reshape([x.shape[0], self.in_channels * 4, x.shape[2] // 2, x.shape[3] // 2])

        out = x.reshape([x.shape[0], 4, self.in_channels, x.shape[2], x.shape[3]])
        out = torch.transpose(out, 1, 2)
        out = out.reshape([x.shape[0], self.in_channels * 4, x.shape[2], x.shape[3]])
        return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups=self.in_channels)


class WFU(nn.Module):
    """Wavelet Feature Upsampling module ported as a minimal standalone block."""

    def __init__(self, chn):
        super().__init__()
        dim_big, dim_small = chn
        self.dim = dim_big
        self.haar_wavelet = HaarWavelet(dim_big, grad=False)
        self.inverse_haar_wavelet = HaarWavelet(dim_big, grad=False)
        self.refine = nn.Sequential(
            Conv(dim_big, dim_big, 3),
            nn.Conv2d(dim_big, dim_big, kernel_size=3, padding=1),
        )
        self.channel_transformation = nn.Sequential(
            Conv(dim_big + dim_small, dim_big + dim_small, 1),
            nn.Conv2d(dim_big + dim_small, dim_big * 3, kernel_size=1, padding=0),
        )

    def forward(self, x):
        x_big, x_small = x
        haar = self.haar_wavelet(x_big, rev=False)
        a = haar.narrow(1, 0, self.dim)
        h = haar.narrow(1, self.dim, self.dim)
        v = haar.narrow(1, self.dim * 2, self.dim)
        d = haar.narrow(1, self.dim * 3, self.dim)

        hvd = self.refine(h + v + d)
        a_ = self.channel_transformation(torch.cat([x_small, a], dim=1))
        return self.inverse_haar_wavelet(torch.cat([hvd, a_], dim=1), rev=True)
