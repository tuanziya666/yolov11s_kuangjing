"""FcaNet-style frequency channel attention modules."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ("MultiSpectralAttentionLayer", "MultiSpectralDCTLayer", "FcaNet")


def get_freq_indices(method):
    """Return canonical frequency indices from the official FcaNet frequency pools."""
    assert method in {
        "top1",
        "top2",
        "top4",
        "top8",
        "top16",
        "top32",
        "bot1",
        "bot2",
        "bot4",
        "bot8",
        "bot16",
        "bot32",
        "low1",
        "low2",
        "low4",
        "low8",
        "low16",
        "low32",
    }
    num_freq = int(method[3:])
    if "top" in method:
        all_top_indices_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2, 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2, 6, 1]
        all_top_indices_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2, 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0, 5, 3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif "low" in method:
        all_low_indices_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4]
        all_low_indices_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    else:
        all_bot_indices_x = [6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4, 6, 2, 5, 6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5, 5, 3, 6]
        all_bot_indices_y = [6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2, 2, 5, 1, 4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5, 3, 3, 3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    return mapper_x, mapper_y


class MultiSpectralDCTLayer(nn.Module):
    """Fixed DCT filter bank used by FcaNet channel attention."""

    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super().__init__()
        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0
        self.register_buffer("weight", self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

    def forward(self, x):
        """Project features onto fixed DCT bases and aggregate responses per channel."""
        return torch.sum(x * self.weight, dim=[2, 3])

    @staticmethod
    def build_filter(pos, freq, size):
        """Build a single 1D DCT basis value."""
        value = math.cos(math.pi * freq * (pos + 0.5) / size) / math.sqrt(size)
        return value if freq == 0 else value * math.sqrt(2)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        """Create the fixed 2D DCT filter bank."""
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)
        c_part = channel // len(mapper_x)
        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part : (i + 1) * c_part, t_x, t_y] = self.build_filter(
                        t_x, u_x, tile_size_x
                    ) * self.build_filter(t_y, v_y, tile_size_y)
        return dct_filter


class MultiSpectralAttentionLayer(nn.Module):
    """Official FcaNet-style multi-spectral channel attention."""

    def __init__(self, channel, dct_h, dct_w, reduction=16, freq_sel_method="top16"):
        super().__init__()
        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
        self.dct_h = dct_h
        self.dct_w = dct_w
        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        hidden = max(channel // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channel, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """Apply frequency-aware channel gating with adaptive pooling for detection compatibility."""
        n, c, h, w = x.shape
        x_pooled = x if (h == self.dct_h and w == self.dct_w) else F.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
        y = self.dct_layer(x_pooled)
        y = self.fc(y).view(n, c, 1, 1)
        return x * y.expand_as(x)


class FcaNet(nn.Module):
    """A thin wrapper so FcaNet can be inserted as a lightweight standalone attention block."""

    def __init__(self, c1, dct_h, dct_w, reduction=16, freq_sel_method="top16"):
        super().__init__()
        self.attn = MultiSpectralAttentionLayer(c1, dct_h, dct_w, reduction=reduction, freq_sel_method=freq_sel_method)

    def forward(self, x):
        """Refine features with multi-spectral channel attention."""
        return self.attn(x)
