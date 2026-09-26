"""Backbone modules for phase 2 (plan §5.2).

* SimAM      - Yang et al. 2021, parameter-free 3D attention
* CBAM       - Woo et al. 2018 (Ultralytics' implementation is reused)
* CoordAtt   - Hou et al. 2021, coordinate attention
* SPDConv    - Sunkara & Luo 2022, space-to-depth followed by a non-strided conv (replaces strided convs)
* AttnWrap   - runs a block then an attention module, keeping the Ultralytics layer attributes (f, i, type, np)
"""

from __future__ import annotations

import torch
from torch import nn
from ultralytics.nn.modules import CBAM, Conv


class SimAM(nn.Module):
    def __init__(self, c1: int | None = None, e_lambda: float = 1e-4):
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[2] * x.shape[3] - 1
        d = (x - x.mean(dim=(2, 3), keepdim=True)).pow(2)
        y = d / (4 * (d.sum(dim=(2, 3), keepdim=True) / n + self.e_lambda)) + 0.5
        return x * torch.sigmoid(y)


class CoordAtt(nn.Module):
    def __init__(self, c1: int, reduction: int = 32):
        super().__init__()
        mid = max(8, c1 // reduction)
        self.conv1 = nn.Conv2d(c1, mid, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.act = nn.Hardswish()
        self.conv_h = nn.Conv2d(mid, c1, 1)
        self.conv_w = nn.Conv2d(mid, c1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        x_h = x.mean(dim=3, keepdim=True)                      # (b, c, h, 1)
        x_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)  # (b, c, w, 1)
        y = self.act(self.bn1(self.conv1(torch.cat([x_h, x_w], dim=2))))
        y_h, y_w = torch.split(y, [h, w], dim=2)
        a_h = torch.sigmoid(self.conv_h(y_h))
        a_w = torch.sigmoid(self.conv_w(y_w.permute(0, 1, 3, 2)))
        return x * a_h * a_w


class SPDConv(nn.Module):
    """Space-to-depth (scale 2) + stride-1 conv: same output shape as Conv(c1, c2, k, s=2), no information dropped."""

    def __init__(self, c1: int, c2: int, k: int = 3):
        super().__init__()
        self.conv = Conv(4 * c1, c2, k, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[2] % 2 or x.shape[3] % 2:
            x = nn.functional.pad(x, (0, x.shape[3] % 2, 0, x.shape[2] % 2))
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], dim=1)
        return self.conv(x)


class AttnWrap(nn.Module):
    def __init__(self, block: nn.Module, attn: nn.Module):
        super().__init__()
        self.block, self.attn = block, attn

    def forward(self, x):
        return self.attn(self.block(x))


ATTENTION = {"simam": SimAM, "cbam": CBAM, "ca": CoordAtt}
