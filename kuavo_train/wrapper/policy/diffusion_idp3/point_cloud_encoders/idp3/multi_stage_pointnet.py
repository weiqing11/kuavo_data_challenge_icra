"""Multi-stage PointNet backbone copied from IDP3 for local use."""

from __future__ import annotations

import torch
import torch.nn as nn


class MultiStagePointNetEncoder(nn.Module):
    """Encode a point cloud [B, N, C] into a global feature [B, out_channels]."""

    def __init__(self, pc_channels: int = 3, h_dim: int = 128, out_channels: int = 128, num_layers: int = 4) -> None:
        super().__init__()

        self.act = nn.LeakyReLU(negative_slope=0.0, inplace=False)
        self.conv_in = nn.Conv1d(pc_channels, h_dim, kernel_size=1)

        self.layers = nn.ModuleList()
        self.global_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.Conv1d(h_dim, h_dim, kernel_size=1))
            self.global_layers.append(nn.Conv1d(h_dim * 2, h_dim, kernel_size=1))

        self.conv_out = nn.Conv1d(h_dim * num_layers, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        y = self.act(self.conv_in(x))

        feat_list = []
        for layer, global_layer in zip(self.layers, self.global_layers, strict=True):
            y = self.act(layer(y))
            y_global = y.max(-1, keepdim=True).values
            y = torch.cat([y, y_global.expand_as(y)], dim=1)
            y = self.act(global_layer(y))
            feat_list.append(y)

        y = torch.cat(feat_list, dim=1)
        y = self.conv_out(y)
        return y.max(-1).values
