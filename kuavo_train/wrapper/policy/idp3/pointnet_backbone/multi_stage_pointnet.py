import torch
import torch.nn as nn


class MultiStagePointNetEncoder(nn.Module):
    """Default point-cloud backbone used by IDP3.

    Input:
        x: torch.Tensor, shape [B, N, C]
           B: batch size
           N: number of points
           C: point channels (must match `pc_channels`, e.g. xyz=3 / xyzrgb=6)

    Output:
        torch.Tensor, shape [B, out_channels]
        One global feature vector per point cloud.
    """

    def __init__(self, pc_channels=3, h_dim=128, out_channels=128, num_layers=4, **kwargs):
        super().__init__()

        self.pc_channels = pc_channels
        self.h_dim = h_dim
        self.out_channels = out_channels
        self.num_layers = num_layers

        self.act = nn.LeakyReLU(negative_slope=0.0, inplace=False)
        self.conv_in = nn.Conv1d(pc_channels, h_dim, kernel_size=1)

        self.layers = nn.ModuleList()
        self.global_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.layers.append(nn.Conv1d(h_dim, h_dim, kernel_size=1))
            self.global_layers.append(nn.Conv1d(h_dim * 2, h_dim, kernel_size=1))

        self.conv_out = nn.Conv1d(h_dim * self.num_layers, out_channels, kernel_size=1)

    def forward(self, x):
        # x: [B, N, C]
        x = x.transpose(1, 2)  # [B, N, C] -> [B, C, N]
        y = self.act(self.conv_in(x))
        feat_list = []
        for i in range(self.num_layers):
            y = self.act(self.layers[i](y))
            y_global = y.max(-1, keepdim=True).values
            y = torch.cat([y, y_global.expand_as(y)], dim=1)
            y = self.act(self.global_layers[i](y))
            feat_list.append(y)

        x = torch.cat(feat_list, dim=1)
        x = self.conv_out(x)
        # global max pooling over point dimension N -> [B, out_channels]
        return x.max(-1).values
