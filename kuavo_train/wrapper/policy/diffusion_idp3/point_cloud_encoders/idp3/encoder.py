"""Multi-view IDP3 point cloud encoder for diffusion_idp3."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from kuavo_train.wrapper.policy.diffusion_idp3.point_cloud_encoders.idp3.multi_stage_pointnet import (
    MultiStagePointNetEncoder,
)


def uniform_sampling_torch(point_cloud: torch.Tensor, num_points: int) -> torch.Tensor:
    """Uniformly sample or pad point clouds to `num_points`."""
    batch_size, num_input_points, point_dim = point_cloud.shape
    device = point_cloud.device

    if num_input_points == num_points:
        return point_cloud

    if num_input_points > num_points:
        indices = torch.randperm(num_input_points, device=device)[:num_points]
        return point_cloud[:, indices]

    num_pad = num_points - num_input_points
    pad = torch.zeros(batch_size, num_pad, point_dim, device=device, dtype=point_cloud.dtype)
    padded = torch.cat([point_cloud, pad], dim=1)
    perm = torch.randperm(num_points, device=device)
    return padded[:, perm]


class IDP3MultiViewPointCloudEncoder(nn.Module):
    """Encode each point cloud view independently with shared IDP3 PointNet."""

    def __init__(
        self,
        num_views: int,
        num_points: int,
        in_channels: int,
        out_channels: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        downsample: bool = True,
    ) -> None:
        super().__init__()
        self.num_views = num_views
        self.num_points = num_points
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample

        self.extractor = MultiStagePointNetEncoder(
            pc_channels=in_channels,
            h_dim=hidden_dim,
            out_channels=out_channels,
            num_layers=num_layers,
        )

    def _flatten_time(self, points: torch.Tensor) -> torch.Tensor:
        if points.dim() == 4:
            batch_size, obs_steps, num_points, point_dim = points.shape
            return points.reshape(batch_size * obs_steps, num_points, point_dim)
        if points.dim() == 3:
            return points
        raise ValueError(f"Point cloud tensor must be 3D or 4D, got {tuple(points.shape)}")

    def _align_points(self, points: torch.Tensor) -> torch.Tensor:
        _, num_points, point_dim = points.shape

        if point_dim < self.in_channels:
            raise ValueError(f"Point cloud channels {point_dim} < expected {self.in_channels}")
        if point_dim > self.in_channels:
            points = points[:, :, : self.in_channels]

        if self.downsample:
            points = uniform_sampling_torch(points, self.num_points)
        elif num_points != self.num_points:
            raise ValueError(
                f"Point cloud has {num_points} points, expected {self.num_points} when downsample=False"
            )
        return points

    def forward(self, point_clouds: Sequence[torch.Tensor]) -> torch.Tensor:
        """Args: list of [B,S,N,C] or [B,N,C]. Returns: [B*S, V, D]."""
        if len(point_clouds) != self.num_views:
            raise ValueError(f"Expected {self.num_views} views, got {len(point_clouds)}")

        view_list = []
        for points in point_clouds:
            flat = self._flatten_time(points)
            aligned = self._align_points(flat)
            view_list.append(aligned)

        stacked = torch.cat(view_list, dim=0)
        feats = self.extractor(stacked)

        batch_obs = view_list[0].shape[0]
        feats = feats.reshape(self.num_views, batch_obs, -1).permute(1, 0, 2)
        return feats
