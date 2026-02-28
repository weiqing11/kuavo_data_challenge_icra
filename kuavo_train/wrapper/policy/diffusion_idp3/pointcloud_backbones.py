from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Sequence

import torch
from torch import Tensor, nn

from kuavo_train.wrapper.policy.idp3.pointnet_extractor import MultiStagePointNetEncoder


class BasePointCloudBackbone(nn.Module, ABC):
    """Backbone interface that converts point clouds to token sequences."""

    def __init__(
        self,
        point_cloud_keys: Sequence[str],
        num_points: int,
        in_channels: int,
        out_channels: int,
        downsample: bool,
    ) -> None:
        super().__init__()
        if len(point_cloud_keys) == 0:
            raise ValueError("`point_cloud_keys` must not be empty.")

        self.point_cloud_keys = list(point_cloud_keys)
        self.num_points = num_points
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample

    @property
    def token_count(self) -> int:
        return len(self.point_cloud_keys)

    @property
    def output_dim(self) -> int:
        return self.out_channels

    @abstractmethod
    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        """
        Returns:
            Tensor: point-cloud tokens with shape `(B*S, T_pc, D_pc_raw)`.
        """


class IDP3MultiStagePointNetBackbone(BasePointCloudBackbone):
    """
    IDP3-style point-cloud encoder using MultiStagePointNet.

    It encodes each configured point-cloud view into one token, then stacks
    all views as the token dimension.
    """

    def __init__(
        self,
        point_cloud_keys: Sequence[str],
        num_points: int,
        in_channels: int,
        out_channels: int,
        downsample: bool = False,
    ) -> None:
        super().__init__(
            point_cloud_keys=point_cloud_keys,
            num_points=num_points,
            in_channels=in_channels,
            out_channels=out_channels,
            downsample=downsample,
        )
        self.extractor = MultiStagePointNetEncoder(
            pc_channels=in_channels,
            out_channels=out_channels,
        )

    def _normalize_single_point_cloud(self, point_cloud: Tensor, key: str) -> Tensor:
        if point_cloud.dim() != 4:
            raise ValueError(
                f"`{key}` expects shape `(B, S, N, C)`, got {tuple(point_cloud.shape)}."
            )

        batch_size, n_obs_steps, current_points, current_channels = point_cloud.shape

        if current_points > self.num_points:
            if self.downsample:
                indices = torch.randperm(current_points, device=point_cloud.device)[: self.num_points]
            else:
                indices = torch.arange(self.num_points, device=point_cloud.device)
            point_cloud = point_cloud[:, :, indices, :]
        elif current_points < self.num_points:
            raise ValueError(
                f"`{key}` has {current_points} points but requires at least {self.num_points}."
            )

        if current_channels < self.in_channels:
            raise ValueError(
                f"`{key}` has {current_channels} channels but requires at least {self.in_channels}."
            )
        if current_channels > self.in_channels:
            point_cloud = point_cloud[:, :, :, : self.in_channels]

        return point_cloud.reshape(batch_size * n_obs_steps, self.num_points, self.in_channels)

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        per_view_points: list[Tensor] = []
        for key in self.point_cloud_keys:
            if key not in batch:
                raise KeyError(
                    f"Missing point-cloud key `{key}` in batch. Available keys: {sorted(batch.keys())}"
                )
            per_view_points.append(self._normalize_single_point_cloud(batch[key], key))

        # Shared backbone forward over all views in one pass for efficiency.
        all_points = torch.cat(per_view_points, dim=0)  # (T_pc*B*S, N, C)
        all_feats = self.extractor(all_points)  # (T_pc*B*S, D_raw)

        bs_steps = per_view_points[0].shape[0]
        tokens = all_feats.view(self.token_count, bs_steps, self.out_channels)
        return tokens.permute(1, 0, 2).contiguous()  # (B*S, T_pc, D_raw)


class PointCloudTokenProjector(nn.Module):
    """Project point-cloud tokens to the diffusion conditioning dimension."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        return self.projector(tokens)


POINT_CLOUD_BACKBONES: dict[str, Callable[..., BasePointCloudBackbone]] = {}


def register_point_cloud_backbone(name: str):
    def decorator(cls):
        POINT_CLOUD_BACKBONES[name] = cls
        return cls

    return decorator


register_point_cloud_backbone("idp3_multi_stage_pointnet")(IDP3MultiStagePointNetBackbone)


def build_point_cloud_backbone(
    name: str,
    point_cloud_keys: Sequence[str],
    num_points: int,
    in_channels: int,
    out_channels: int,
    downsample: bool = False,
) -> BasePointCloudBackbone:
    if name not in POINT_CLOUD_BACKBONES:
        raise ValueError(
            f"Unknown point-cloud backbone `{name}`. "
            f"Available backbones: {sorted(POINT_CLOUD_BACKBONES.keys())}"
        )
    return POINT_CLOUD_BACKBONES[name](
        point_cloud_keys=point_cloud_keys,
        num_points=num_points,
        in_channels=in_channels,
        out_channels=out_channels,
        downsample=downsample,
    )
