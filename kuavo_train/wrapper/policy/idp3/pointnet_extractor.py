# Credit: https://github.com/YanjieZe/Improved-3D-Diffusion-Policy

'''
2026.1.23
改动 支持可变点云维度输入
这整个文件中只有IDP3Encoder会被外部调用，如果想改变点云维度，比如x,y,z变为x,y,z,r,g,b
请在 pointcloud_encoder_cfg.in_channels 中设置点云维度
'''

import logging
from typing import Dict, List, Type

import torch
import torch.nn as nn

from kuavo_train.wrapper.policy.idp3.pointnet_backbone import build_pointnet_backbone

logger = logging.getLogger(__name__)

def create_mlp(
    input_dim: int,
    output_dim: int,
    net_arch: List[int],
    activation_fn: Type[nn.Module] = nn.ReLU,
    squash_output: bool = False,
) -> List[nn.Module]:
    # 动态构建一个MLP
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


def meanpool(x, dim=-1, keepdim=False):
    out = x.mean(dim=dim, keepdim=keepdim)
    return out


def maxpool(x, dim=-1, keepdim=False):
    out = x.max(dim=dim, keepdim=keepdim).values
    return out


class StateEncoder(nn.Module):
    # 将环境观察值（Observation）中的“全量状态”（Full State）向量通过一个 MLP（多层感知机）编码成高维特征
    def __init__(self, observation_space: Dict, state_mlp_size=(64, 64), state_mlp_activation_fn=nn.ReLU):
        super().__init__()
        self.state_key = "full_state"
        self.state_shape = observation_space[self.state_key]
        logger.debug(f"[StateEncoder] state shape: {self.state_shape}")

        if len(state_mlp_size) == 0:
            raise RuntimeError("State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        output_dim = state_mlp_size[-1]

        self.state_mlp = nn.Sequential(
            *create_mlp(self.state_shape[0], output_dim, net_arch, state_mlp_activation_fn)
        )

        logger.debug(f"[StateEncoder] output dim: {output_dim}")
        self.output_dim = output_dim

    def output_shape(self):
        return self.output_dim

    def forward(self, observations: Dict) -> torch.Tensor:
        state = observations[self.state_key]
        state_feat = self.state_mlp(state)
        return state_feat # [Batch, output_dim]的特征向量


class IDP3Encoder(nn.Module):  # noqa: N801
    def __init__(
        self,
        observation_space: Dict,
        state_mlp_size=(64, 64),
        state_mlp_activation_fn=nn.ReLU,
        pointcloud_encoder_cfg=None,
        pointnet_type="multi_stage_pointnet",
    ):
        super().__init__()
        self.state_key = "observation.state"
        self.point_cloud_keys = ["observation.pc_h", "observation.pc_l", "observation.pc_r"]

        self.point_cloud_shape = observation_space[self.point_cloud_keys[0]]
        self.state_shape = observation_space[self.state_key]
        self.expected_point_cloud_shape = (
            pointcloud_encoder_cfg.num_points,
            pointcloud_encoder_cfg.in_channels,
        )

        self.num_views = len(self.point_cloud_keys)
        self.encoder_out_channels = pointcloud_encoder_cfg.out_channels
        self.n_output_channels = self.encoder_out_channels * self.num_views

        logger.debug(f"[IDP3Encoder] point cloud shape: {self.point_cloud_shape}")
        logger.debug(f"[IDP3Encoder] state shape: {self.state_shape}")

        self.pointnet_type = pointnet_type

        # Pass all point-cloud encoder config fields to the selected backbone.
        # Keep the IDP3 naming (`in_channels`) and map it to backbone naming (`pc_channels`).
        backbone_kwargs = {}
        if pointcloud_encoder_cfg is not None:
            if hasattr(pointcloud_encoder_cfg, "__dict__"):
                backbone_kwargs = dict(pointcloud_encoder_cfg.__dict__)
            elif isinstance(pointcloud_encoder_cfg, dict):
                backbone_kwargs = dict(pointcloud_encoder_cfg)
        backbone_kwargs.pop("backbone_type", None)
        backbone_kwargs["pc_channels"] = backbone_kwargs.pop("in_channels", pointcloud_encoder_cfg.in_channels)
        backbone_kwargs["out_channels"] = backbone_kwargs.get("out_channels", pointcloud_encoder_cfg.out_channels)

        self.extractor = build_pointnet_backbone(
            backbone_type=self.pointnet_type,
            **backbone_kwargs,
        )

        if len(state_mlp_size) == 0:
            raise RuntimeError("State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        output_dim = state_mlp_size[-1]

        self.n_output_channels += output_dim
        self.state_mlp = nn.Sequential(
            *create_mlp(self.state_shape[0], output_dim, net_arch, state_mlp_activation_fn)
        )

        logger.debug(f"[DP3Encoder] output dim: {self.n_output_channels}")

    def forward(self, observations: Dict) -> torch.Tensor:
        # 处理多视角输入
        pc_list = []
        for key in self.point_cloud_keys:
            points = observations[key]
            # points shape: (Batch, N, C) 或 (Batch*T, N, C) 取決於外部調用
            assert len(points.shape) == 3, f"point cloud shape: {points.shape}, length should be 3"

            if points.shape[1:] != self.expected_point_cloud_shape:
                raise ValueError(
                    f"{key} shape mismatch: got {tuple(points.shape[1:])}, "
                    f"expected {self.expected_point_cloud_shape} from pointcloud_encoder_cfg"
                )
            pc_list.append(points)
        
        # 將多視角數據在 Batch 維度拼接，以實現共享編碼器的並行計算
        # [B, N, C] * 3 -> [B * 3, N, C]
        batch_size = pc_list[0].shape[0]
        all_points = torch.cat(pc_list, dim=0)

        all_feat = self.extractor(all_points)
        pn_feat = all_feat.view(self.num_views, batch_size, -1).permute(1, 0, 2).reshape(batch_size, -1)
        
        state = observations[self.state_key]
        state_feat = self.state_mlp(state)  # B * 64
        final_feat = torch.cat([pn_feat, state_feat], dim=-1)
        return final_feat

    def output_shape(self):
        return self.n_output_channels
