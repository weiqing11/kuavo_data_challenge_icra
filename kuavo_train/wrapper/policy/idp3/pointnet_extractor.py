# Credit: https://github.com/YanjieZe/Improved-3D-Diffusion-Policy

'''
2026.1.23
改动 支持可变点云维度输入
这整个文件中只有IDP3Encoder会被外部调用，如果想改变点云维度，比如x,y,z变为x,y,z,r,g,b
请在实例化IDP3Encoder时指定pc_channels，这个参数代表点云维度
'''

import logging
from typing import Dict, List, Type

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def shuffle_point_numpy(point_cloud):
    # 随机打乱点云顺序
    B, N, C = point_cloud.shape
    indices = np.random.permutation(N)
    return point_cloud[:, indices]


def pad_point_numpy(point_cloud, num_points):
    # 如果点的数量小于目标数量，在原点进行零填充
    B, N, C = point_cloud.shape
    if num_points > N:
        num_pad = num_points - N
        pad_points = np.zeros((B, num_pad, C))
        point_cloud = np.concatenate([point_cloud, pad_points], axis=1)
        point_cloud = shuffle_point_numpy(point_cloud)
    return point_cloud


def uniform_sampling_numpy(point_cloud, num_points):
    # 点数对齐，如果不够就填充，过多则随机降采样
    B, N, C = point_cloud.shape
    # padd if num_points > N
    if num_points > N:
        return pad_point_numpy(point_cloud, num_points)

    # random sampling
    indices = np.random.permutation(N)[:num_points]
    sampled_points = point_cloud[:, indices]
    return sampled_points


def shuffle_point_torch(point_cloud):
    # 随机打乱点云顺序
    B, N, C = point_cloud.shape
    indices = torch.randperm(N)
    return point_cloud[:, indices]


def pad_point_torch(point_cloud, num_points):
    # 填充点数量
    B, N, C = point_cloud.shape
    device = point_cloud.device
    if num_points > N:
        num_pad = num_points - N
        pad_points = torch.zeros(B, num_pad, C).to(device)
        point_cloud = torch.cat([point_cloud, pad_points], dim=1)
        point_cloud = shuffle_point_torch(point_cloud)
    return point_cloud


def uniform_sampling_torch(point_cloud, num_points):
    # 对齐点云数量
    B, N, C = point_cloud.shape
    device = point_cloud.device
    # padd if num_points > N
    if num_points == N:
        return point_cloud
    if num_points > N:
        return pad_point_torch(point_cloud, num_points)

    # random sampling
    indices = torch.randperm(N)[:num_points]
    sampled_points = point_cloud[:, indices]
    return sampled_points


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


class MultiStagePointNetEncoder(nn.Module):
    def __init__(self, pc_channels=3, h_dim=128, out_channels=128, num_layers=4, **kwargs):
        super().__init__()

        self.pc_channels = pc_channels # 点云中每个点的维度，如果是6，则代表为x,y,z,r,g,b
        self.h_dim = h_dim # 隐藏层维度
        self.out_channels = out_channels # 输出维度
        self.num_layers = num_layers # 深度

        self.act = nn.LeakyReLU(negative_slope=0.0, inplace=False)

        self.conv_in = nn.Conv1d(pc_channels, h_dim, kernel_size=1) # 初始映射

        # self.layers存储局部特征，global存储全局特征
        self.layers, self.global_layers = nn.ModuleList(), nn.ModuleList()

        for i in range(self.num_layers):
            self.layers.append(nn.Conv1d(h_dim, h_dim, kernel_size=1))
            self.global_layers.append(nn.Conv1d(h_dim * 2, h_dim, kernel_size=1))

        self.conv_out = nn.Conv1d(h_dim * self.num_layers, out_channels, kernel_size=1)

    def forward(self, x):
        x = x.transpose(1, 2)  # [B, N, 3] --> [B, 3, N] (以3维x,y,z为例)
        y = self.act(self.conv_in(x))
        feat_list = []
        for i in range(self.num_layers):
            # 局部特征变换
            y = self.act(self.layers[i](y))
            # 提取全局特征
            y_global = y.max(-1, keepdim=True).values
            # 将全局特征复制 N 份，拼接到每个点的局部特征后面
            y = torch.cat([y, y_global.expand_as(y)], dim=1)
            # 将拼接后的双倍维度压回 h_dim，并存储到列表中
            y = self.act(self.global_layers[i](y))
            feat_list.append(y)
        # cat all features
        x = torch.cat(feat_list, dim=1)
        x = self.conv_out(x)

        x_global = x.max(-1).values

        return x_global


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
        pc_channels = 3,
        use_pc_color=False,
        pointnet_type="dp3_encoder",
        point_downsample=True,
    ):
        super().__init__()
        self.state_key = "observation.state"
        self.point_cloud_keys = ["observation.pc_h", "observation.pc_l", "observation.pc_r"]

        self.point_cloud_shape = observation_space[self.point_cloud_keys[0]]
        self.state_shape = observation_space[self.state_key]

        self.num_views = len(self.point_cloud_keys)
        self.encoder_out_channels = pointcloud_encoder_cfg.out_channels
        self.n_output_channels = self.encoder_out_channels * self.num_views

        self.num_points = pointcloud_encoder_cfg.num_points  # 4096

        logger.debug(f"[IDP3Encoder] point cloud shape: {self.point_cloud_shape}")
        logger.debug(f"[IDP3Encoder] state shape: {self.state_shape}")

        self.pc_channels = pc_channels
        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type

        self.downsample = point_downsample
        if self.downsample:
            self.point_preprocess = uniform_sampling_torch
        else:
            self.point_preprocess = nn.Identity()

        if pointnet_type == "multi_stage_pointnet":
            self.extractor = MultiStagePointNetEncoder(pc_channels=self.pc_channels, 
                                                       out_channels=pointcloud_encoder_cfg.out_channels)
        else:
            raise NotImplementedError(f"pointnet_type: {pointnet_type}")

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
            
            if self.downsample:
                points = self.point_preprocess(points, self.num_points)
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
