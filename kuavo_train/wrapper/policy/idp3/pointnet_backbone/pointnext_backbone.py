"""Standalone PointNeXt-style backbone for IDP3.

This file vendors a minimal PointNeXt encoder path in pure PyTorch, without
depending on `tmp/PointNeXt` or custom CUDA extensions from OpenPoints.

Backbone interface contract (for IDP3):
- Input:  [B, N, C]
- Output: [B, F]
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence, Type

import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


class LayerNorm2d(nn.LayerNorm):
    def __init__(self, num_channels: int, **kwargs):
        super().__init__(num_channels, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x.permute(0, 2, 3, 1),
            self.normalized_shape,
            self.weight,
            self.bias,
            self.eps,
        ).permute(0, 3, 1, 2).contiguous()


class LayerNorm1d(nn.LayerNorm):
    def __init__(self, num_channels: int, **kwargs):
        super().__init__(num_channels, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x.permute(0, 2, 1),
            self.normalized_shape,
            self.weight,
            self.bias,
            self.eps,
        ).permute(0, 2, 1).contiguous()


_ACT_LAYER = {
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "leakyrelu": nn.LeakyReLU,
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "swish": nn.SiLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}

_NORM_LAYER = {
    "bn1d": nn.BatchNorm1d,
    "bn2d": nn.BatchNorm2d,
    "syncbn": nn.SyncBatchNorm,
    "ln1d": LayerNorm1d,
    "ln2d": LayerNorm2d,
}


def create_act(act_args: Dict[str, Any] | str | None) -> nn.Module | None:
    if act_args is None:
        return None
    if isinstance(act_args, str):
        act_args = {"act": act_args}
    act_args = copy.deepcopy(act_args)
    act_name = act_args.pop("act", None)
    if act_name is None:
        return None
    act_name = act_name.lower()
    if act_name not in _ACT_LAYER:
        raise ValueError(f"Unsupported activation: {act_name}")
    act_cls = _ACT_LAYER[act_name]
    inplace = act_args.pop("inplace", True)
    if act_name in {"gelu", "sigmoid", "tanh"}:
        return act_cls(**act_args)
    return act_cls(inplace=inplace, **act_args)


def create_norm(norm_args: Dict[str, Any] | str | None, channels: int, dimension: str | None = None):
    if norm_args is None:
        return None

    if isinstance(norm_args, str):
        norm_name = norm_args
        kwargs = {}
    else:
        cfg = copy.deepcopy(norm_args)
        norm_name = cfg.pop("norm", None)
        kwargs = cfg

    if norm_name is None:
        return None

    norm_name = str(norm_name).lower()
    if norm_name == "bn":
        dim = str(dimension or "1d").lower()
        if "2" in dim:
            norm_name = "bn2d"
        else:
            norm_name = "bn1d"
    elif norm_name == "ln":
        dim = str(dimension or "1d").lower()
        if "2" in dim:
            norm_name = "ln2d"
        else:
            norm_name = "ln1d"
    elif dimension is not None and norm_name in {"bn1d", "bn2d", "ln1d", "ln2d"}:
        pass

    if norm_name not in _NORM_LAYER:
        raise ValueError(f"Unsupported normalization: {norm_name}")
    return _NORM_LAYER[norm_name](channels, **kwargs)


class Conv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        if len(args) == 2 and "kernel_size" not in kwargs:
            super().__init__(*args, kernel_size=(1, 1), **kwargs)
        else:
            super().__init__(*args, **kwargs)


class Conv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        if len(args) == 2 and "kernel_size" not in kwargs:
            super().__init__(*args, kernel_size=1, **kwargs)
        else:
            super().__init__(*args, **kwargs)


def create_convblock2d(*args, norm_args=None, act_args=None, order: str = "conv-norm-act", **kwargs):
    in_channels = args[0]
    out_channels = args[1]
    bias = kwargs.pop("bias", True)

    if order == "conv-norm-act":
        norm_layer = create_norm(norm_args, out_channels, dimension="2d")
        bias = False if norm_layer is not None else bias
        layers: List[nn.Module] = [Conv2d(*args, bias=bias, **kwargs)]
        if norm_layer is not None:
            layers.append(norm_layer)
        act_layer = create_act(act_args)
        if act_layer is not None:
            layers.append(act_layer)
    elif order == "norm-act-conv":
        layers = []
        norm_layer = create_norm(norm_args, in_channels, dimension="2d")
        bias = False if norm_layer is not None else bias
        if norm_layer is not None:
            layers.append(norm_layer)
        act_layer = create_act(act_args)
        if act_layer is not None:
            layers.append(act_layer)
        layers.append(Conv2d(*args, bias=bias, **kwargs))
    elif order == "conv-act-norm":
        norm_layer = create_norm(norm_args, out_channels, dimension="2d")
        bias = False if norm_layer is not None else bias
        layers = [Conv2d(*args, bias=bias, **kwargs)]
        act_layer = create_act(act_args)
        if act_layer is not None:
            layers.append(act_layer)
        if norm_layer is not None:
            layers.append(norm_layer)
    else:
        raise NotImplementedError(f"Unsupported conv2d order: {order}")

    return nn.Sequential(*layers)


def create_convblock1d(*args, norm_args=None, act_args=None, order: str = "conv-norm-act", **kwargs):
    in_channels = args[0]
    out_channels = args[1]
    bias = kwargs.pop("bias", True)

    if order == "conv-norm-act":
        norm_layer = create_norm(norm_args, out_channels, dimension="1d")
        bias = False if norm_layer is not None else bias
        layers: List[nn.Module] = [Conv1d(*args, bias=bias, **kwargs)]
        if norm_layer is not None:
            layers.append(norm_layer)
        act_layer = create_act(act_args)
        if act_layer is not None:
            layers.append(act_layer)
    elif order == "norm-act-conv":
        layers = []
        norm_layer = create_norm(norm_args, in_channels, dimension="1d")
        bias = False if norm_layer is not None else bias
        if norm_layer is not None:
            layers.append(norm_layer)
        act_layer = create_act(act_args)
        if act_layer is not None:
            layers.append(act_layer)
        layers.append(Conv1d(*args, bias=bias, **kwargs))
    elif order == "conv-act-norm":
        norm_layer = create_norm(norm_args, out_channels, dimension="1d")
        bias = False if norm_layer is not None else bias
        layers = [Conv1d(*args, bias=bias, **kwargs)]
        act_layer = create_act(act_args)
        if act_layer is not None:
            layers.append(act_layer)
        if norm_layer is not None:
            layers.append(norm_layer)
    else:
        raise NotImplementedError(f"Unsupported conv1d order: {order}")

    return nn.Sequential(*layers)


def _index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather points/features by index.

    Args:
        points: [B, N, C]
        idx:    [B, S] or [B, S, K]

    Returns:
        Gathered points with shape [B, S, C] or [B, S, K, C].
    """
    bsz, _, channels = points.shape
    idx_shape = idx.shape
    idx_flat = idx.reshape(bsz, -1).long()
    gathered = torch.gather(points, 1, idx_flat.unsqueeze(-1).expand(-1, -1, channels))
    return gathered.reshape(*idx_shape, channels)


def _torch_grouping_operation(features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """features: [B, C, N], idx: [B, S, K] -> [B, C, S, K]."""
    features_t = features.transpose(1, 2).contiguous()
    grouped = _index_points(features_t, idx)  # [B, S, K, C]
    return grouped.permute(0, 3, 1, 2).contiguous()

@torch.no_grad()
def furthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Pure PyTorch FPS fallback.

    xyz: [B, N, 3]
    returns idx: [B, npoint]
    """
    bsz, n_total, _ = xyz.shape
    npoint = min(max(1, npoint), n_total)
    if npoint == n_total:
        return torch.arange(n_total, device=xyz.device).view(1, -1).repeat(bsz, 1)

    centroids = torch.zeros((bsz, npoint), dtype=torch.long, device=xyz.device)
    distance = torch.full((bsz, n_total), 1e10, dtype=xyz.dtype, device=xyz.device)
    farthest = torch.randint(0, n_total, (bsz,), dtype=torch.long, device=xyz.device)
    batch_indices = torch.arange(bsz, dtype=torch.long, device=xyz.device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(bsz, 1, 3)
        dist = ((xyz - centroid) ** 2).sum(-1)
        distance = torch.minimum(distance, dist)
        farthest = distance.max(-1)[1]
    return centroids


@torch.no_grad()
def _pairwise_topk_indices(
    query_xyz: torch.Tensor,
    support_xyz: torch.Tensor,
    k: int,
    query_chunk_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k nearest neighbors for each query point.

    Returns:
        dists: [B, M, K]
        idx:   [B, M, K]
    """
    bsz, m, _ = query_xyz.shape
    n = support_xyz.shape[1]
    k = min(max(1, k), n)

    dists_chunks: List[torch.Tensor] = []
    idx_chunks: List[torch.Tensor] = []
    for start in range(0, m, query_chunk_size):
        end = min(start + query_chunk_size, m)
        q = query_xyz[:, start:end, :].float()
        s = support_xyz.float()
        pair_dist = torch.cdist(q, s)  # [B, m_chunk, N]
        topk = torch.topk(pair_dist, k=k, dim=-1, largest=False, sorted=False)
        dists_chunks.append(topk.values)
        idx_chunks.append(topk.indices)

    dists = torch.cat(dists_chunks, dim=1)
    idx = torch.cat(idx_chunks, dim=1)
    return dists, idx


def ball_query_indices(
    query_xyz: torch.Tensor,
    support_xyz: torch.Tensor,
    radius: float | None,
    nsample: int,
    query_chunk_size: int = 256,
) -> torch.Tensor:
    dists, idx = _pairwise_topk_indices(query_xyz, support_xyz, k=nsample, query_chunk_size=query_chunk_size)
    if radius is None:
        return idx.long()
    mask = dists <= float(radius)
    first_idx = idx[..., :1].expand_as(idx)
    idx = torch.where(mask, idx, first_idx)
    return idx.long()


class GroupAll(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query_xyz: torch.Tensor, support_xyz: torch.Tensor, features: torch.Tensor | None = None):
        del query_xyz
        grouped_xyz = support_xyz.transpose(1, 2).unsqueeze(2)  # [B, 3, 1, N]
        grouped_features = features.unsqueeze(2) if features is not None else None
        return grouped_xyz, grouped_features


class QueryAndGroup(nn.Module):
    def __init__(
        self,
        radius: float,
        nsample: int,
        relative_xyz: bool = True,
        normalize_dp: bool = False,
        return_only_idx: bool = False,
        query_chunk_size: int = 256,
        **kwargs,
    ):
        super().__init__()
        del kwargs
        self.radius = radius
        self.nsample = nsample
        self.relative_xyz = relative_xyz
        self.normalize_dp = normalize_dp
        self.return_only_idx = return_only_idx
        self.query_chunk_size = query_chunk_size

    def forward(
        self,
        query_xyz: torch.Tensor,
        support_xyz: torch.Tensor,
        features: torch.Tensor | None = None,
    ):
        idx = ball_query_indices(
            query_xyz,
            support_xyz,
            radius=self.radius,
            nsample=self.nsample,
            query_chunk_size=self.query_chunk_size,
        )
        if self.return_only_idx:
            return idx

        grouped_xyz = _index_points(support_xyz, idx).permute(0, 3, 1, 2).contiguous()  # [B, 3, M, K]
        if self.relative_xyz:
            grouped_xyz = grouped_xyz - query_xyz.transpose(1, 2).unsqueeze(-1)
        if self.normalize_dp:
            grouped_xyz = grouped_xyz / max(float(self.radius), 1e-6)

        grouped_features = _torch_grouping_operation(features, idx) if features is not None else None
        return grouped_xyz, grouped_features


def create_grouper(group_args: Dict[str, Any], query_chunk_size: int = 256):
    cfg = copy.deepcopy(group_args)
    method = str(cfg.pop("NAME", "ballquery")).lower()
    radius = cfg.pop("radius", 0.1)
    nsample = cfg.pop("nsample", 32)

    if nsample is None:
        return GroupAll()

    if method == "ballquery":
        return QueryAndGroup(radius=radius, nsample=nsample, query_chunk_size=query_chunk_size, **cfg)
    raise ValueError(f"Unsupported group method: {method}. Only 'ballquery' is supported.")


def get_aggregation_features(
    p: torch.Tensor,
    dp: torch.Tensor,
    f: torch.Tensor | None,
    fj: torch.Tensor,
    feature_type: str = "dp_fj",
) -> torch.Tensor:
    if feature_type == "dp_fj":
        return torch.cat([dp, fj], dim=1)
    if feature_type == "dp_fj_df":
        if f is None:
            raise ValueError("feature_type 'dp_fj_df' requires input features f.")
        df = fj - f.unsqueeze(-1)
        return torch.cat([dp, fj, df], dim=1)
    if feature_type == "pi_dp_fj_df":
        if f is None:
            raise ValueError("feature_type 'pi_dp_fj_df' requires input features f.")
        df = fj - f.unsqueeze(-1)
        pi = p.transpose(1, 2).unsqueeze(-1).expand(-1, -1, -1, df.shape[-1])
        return torch.cat([pi, dp, fj, df], dim=1)
    if feature_type == "dp_df":
        if f is None:
            raise ValueError("feature_type 'dp_df' requires input features f.")
        df = fj - f.unsqueeze(-1)
        return torch.cat([dp, df], dim=1)
    raise ValueError(f"Unsupported feature_type: {feature_type}")


CHANNEL_MAP = {
    "fj": lambda x: x,
    "df": lambda x: x,
    "assa": lambda x: x * 3,
    "assa_dp": lambda x: x * 3 + 3,
    "dp_fj": lambda x: 3 + x,
    "pj": lambda x: x,
    "dp": lambda x: 3,
    "pi_dp": lambda x: x + 3,
    "pj_dp": lambda x: x + 3,
    "dp_fj_df": lambda x: x * 2 + 3,
    "dp_fi_df": lambda x: x * 2 + 3,
    "pi_dp_fj_df": lambda x: x * 2 + 6,
    "pj_dp_fj_df": lambda x: x * 2 + 6,
    "pj_dp_df": lambda x: x + 6,
    "dp_df": lambda x: x + 3,
}


def get_reduction_fn(reduction: str):
    reduction = "mean" if reduction.lower() == "avg" else reduction.lower()
    if reduction == "max":
        return lambda x: torch.max(x, dim=-1, keepdim=False)[0]
    if reduction == "mean":
        return lambda x: torch.mean(x, dim=-1, keepdim=False)
    if reduction == "sum":
        return lambda x: torch.sum(x, dim=-1, keepdim=False)
    raise ValueError(f"Unsupported reduction: {reduction}")


class LocalAggregation(nn.Module):
    def __init__(
        self,
        channels: List[int],
        norm_args: Dict[str, Any] | None = None,
        act_args: Dict[str, Any] | None = None,
        group_args: Dict[str, Any] | None = None,
        conv_args: Dict[str, Any] | None = None,
        feature_type: str = "dp_fj",
        reduction: str = "max",
        last_act: bool = True,
        query_chunk_size: int = 256,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            logger.warning("Unused kwargs in LocalAggregation: %s", kwargs)

        norm_args = norm_args or {"norm": "bn1d"}
        act_args = act_args or {"act": "relu"}
        group_args = group_args or {"NAME": "ballquery", "nsample": 32}
        conv_args = conv_args or {}

        channels = list(channels)
        channels[0] = CHANNEL_MAP[feature_type](channels[0])
        convs: List[nn.Module] = []
        for i in range(len(channels) - 1):
            convs.append(
                create_convblock2d(
                    channels[i],
                    channels[i + 1],
                    norm_args=norm_args,
                    act_args=None if i == (len(channels) - 2) and not last_act else act_args,
                    **conv_args,
                )
            )

        self.convs = nn.Sequential(*convs)
        self.grouper = create_grouper(group_args, query_chunk_size=query_chunk_size)
        self.pool = get_reduction_fn(reduction)
        self.feature_type = feature_type

    def forward(self, pf):
        p, f = pf
        dp, fj = self.grouper(p, p, f)
        fj = get_aggregation_features(p, dp, f, fj, self.feature_type)
        return self.pool(self.convs(fj))


class SetAbstraction(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        layers: int = 1,
        stride: int = 1,
        group_args: Dict[str, Any] | None = None,
        norm_args: Dict[str, Any] | None = None,
        act_args: Dict[str, Any] | None = None,
        conv_args: Dict[str, Any] | None = None,
        sampler: str = "fps",
        feature_type: str = "dp_fj",
        use_res: bool = False,
        is_head: bool = False,
        query_chunk_size: int = 256,
        **kwargs,
    ):
        super().__init__()
        del kwargs

        norm_args = norm_args or {"norm": "bn"}
        act_args = act_args or {"act": "relu"}
        group_args = copy.deepcopy(group_args or {"NAME": "ballquery", "nsample": 32})
        conv_args = conv_args or {}

        self.stride = stride
        self.is_head = is_head
        self.all_aggr = (not is_head) and stride == 1
        self.use_res = use_res and (not self.all_aggr) and (not self.is_head)
        self.feature_type = feature_type

        mid_channel = out_channels // 2 if stride > 1 else out_channels
        channels = [in_channels] + [mid_channel] * (layers - 1) + [out_channels]
        channels[0] = in_channels if is_head else CHANNEL_MAP[feature_type](channels[0])

        if self.use_res:
            if in_channels != channels[-1]:
                self.skipconv = create_convblock1d(in_channels, channels[-1], norm_args=None, act_args=None)
            else:
                self.skipconv = nn.Identity()
            self.act = create_act(act_args)

        create_conv = create_convblock1d if is_head else create_convblock2d
        convs: List[nn.Module] = []
        for i in range(len(channels) - 1):
            convs.append(
                create_conv(
                    channels[i],
                    channels[i + 1],
                    norm_args=norm_args if not is_head else None,
                    act_args=None if i == len(channels) - 2 and (self.use_res or is_head) else act_args,
                    **conv_args,
                )
            )
        self.convs = nn.Sequential(*convs)

        if not is_head:
            if self.all_aggr:
                group_args["nsample"] = None
                group_args["radius"] = None
            self.grouper = create_grouper(group_args, query_chunk_size=query_chunk_size)
            self.pool = lambda x: torch.max(x, dim=-1, keepdim=False)[0]

            sampler = sampler.lower()
            if sampler == "fps":
                self.sample_fn = furthest_point_sample
            else:
                raise ValueError(f"Unsupported sampler: {sampler}. Only 'fps' is supported.")

    def forward(self, pf):
        p, f = pf
        if self.is_head:
            f = self.convs(f)
            return p, f

        if not self.all_aggr:
            new_n = max(1, p.shape[1] // self.stride)
            idx = self.sample_fn(p, new_n).long()
            new_p = _index_points(p, idx)
        else:
            idx = None
            new_p = p

        if self.use_res or "df" in self.feature_type:
            if idx is None:
                fi = f
            else:
                fi = torch.gather(f, -1, idx.unsqueeze(1).expand(-1, f.shape[1], -1))
            if self.use_res:
                identity = self.skipconv(fi)
        else:
            fi = None
            identity = None

        dp, fj = self.grouper(new_p, p, f)
        fj = get_aggregation_features(new_p, dp, fi, fj, feature_type=self.feature_type)
        f = self.pool(self.convs(fj))

        if self.use_res:
            f = self.act(f + identity)
        return new_p, f


class InvResMLP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        norm_args: Dict[str, Any] | None = None,
        act_args: Dict[str, Any] | None = None,
        aggr_args: Dict[str, Any] | None = None,
        group_args: Dict[str, Any] | None = None,
        conv_args: Dict[str, Any] | None = None,
        expansion: int = 1,
        use_res: bool = True,
        num_posconvs: int = 2,
        less_act: bool = False,
        query_chunk_size: int = 256,
        **kwargs,
    ):
        super().__init__()
        self.use_res = use_res
        norm_args = norm_args or {"norm": "bn"}
        act_args = act_args or {"act": "relu"}
        aggr_args = aggr_args or {"feature_type": "dp_fj", "reduction": "max"}
        group_args = group_args or {"NAME": "ballquery", "nsample": 32}
        conv_args = conv_args or {}

        mid_channels = int(in_channels * expansion)
        self.convs = LocalAggregation(
            [in_channels, in_channels],
            norm_args=norm_args,
            act_args=act_args if num_posconvs > 0 else None,
            group_args=group_args,
            conv_args=conv_args,
            query_chunk_size=query_chunk_size,
            **aggr_args,
            **kwargs,
        )

        if num_posconvs < 1:
            channels = []
        elif num_posconvs == 1:
            channels = [in_channels, in_channels]
        else:
            channels = [in_channels, mid_channels, in_channels]

        pwconv: List[nn.Module] = []
        for i in range(len(channels) - 1):
            pwconv.append(
                create_convblock1d(
                    channels[i],
                    channels[i + 1],
                    norm_args=norm_args,
                    act_args=act_args if (i != len(channels) - 2) and not less_act else None,
                    **conv_args,
                )
            )
        self.pwconv = nn.Sequential(*pwconv)
        self.act = create_act(act_args)

    def forward(self, pf):
        p, f = pf
        identity = f
        f = self.convs([p, f])
        f = self.pwconv(f)
        if f.shape[-1] == identity.shape[-1] and self.use_res:
            f = f + identity
        f = self.act(f)
        return [p, f]


class PointNextEncoder(nn.Module):
    """Minimal PointNeXt encoder path for classification features."""

    def __init__(
        self,
        in_channels: int = 4,
        width: int = 32,
        blocks: Sequence[int] = (1, 1, 1, 1, 1, 1),
        strides: Sequence[int] = (1, 2, 2, 2, 2, 1),
        block: str | Type[InvResMLP] = "InvResMLP",
        nsample: int | Sequence[int] = 32,
        radius: float | Sequence[float] = 0.15,
        aggr_args: Dict[str, Any] | None = None,
        group_args: Dict[str, Any] | None = None,
        sa_layers: int = 2,
        sa_use_res: bool = True,
        query_chunk_size: int = 256,
        **kwargs,
    ):
        super().__init__()
        if isinstance(block, str):
            block_name = block.lower()
            if block_name == "invresmlp":
                block = InvResMLP
            else:
                raise ValueError(f"Unsupported block type: {block}")

        self.blocks = list(blocks)
        self.strides = list(strides)
        self.in_channels = in_channels
        self.aggr_args = aggr_args or {"feature_type": "dp_fj", "reduction": "max"}
        self.norm_args = kwargs.get("norm_args", {"norm": "bn"})
        self.act_args = kwargs.get("act_args", {"act": "relu"})
        self.conv_args = kwargs.get("conv_args", None)
        self.sampler = kwargs.get("sampler", "fps")
        self.expansion = kwargs.get("expansion", 4)
        self.sa_layers = sa_layers
        self.sa_use_res = sa_use_res
        self.use_res = kwargs.get("use_res", True)
        self.query_chunk_size = query_chunk_size

        radius_scaling = kwargs.get("radius_scaling", 1.5)
        nsample_scaling = kwargs.get("nsample_scaling", 1.0)

        self.radii = self._to_full_list(radius, radius_scaling)
        self.nsample = self._to_full_list(nsample, nsample_scaling)
        self.nsample = [[int(max(1, round(v))) for v in stage] for stage in self.nsample]

        channels = []
        current_width = width
        for stride in self.strides:
            if stride != 1:
                current_width *= 2
            channels.append(current_width)

        encoder_stages: List[nn.Module] = []
        for i in range(len(self.blocks)):
            stage_group_args = copy.deepcopy(group_args or {"NAME": "ballquery", "normalize_dp": True})
            stage_group_args["radius"] = self.radii[i]
            stage_group_args["nsample"] = self.nsample[i]
            encoder_stages.append(
                self._make_enc(
                    block=block,
                    channels=channels[i],
                    blocks=self.blocks[i],
                    stride=self.strides[i],
                    group_args=stage_group_args,
                    is_head=(i == 0 and self.strides[i] == 1),
                )
            )

        self.encoder = nn.Sequential(*encoder_stages)
        self.out_channels = channels[-1]
        self.channel_list = channels

    def _to_full_list(self, param, param_scaling=1):
        param_list = []
        if isinstance(param, (list, tuple)):
            if len(param) != len(self.blocks):
                raise ValueError(
                    f"Expected list length {len(self.blocks)} for param, got {len(param)}."
                )
            for i, value in enumerate(param):
                value_list = [value] if not isinstance(value, (list, tuple)) else list(value)
                if len(value_list) != self.blocks[i]:
                    value_list += [value_list[-1]] * (self.blocks[i] - len(value_list))
                param_list.append(value_list)
        else:
            current = param
            for i, stride in enumerate(self.strides):
                if stride == 1:
                    param_list.append([current] * self.blocks[i])
                else:
                    stage_values = [current] + [current * param_scaling] * (self.blocks[i] - 1)
                    param_list.append(stage_values)
                    current = current * param_scaling
        return param_list

    def _make_enc(self, block, channels, blocks, stride, group_args, is_head=False):
        layers: List[nn.Module] = []
        radii = group_args["radius"]
        nsample = group_args["nsample"]

        sa_group_args = copy.deepcopy(group_args)
        sa_group_args["radius"] = radii[0]
        sa_group_args["nsample"] = nsample[0]
        layers.append(
            SetAbstraction(
                self.in_channels,
                channels,
                self.sa_layers if not is_head else 1,
                stride,
                group_args=sa_group_args,
                sampler=self.sampler,
                norm_args=self.norm_args,
                act_args=self.act_args,
                conv_args=self.conv_args,
                is_head=is_head,
                use_res=self.sa_use_res,
                query_chunk_size=self.query_chunk_size,
                **self.aggr_args,
            )
        )
        self.in_channels = channels

        for i in range(1, blocks):
            block_group_args = copy.deepcopy(group_args)
            block_group_args["radius"] = radii[i]
            block_group_args["nsample"] = nsample[i]
            layers.append(
                block(
                    self.in_channels,
                    aggr_args=self.aggr_args,
                    norm_args=self.norm_args,
                    act_args=self.act_args,
                    group_args=block_group_args,
                    conv_args=self.conv_args,
                    expansion=self.expansion,
                    use_res=self.use_res,
                    query_chunk_size=self.query_chunk_size,
                )
            )
        return nn.Sequential(*layers)

    def forward_cls_feat(self, p0: torch.Tensor, f0: torch.Tensor | None = None) -> torch.Tensor:
        if f0 is None:
            f0 = p0.clone().transpose(1, 2).contiguous()
        for i in range(len(self.encoder)):
            p0, f0 = self.encoder[i]([p0, f0])
        if f0.shape[-1] == 1:
            return f0.squeeze(-1)
        return torch.max(f0, dim=-1, keepdim=False)[0]

    def forward(self, p0: torch.Tensor, f0: torch.Tensor | None = None) -> torch.Tensor:
        return self.forward_cls_feat(p0, f0)


class PointNeXtBackbone(nn.Module):
    """PointNeXt backbone adapter for IDP3.

    Converts IDP3 point-cloud format [B, N, C] into PointNeXt input.
    This module is single-view only; multi-view merge/split is handled in IDP3Encoder.
    """

    def __init__(
        self,
        pc_channels: int = 3,
        out_channels: int = 128,
        pointnext_in_channels: int = 4,
        pointnext_width: int = 32,
        pointnext_blocks: Sequence[int] = (1, 1, 1, 1, 1, 1),
        pointnext_strides: Sequence[int] = (1, 2, 2, 2, 2, 1),
        pointnext_nsample: int = 32,
        pointnext_radius: float = 0.15,
        pointnext_radius_scaling: float = 1.5,
        pointnext_sa_layers: int = 2,
        pointnext_sa_use_res: bool = True,
        pointnext_expansion: int = 4,
        pointnext_sampler: str = "fps",
        pointnext_group_type: str = "ballquery",
        pointnext_query_chunk_size: int = 256,
        pointnext_batch_chunk_size: int = 16,
        pointnext_gravity_dim: int = 1,
        pointnext_auto_load_pretrained: bool = True,
        pointnext_pretrained_path: str = "dataset/models/pointnext/scanobjectnn-pointnext-s_best.pth",
        pointnext_freeze_backbone: bool = False,
        **kwargs,
    ):
        super().__init__()
        del kwargs

        self.pc_channels = pc_channels
        self.requested_out_channels = int(out_channels)
        self.pointnext_in_channels = pointnext_in_channels
        self.pointnext_batch_chunk_size = pointnext_batch_chunk_size
        self.pointnext_gravity_dim = int(pointnext_gravity_dim)

        # For pc_channels > pointnext_in_channels,
        # use direct channel slicing for lower latency and no extra params.
        self.use_channel_slice = self.pc_channels > self.pointnext_in_channels
        if self.pc_channels == self.pointnext_in_channels:
            self.input_adapter = nn.Identity()
        elif self.use_channel_slice:
            self.input_adapter = None
        else:
            self.input_adapter = nn.Conv1d(self.pc_channels, self.pointnext_in_channels, kernel_size=1, bias=True)
            self._init_input_adapter()

        self.encoder = PointNextEncoder(
            in_channels=pointnext_in_channels,
            width=pointnext_width,
            blocks=list(pointnext_blocks),
            strides=list(pointnext_strides),
            nsample=pointnext_nsample,
            radius=pointnext_radius,
            radius_scaling=pointnext_radius_scaling,
            sa_layers=pointnext_sa_layers,
            sa_use_res=pointnext_sa_use_res,
            expansion=pointnext_expansion,
            sampler=pointnext_sampler,
            aggr_args={"feature_type": "dp_fj", "reduction": "max"},
            group_args={"NAME": pointnext_group_type, "normalize_dp": True},
            conv_args={"order": "conv-norm-act"},
            act_args={"act": "relu"},
            norm_args={"norm": "bn"},
            query_chunk_size=pointnext_query_chunk_size,
        )
        encoder_out_channels = self.encoder.out_channels
        self.out_channels = int(encoder_out_channels)
        # Temporarily disable out_channels mapping. Keep these lines for quick re-enable:
        # self.output_proj = (
        #     nn.Identity()
        #     if encoder_out_channels == out_channels
        #     else nn.Linear(encoder_out_channels, out_channels)
        # )

        if pointnext_auto_load_pretrained:
            self.load_pointnext_pretrained_weights(pointnext_pretrained_path)
        if pointnext_freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def _init_input_adapter(self):
        if not isinstance(self.input_adapter, nn.Conv1d):
            return
        with torch.no_grad():
            self.input_adapter.weight.zero_()
            if self.input_adapter.bias is not None:
                self.input_adapter.bias.zero_()
            shared = min(self.pc_channels, self.pointnext_in_channels)
            for i in range(shared):
                self.input_adapter.weight[i, i, 0] = 1.0

    @staticmethod
    def _resolve_path(path_like: str | Path) -> Path | None:
        path = Path(path_like).expanduser()
        if path.is_file():
            return path
        repo_root = Path(__file__).resolve().parents[5]
        alt = repo_root / path
        if alt.is_file():
            return alt
        return None

    @staticmethod
    def _extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
        if isinstance(ckpt, dict):
            for key in ("model", "state_dict", "network", "net"):
                value = ckpt.get(key, None)
                if isinstance(value, dict):
                    return value
            if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
                return ckpt
        return {}

    def load_pointnext_pretrained_weights(self, pretrained_path: str | Path) -> bool:
        path = self._resolve_path(pretrained_path)
        if path is None:
            raise FileNotFoundError(f"[PointNeXtBackbone] pretrained checkpoint not found: {pretrained_path}")

        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location="cpu")
        raw_state = self._extract_state_dict(ckpt)
        if not raw_state:
            raise RuntimeError(f"[PointNeXtBackbone] invalid checkpoint format: {path}")

        encoder_state_to_load: Dict[str, torch.Tensor] = {}
        for raw_key, value in raw_state.items():
            key = raw_key
            if key.startswith("module."):
                key = key[len("module.") :]
            if not key.startswith("encoder."):
                continue
            key = key[len("encoder.") :]
            encoder_state_to_load[key] = value

        if not encoder_state_to_load:
            raise RuntimeError(f"[PointNeXtBackbone] no encoder weights found in checkpoint: {path}")

        # Strict loading by design: if config and checkpoint architecture mismatch,
        # PyTorch raises immediately.
        self.encoder.load_state_dict(encoder_state_to_load, strict=True)
        logger.info(
            "[PointNeXtBackbone] loaded pretrained encoder weights from %s (strict=True, tensors=%d)",
            path,
            len(encoder_state_to_load),
        )
        return True

    def _convert_idp3_to_pointnext_input(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert IDP3 input [B, N, C] to PointNeXt inputs.

        Returns:
            pos: [B, N, 3]
            feat:[B, C_pn, N]
        """
        if x.dim() != 3:
            raise ValueError(f"Point cloud must be [B, N, C], got {tuple(x.shape)}")
        if x.shape[-1] < 3:
            raise ValueError(f"Point cloud channel must be >= 3 (xyz), got {x.shape[-1]}")
        if self.pointnext_in_channels != 4:
            raise ValueError(
                f"Simplified PointNeXt input path expects pointnext_in_channels=4, got {self.pointnext_in_channels}"
            )
        if not 0 <= self.pointnext_gravity_dim < 3:
            raise ValueError(
                f"pointnext_gravity_dim must be in [0, 1, 2], got {self.pointnext_gravity_dim}"
            )

        # Always use xyz only, then append one relative-height channel.
        pos = x[:, :, :3].contiguous()
        height = pos[:, :, self.pointnext_gravity_dim : self.pointnext_gravity_dim + 1]
        height = height - height.amin(dim=1, keepdim=True)
        feat = torch.cat([pos, height], dim=-1).transpose(1, 2).contiguous()
        return pos, feat

    def _convert_pointnext_to_idp3_output(self, pointnext_feat: torch.Tensor) -> torch.Tensor:
        if pointnext_feat.dim() == 3:
            pointnext_feat = torch.max(pointnext_feat, dim=-1, keepdim=False)[0]
        if pointnext_feat.dim() != 2:
            raise ValueError(f"PointNeXt feature must be [B, F], got {tuple(pointnext_feat.shape)}")
        # Temporarily disable out_channels mapping. Keep this line for quick re-enable:
        # return self.output_proj(pointnext_feat)
        return pointnext_feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos, feat = self._convert_idp3_to_pointnext_input(x)
        bsz = pos.shape[0]

        if self.pointnext_batch_chunk_size and bsz > self.pointnext_batch_chunk_size:
            chunks = []
            for start in range(0, bsz, self.pointnext_batch_chunk_size):
                end = min(start + self.pointnext_batch_chunk_size, bsz)
                chunk_feat = self.encoder.forward_cls_feat(pos[start:end], feat[start:end])
                chunks.append(chunk_feat)
            pointnext_feat = torch.cat(chunks, dim=0)
        else:
            pointnext_feat = self.encoder.forward_cls_feat(pos, feat)

        return self._convert_pointnext_to_idp3_output(pointnext_feat)
