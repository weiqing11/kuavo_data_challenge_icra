from typing import Dict, Type

import torch.nn as nn

from .multi_stage_pointnet import MultiStagePointNetEncoder
from .pointnext_backbone import PointNeXtBackbone


POINTNET_BACKBONE_REGISTRY: Dict[str, Type[nn.Module]] = {
    "multi_stage_pointnet": MultiStagePointNetEncoder,
    "pointnext": PointNeXtBackbone,
}


def build_pointnet_backbone(backbone_type: str, **kwargs) -> nn.Module:
    """Build a point-cloud backbone.

    Backbone interface contract:
    - Input:  [B, N, C]
    - Output: [B, F]
      where F is the configured feature dimension (typically `out_channels`).
    """
    if backbone_type not in POINTNET_BACKBONE_REGISTRY:
        available = ", ".join(sorted(POINTNET_BACKBONE_REGISTRY))
        raise NotImplementedError(
            f"Unsupported point cloud backbone '{backbone_type}'. Available backbones: {available}"
        )
    return POINTNET_BACKBONE_REGISTRY[backbone_type](**kwargs)
