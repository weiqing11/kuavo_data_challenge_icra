import importlib
from typing import Dict, Type

import torch.nn as nn

POINTNET_BACKBONE_REGISTRY: Dict[str, Type[nn.Module] | tuple[str, str]] = {
    "multi_stage_pointnet": (".multi_stage_pointnet", "MultiStagePointNetEncoder"),
    "pointnext": (".pointnext_backbone", "PointNeXtBackbone"),
    "ptv3": (".pointtransformerv3_backbone", "PointTransformerV3Backbone"),
}

_MODULE_PACKAGE = __name__


def _resolve_backbone_class(backbone_type: str) -> Type[nn.Module]:
    entry = POINTNET_BACKBONE_REGISTRY[backbone_type]
    if isinstance(entry, tuple):
        module_name, class_name = entry
        module = importlib.import_module(module_name, package=_MODULE_PACKAGE)
        cls = getattr(module, class_name)
        POINTNET_BACKBONE_REGISTRY[backbone_type] = cls
        return cls
    return entry


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
    backbone_cls = _resolve_backbone_class(backbone_type)
    return backbone_cls(**kwargs)
