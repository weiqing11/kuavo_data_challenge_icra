"""Encoder modules for the diffusion_idp3 policy.

All functions include explicit purpose and input/output shape annotations.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torchvision

from transformers import SiglipImageProcessor, SiglipVisionModel
from peft import LoraConfig, get_peft_model

from lerobot.policies.diffusion.modeling_diffusion import SpatialSoftmax, _replace_submodules
from lerobot.policies.utils import get_output_shape
from kuavo_train.logger import logger


# =============================
# State / generic feature encoder
# =============================
class FeatureEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        """
        Purpose:
            Map low-dimensional state vectors to a shared token dimension using an MLP.
        Inputs (constructor):
            in_dim: int, input feature dimension.
            out_dim: int, output feature dimension.
        Outputs (constructor):
            None.
        Forward input shape:
            x: [B, D] or [B, T, D].
        Forward output shape:
            y: [B, out_dim] or [B, T, out_dim] matching input rank.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=False),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Purpose:
            Apply the MLP to either a batch of vectors or a batch of sequences.
        Inputs:
            x: Tensor with shape [B, D] or [B, T, D].
        Outputs:
            y: Tensor with shape [B, out_dim] or [B, T, out_dim].
        """
        if x.dim() == 2:
            return self.net(x)
        if x.dim() == 3:
            b, t, d = x.shape
            x_flat = x.reshape(b * t, d)
            y = self.net(x_flat).reshape(b, t, -1)
            return y
        raise ValueError("FeatureEncoder expects a 2D or 3D tensor.")


# =============================
# Token resampler
# =============================
class PerceiverResampler(nn.Module):
    def __init__(
        self,
        dim: int,
        num_queries: int = 64,
        depth: int = 2,
        heads: int = 8,
        dim_head: int = 64,
        ff_mult: int = 4,
    ) -> None:
        """
        Purpose:
            Compress a variable-length token sequence into a fixed number of latent tokens.
        Inputs (constructor):
            dim: int, token dimension.
            num_queries: int, number of latent queries.
            depth: int, number of cross/self-attention layers.
            heads: int, number of attention heads.
            dim_head: int, per-head dimension (unused, kept for API compatibility).
            ff_mult: int, expansion ratio for the feedforward layer.
        Outputs (constructor):
            None.
        Forward input shape:
            x: [B, N_in, dim].
        Forward output shape:
            latents: [B, num_queries, dim].
        """
        super().__init__()
        self.num_queries = num_queries
        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "cross_attn": nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True),
                        "cross_norm_q": nn.LayerNorm(dim),
                        "cross_norm_kv": nn.LayerNorm(dim),
                        "self_attn": nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True),
                        "self_norm": nn.LayerNorm(dim),
                        "ff": nn.Sequential(
                            nn.LayerNorm(dim),
                            nn.Linear(dim, dim * ff_mult),
                            nn.GELU(),
                            nn.Linear(dim * ff_mult, dim),
                        ),
                    }
                )
            )

        self.norm_out = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Purpose:
            Resample input tokens into a fixed number of latent tokens.
        Inputs:
            x: Tensor with shape [B, N_in, dim].
        Outputs:
            latents: Tensor with shape [B, num_queries, dim].
        """
        b = x.shape[0]
        latents = self.latents.repeat(b, 1, 1)

        for layer in self.layers:
            q = layer["cross_norm_q"](latents)
            k = v = layer["cross_norm_kv"](x)
            cross_out, _ = layer["cross_attn"](query=q, key=k, value=v)
            latents = latents + cross_out

            q_sa = layer["self_norm"](latents)
            self_out, _ = layer["self_attn"](query=q_sa, key=q_sa, value=q_sa)
            latents = latents + self_out

            latents = latents + layer["ff"](latents)

        return self.norm_out(latents)


# =============================
# RGB encoders
# =============================
class SiglipRGBEncoder(nn.Module):
    def __init__(self, config, target_dim: int) -> None:
        """
        Purpose:
            Extract RGB patch tokens with SigLIP and project them to a target token dimension.
        Inputs (constructor):
            config: policy config with SigLIP and LoRA settings.
            target_dim: int, output token dimension.
        Outputs (constructor):
            None.
        Forward input shape:
            rgb: [B, C, H, W].
        Forward output shape:
            tokens: [B, N_patches, target_dim].
        """
        super().__init__()

        self.mode = getattr(config, "vision_encoder_mode", "patches")
        if self.mode != "patches":
            raise ValueError(f"SiglipRGBEncoder only supports 'patches' mode. Got: {self.mode}")

        self.siglip_model_name = config.siglip_model_name
        is_local = "/" in self.siglip_model_name
        logger.info(f"Loading SigLIP model: {self.siglip_model_name}")
        self.siglip = SiglipVisionModel.from_pretrained(self.siglip_model_name, local_files_only=is_local)
        self.processor = SiglipImageProcessor.from_pretrained(self.siglip_model_name, local_files_only=is_local)

        self.use_lora = getattr(config, "use_lora", False)
        self.vision_freeze = getattr(config, "vision_freeze", True)

        if self.use_lora:
            peft_config = LoraConfig(
                r=getattr(config, "lora_rank", 16),
                lora_alpha=getattr(config, "lora_alpha", 32),
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
                lora_dropout=getattr(config, "lora_dropout", 0.05),
                bias="none",
            )
            self.siglip = get_peft_model(self.siglip, peft_config)
            self.vision_freeze = False
        elif self.vision_freeze:
            self.siglip.requires_grad_(False)
            self.siglip.eval()
        else:
            self.siglip.train()

        self.hidden_size = self.siglip.config.hidden_size
        self.patch_size = self.siglip.config.patch_size

        self.proj = nn.Linear(self.hidden_size, target_dim)
        self.norm = nn.LayerNorm(target_dim)

        # Compute number of patches based on resize/crop settings.
        if config.resize_shape is not None:
            h, w = config.resize_shape
        elif config.crop_shape is not None:
            if isinstance(list(config.crop_shape)[0], (list, tuple)):
                (x_start, x_end), (y_start, y_end) = config.crop_shape
                h, w = x_end - x_start, y_end - y_start
            else:
                h, w = config.crop_shape
        else:
            first_img_shape = next(iter(config.image_features.values())).shape
            h, w = first_img_shape[1], first_img_shape[2]
        self.num_patches = (h // self.patch_size) * (w // self.patch_size)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Purpose:
            Encode RGB images into projected SigLIP patch tokens.
        Inputs:
            rgb: Tensor with shape [B, C, H, W].
        Outputs:
            tokens: Tensor with shape [B, N_patches, target_dim].
        """
        requires_grad = self.use_lora or (not self.vision_freeze)
        context = torch.enable_grad() if requires_grad else torch.no_grad()

        with context:
            siglip_in = self.processor(images=rgb, do_resize=False, do_rescale=False, return_tensors="pt")
            siglip_out = self.siglip(siglip_in["pixel_values"].to(rgb.device), interpolate_pos_encoding=True)
            siglip_feat = siglip_out.last_hidden_state  # [B, N_patches, hidden_size]

        tokens = self.proj(siglip_feat)
        tokens = self.norm(tokens)
        return tokens


class ResnetRgbEncoder(nn.Module):
    def __init__(self, config, target_dim: int) -> None:
        """
        Purpose:
            Extract a single RGB token per image using a ResNet + SpatialSoftmax.
        Inputs (constructor):
            config: policy config with ResNet settings.
            target_dim: int, output token dimension.
        Outputs (constructor):
            None.
        Forward input shape:
            rgb: [B, C, H, W].
        Forward output shape:
            tokens: [B, 1, target_dim].
        """
        super().__init__()
        backbone_model = getattr(torchvision.models, config.vision_backbone)(weights=config.pretrained_backbone_weights)
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))
        if config.use_group_norm:
            if config.pretrained_backbone_weights:
                raise ValueError("Cannot replace BatchNorm in a pretrained model.")
            self.backbone = _replace_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(num_groups=max(1, x.num_features // 16), num_channels=x.num_features),
            )

        images_shape = next(iter(config.image_features.values())).shape
        if config.resize_shape is not None:
            dummy_shape_h_w = config.resize_shape
        elif config.crop_shape is not None:
            if isinstance(list(config.crop_shape)[0], (list, tuple)):
                (x_start, x_end), (y_start, y_end) = config.crop_shape
                dummy_shape_h_w = (x_end - x_start, y_end - y_start)
            else:
                dummy_shape_h_w = config.crop_shape
        else:
            dummy_shape_h_w = images_shape[1:]

        dummy_shape = (1, images_shape[0], *dummy_shape_h_w)
        feature_map_shape = get_output_shape(self.backbone, dummy_shape)[1:]

        self.pool = SpatialSoftmax(feature_map_shape, num_kp=config.spatial_softmax_num_keypoints)
        self.feature_dim = config.spatial_softmax_num_keypoints * 2
        self.proj = nn.Sequential(
            nn.Linear(self.feature_dim, target_dim),
            nn.ReLU(inplace=False),
        )
        self.num_patches = 1

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Purpose:
            Encode RGB images into a single token per image.
        Inputs:
            rgb: Tensor with shape [B, C, H, W].
        Outputs:
            tokens: Tensor with shape [B, 1, target_dim].
        """
        x = self.backbone(rgb)
        x = torch.flatten(self.pool(x), start_dim=1)
        x = self.proj(x)
        return x.unsqueeze(1)


class ResnetDepthEncoder(nn.Module):
    def __init__(self, config, target_dim: int) -> None:
        """
        Purpose:
            Extract a single depth token per image using a ResNet + SpatialSoftmax.
        Inputs (constructor):
            config: policy config with depth backbone settings.
            target_dim: int, output token dimension.
        Outputs (constructor):
            None.
        Forward input shape:
            depth: [B, 1, H, W].
        Forward output shape:
            tokens: [B, 1, target_dim].
        """
        super().__init__()
        backbone_model = getattr(torchvision.models, config.depth_backbone)(weights=config.pretrained_backbone_weights)
        modules = list(backbone_model.children())[:-2]
        if isinstance(modules[0], nn.Conv2d):
            old_conv = modules[0]
            modules[0] = nn.Conv2d(
                in_channels=1,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )
            with torch.no_grad():
                modules[0].weight = nn.Parameter(old_conv.weight.mean(dim=1, keepdim=True))
        self.backbone = nn.Sequential(*modules)
        if config.use_group_norm:
            if config.pretrained_backbone_weights:
                raise ValueError("Cannot replace BatchNorm in a pretrained model.")
            self.backbone = _replace_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(num_groups=max(1, x.num_features // 16), num_channels=x.num_features),
            )

        images_shape = next(iter(config.depth_features.values())).shape
        if config.resize_shape is not None:
            dummy_shape_h_w = config.resize_shape
        elif config.crop_shape is not None:
            if isinstance(list(config.crop_shape)[0], (list, tuple)):
                (x_start, x_end), (y_start, y_end) = config.crop_shape
                dummy_shape_h_w = (x_end - x_start, y_end - y_start)
            else:
                dummy_shape_h_w = config.crop_shape
        else:
            dummy_shape_h_w = images_shape[1:]

        dummy_shape = (1, 1, *dummy_shape_h_w)
        feature_map_shape = get_output_shape(self.backbone, dummy_shape)[1:]

        self.pool = SpatialSoftmax(feature_map_shape, num_kp=config.spatial_softmax_num_keypoints)
        self.feature_dim = config.spatial_softmax_num_keypoints * 2
        self.proj = nn.Sequential(
            nn.Linear(self.feature_dim, target_dim),
            nn.ReLU(inplace=False),
        )
        self.num_patches = 1

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        """
        Purpose:
            Encode depth images into a single token per image.
        Inputs:
            depth: Tensor with shape [B, 1, H, W].
        Outputs:
            tokens: Tensor with shape [B, 1, target_dim].
        """
        x = self.backbone(depth)
        x = torch.flatten(self.pool(x), start_dim=1)
        x = self.proj(x)
        return x.unsqueeze(1)


class DiffusionRgbEncoder(nn.Module):
    def __init__(self, config, target_dim: int) -> None:
        """
        Purpose:
            Select and wrap an RGB encoder based on configuration.
        Inputs (constructor):
            config: policy config with vision_backbone settings.
            target_dim: int, output token dimension.
        Outputs (constructor):
            None.
        Forward input shape:
            rgb: [B, C, H, W].
        Forward output shape:
            tokens: [B, N_tokens, target_dim].
        """
        super().__init__()
        backbone_type = config.vision_backbone

        if "siglip" in backbone_type:
            if "dformer" in backbone_type:
                logger.warning("vision_backbone contains 'dformer' but this policy uses SigLIP-only tokens.")
            self.model = SiglipRGBEncoder(config, target_dim=target_dim)
        elif "resnet" in backbone_type:
            self.model = ResnetRgbEncoder(config, target_dim=target_dim)
        else:
            raise ValueError(f"Unsupported vision_backbone: {backbone_type}")

        self.feature_dim = target_dim
        self.num_patches = self.model.num_patches

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Purpose:
            Encode RGB images into token sequences.
        Inputs:
            rgb: Tensor with shape [B, C, H, W].
        Outputs:
            tokens: Tensor with shape [B, N_tokens, target_dim].
        """
        return self.model(rgb)


# =============================
# Point cloud encoders (IDP3-style)
# =============================

def uniform_sampling_torch(point_cloud: torch.Tensor, num_points: int) -> torch.Tensor:
    """
    Purpose:
        Uniformly sample or pad a point cloud to a fixed number of points.
    Inputs:
        point_cloud: Tensor with shape [B, N, C].
        num_points: int, target number of points.
    Outputs:
        sampled: Tensor with shape [B, num_points, C].
    """
    b, n, c = point_cloud.shape
    device = point_cloud.device
    if n == num_points:
        return point_cloud
    if n > num_points:
        indices = torch.randperm(n, device=device)[:num_points]
        return point_cloud[:, indices]
    # pad if n < num_points
    num_pad = num_points - n
    pad = torch.zeros(b, num_pad, c, device=device, dtype=point_cloud.dtype)
    padded = torch.cat([point_cloud, pad], dim=1)
    # shuffle to avoid padding bias
    perm = torch.randperm(num_points, device=device)
    return padded[:, perm]


class MultiStagePointNetEncoder(nn.Module):
    def __init__(self, pc_channels: int = 3, h_dim: int = 128, out_channels: int = 128, num_layers: int = 4) -> None:
        """
        Purpose:
            Encode a point cloud into a single global feature via multi-stage PointNet.
        Inputs (constructor):
            pc_channels: int, point feature dimension (e.g., 3 or 6).
            h_dim: int, hidden dimension.
            out_channels: int, output feature dimension.
            num_layers: int, number of pointnet blocks.
        Outputs (constructor):
            None.
        Forward input shape:
            x: [B, N, C].
        Forward output shape:
            feat: [B, out_channels].
        """
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
        """
        Purpose:
            Compute global point cloud features via multi-stage local/global pooling.
        Inputs:
            x: Tensor with shape [B, N, C].
        Outputs:
            feat: Tensor with shape [B, out_channels].
        """
        x = x.transpose(1, 2)  # [B, C, N]
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


class IDP3PointCloudEncoder(nn.Module):
    def __init__(
        self,
        num_views: int,
        num_points: int,
        in_channels: int,
        out_channels: int,
        downsample: bool = True,
    ) -> None:
        """
        Purpose:
            Encode multi-view point clouds using a shared IDP3-style PointNet backbone.
        Inputs (constructor):
            num_views: int, number of point cloud views.
            num_points: int, target number of points per view.
            in_channels: int, point feature dimension.
            out_channels: int, output feature dimension per view.
            downsample: bool, whether to downsample/pad to num_points.
        Outputs (constructor):
            None.
        Forward input shape:
            point_clouds: list of length num_views, each tensor with shape [B, S, N, C] or [B, N, C].
        Forward output shape:
            feats: Tensor with shape [B*S, num_views, out_channels].
        """
        super().__init__()
        self.num_views = num_views
        self.num_points = num_points
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample

        self.extractor = MultiStagePointNetEncoder(pc_channels=in_channels, out_channels=out_channels)

    def _flatten_time(self, points: torch.Tensor) -> torch.Tensor:
        """
        Purpose:
            Flatten an optional time dimension into the batch dimension.
        Inputs:
            points: Tensor with shape [B, S, N, C] or [B, N, C].
        Outputs:
            flat: Tensor with shape [B*S, N, C].
        """
        if points.dim() == 4:
            b, s, n, c = points.shape
            return points.reshape(b * s, n, c)
        if points.dim() == 3:
            return points
        raise ValueError(f"Point cloud tensor must be 3D or 4D, got shape: {points.shape}")

    def _align_points(self, points: torch.Tensor) -> torch.Tensor:
        """
        Purpose:
            Ensure point clouds have the expected number of points and channels.
        Inputs:
            points: Tensor with shape [B, N, C].
        Outputs:
            aligned: Tensor with shape [B, num_points, in_channels].
        """
        b, n, c = points.shape
        if c < self.in_channels:
            raise ValueError(f"Point cloud channels {c} < expected {self.in_channels}")
        if c > self.in_channels:
            points = points[:, :, : self.in_channels]

        if self.downsample:
            points = uniform_sampling_torch(points, self.num_points)
        elif n != self.num_points:
            raise ValueError(
                f"Point cloud has {n} points, expected {self.num_points} when downsample=False"
            )
        return points

    def forward(self, point_clouds: Sequence[torch.Tensor]) -> torch.Tensor:
        """
        Purpose:
            Encode each view and return a per-view feature tensor.
        Inputs:
            point_clouds: list/tuple of tensors, each with shape [B, S, N, C] or [B, N, C].
        Outputs:
            feats: Tensor with shape [B*S, num_views, out_channels].
        """
        if len(point_clouds) != self.num_views:
            raise ValueError(f"Expected {self.num_views} point cloud views, got {len(point_clouds)}")

        view_list = []
        for points in point_clouds:
            flat = self._flatten_time(points)
            aligned = self._align_points(flat)
            view_list.append(aligned)

        # Stack along batch dimension for shared encoder execution.
        stacked = torch.cat(view_list, dim=0)  # [B*S*num_views, N, C]
        feats = self.extractor(stacked)  # [B*S*num_views, out_channels]

        # Reshape back to [B*S, num_views, out_channels]
        bs = view_list[0].shape[0]
        feats = feats.reshape(self.num_views, bs, -1).permute(1, 0, 2)
        return feats
