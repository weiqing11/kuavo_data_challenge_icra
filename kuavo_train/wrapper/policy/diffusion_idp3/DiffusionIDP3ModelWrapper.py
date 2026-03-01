"""Model wrapper for diffusion_idp3 policy with RGB + point cloud fusion."""

from __future__ import annotations

from typing import Dict, Optional

import einops
import torch
import torch.nn as nn
from torch import Tensor
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from lerobot.policies.diffusion.modeling_diffusion import DiffusionConditionalUnet1d, DiffusionModel
from lerobot.policies.utils import get_device_from_parameters, get_dtype_from_parameters
from lerobot.utils.constants import OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

from kuavo_train.logger import logger
from kuavo_train.wrapper.policy.diffusion_idp3.DiffusionIDP3ConfigWrapper import DiffusionIDP3ConfigWrapper
from kuavo_train.wrapper.policy.diffusion_idp3.dit import DiT_S
from kuavo_train.wrapper.policy.diffusion_idp3.encoders import (
    DiffusionRgbEncoder,
    FeatureEncoder,
    IDP3PointCloudEncoder,
    ResnetDepthEncoder,
)
from kuavo_train.wrapper.policy.diffusion_idp3.fusion import RgbPointCloudFusion

OBS_DEPTH = "observation.depth"


def _make_noise_scheduler_factory(name: str, **kwargs: Dict) -> DDPMScheduler | DDIMScheduler:
    """
    Purpose:
        Create a DDPM or DDIM noise scheduler based on the config name.
    Inputs:
        name: str, "DDPM" or "DDIM".
        **kwargs: scheduler keyword arguments.
    Outputs:
        scheduler: DDPMScheduler or DDIMScheduler instance.
    """
    if name == "DDPM":
        return DDPMScheduler(**kwargs)
    if name == "DDIM":
        return DDIMScheduler(**kwargs)
    raise ValueError(f"Unsupported noise scheduler type {name}")


class DiffusionIDP3ModelWrapper(DiffusionModel):
    def __init__(self, config: DiffusionIDP3ConfigWrapper) -> None:
        """
        Purpose:
            Build a diffusion model that fuses RGB SigLIP tokens with IDP3 point cloud features.
        Inputs (constructor):
            config: DiffusionIDP3ConfigWrapper with encoder and denoiser settings.
        Outputs (constructor):
            None.
        """
        # Parent init hack for DiffusionModel validation.
        orig_vis = config.vision_backbone
        orig_noise = config.noise_scheduler_type
        config.vision_backbone = "resnet18"
        config.noise_scheduler_type = "DDPM"
        super().__init__(config)
        config.vision_backbone = orig_vis
        config.noise_scheduler_type = orig_noise

        self.config = config
        self.cond_feat_dim = getattr(config, "transformer_n_emb", 384)

        self.use_point_cloud = getattr(config, "use_point_cloud", False)
        self.point_cloud_keys = getattr(config, "point_cloud_keys", [])
        self.strict_point_cloud_keys = getattr(config, "strict_point_cloud_keys", False)

        # Camera count is derived from image features if present, else from point cloud keys.
        if getattr(config, "image_features", None) and len(config.image_features) > 0:
            self.num_cameras = len(config.image_features)
        elif self.use_point_cloud:
            self.num_cameras = len(self.point_cloud_keys)
        else:
            self.num_cameras = 0

        if self.use_point_cloud and self.strict_point_cloud_keys:
            if len(self.point_cloud_keys) != self.num_cameras:
                raise ValueError("point_cloud_keys length must match number of cameras when strict_point_cloud_keys is True")

        # State encoder (optionally merges env_state).
        self.state_encoder = None
        self.state_projector = None
        self.state_input_dim = 0
        if getattr(self.config, "robot_state_feature", None) is not None:
            self.state_input_dim = self.config.robot_state_feature.shape[0]
            if getattr(self.config, "env_state_feature", None) is not None:
                self.state_input_dim += self.config.env_state_feature.shape[0]

            if getattr(self.config, "use_state_encoder", False):
                self.state_encoder = FeatureEncoder(self.state_input_dim, self.cond_feat_dim)
                self.state_token_dim = self.cond_feat_dim
            else:
                if self.state_input_dim != self.cond_feat_dim:
                    self.state_projector = nn.Linear(self.state_input_dim, self.cond_feat_dim)
                    self.state_token_dim = self.cond_feat_dim
                else:
                    self.state_token_dim = self.state_input_dim
        else:
            self.state_token_dim = 0

        # RGB encoder(s).
        self.rgb_encoder = None
        self.rgb_patches_per_img = 0
        if getattr(self.config, "image_features", None):
            if getattr(self.config, "use_separate_rgb_encoder_per_camera", False):
                encoders = [DiffusionRgbEncoder(self.config, target_dim=self.cond_feat_dim) for _ in range(self.num_cameras)]
                self.rgb_encoder = nn.ModuleList(encoders)
                self.rgb_patches_per_img = encoders[0].num_patches
            else:
                self.rgb_encoder = DiffusionRgbEncoder(self.config, target_dim=self.cond_feat_dim)
                self.rgb_patches_per_img = self.rgb_encoder.num_patches

        # Depth encoder (optional single token per camera).
        self.use_depth = getattr(self.config, "use_depth", False)
        self.depth_encoder = None
        self.depth_tokens_per_img = 0
        if self.use_depth and getattr(self.config, "depth_features", None) and len(self.config.depth_features) > 0:
            self.depth_encoder = ResnetDepthEncoder(self.config, target_dim=self.cond_feat_dim)
            self.depth_tokens_per_img = self.depth_encoder.num_patches

        # Point cloud encoder.
        self.point_cloud_encoder = None
        self.pc_out_dim = 0
        if self.use_point_cloud:
            pc_encoder_type = getattr(self.config, "point_cloud_encoder_type", "idp3_multi_stage_pointnet")
            if pc_encoder_type != "idp3_multi_stage_pointnet":
                raise ValueError(f"Unsupported point_cloud_encoder_type: {pc_encoder_type}")
            pc_num_points = getattr(self.config, "point_cloud_num_points", 1024)
            pc_in_channels = getattr(self.config, "point_cloud_in_channels", 3)
            pc_out_channels = getattr(self.config, "point_cloud_out_channels", 128)
            pc_downsample = getattr(self.config, "point_cloud_downsample", True)
            self.point_cloud_encoder = IDP3PointCloudEncoder(
                num_views=len(self.point_cloud_keys),
                num_points=pc_num_points,
                in_channels=pc_in_channels,
                out_channels=pc_out_channels,
                downsample=pc_downsample,
            )
            self.pc_out_dim = pc_out_channels

        # Fusion module for RGB + point cloud.
        self.fusion = None
        if self.use_point_cloud:
            self.fusion = RgbPointCloudFusion(token_dim=self.cond_feat_dim, pc_dim=self.pc_out_dim)

        # Token counts per camera.
        pc_tokens_per_cam = 1 if self.use_point_cloud else 0
        rgb_tokens_per_cam = self.rgb_patches_per_img
        depth_tokens_per_cam = self.depth_tokens_per_img
        self.tokens_per_camera = rgb_tokens_per_cam + pc_tokens_per_cam + depth_tokens_per_cam

        # Perceiver resampler (optional).
        self.use_perceiver = getattr(self.config, "use_perceiver", False)
        if self.use_perceiver:
            from kuavo_train.wrapper.policy.diffusion_idp3.encoders import PerceiverResampler

            self.perceiver = PerceiverResampler(
                dim=self.cond_feat_dim,
                num_queries=getattr(self.config, "perceiver_num_queries", 64),
                depth=getattr(self.config, "perceiver_depth", 2),
                heads=getattr(self.config, "transformer_n_head", 8),
            )
            self.vision_seq_len = getattr(self.config, "perceiver_num_queries", 64)
            self.use_camera_embed = False
        else:
            self.perceiver = None
            self.vision_seq_len = self.num_cameras * self.tokens_per_camera
            self.use_camera_embed = True

        # Denoiser selection.
        if getattr(self.config, "use_dit", False):
            self.unet = DiT_S(
                action_dim=self.config.output_features["action"].shape[0],
                action_seq_len=self.config.horizon,
                n_obs_steps=self.config.n_obs_steps,
                token_dim=self.cond_feat_dim,
                max_image_tokens=self.vision_seq_len,
                num_cameras=self.num_cameras,
                use_camera_embed=self.use_camera_embed,
            )
            self.use_token_conditioning = True
        elif getattr(self.config, "use_unet", False):
            self.use_token_conditioning = False
            global_cond_dim = self._compute_vector_cond_dim()
            self.unet = DiffusionConditionalUnet1d(self.config, global_cond_dim=global_cond_dim)
        else:
            raise ValueError("Config must enable either use_dit or use_unet.")

        # Noise scheduler.
        self.noise_scheduler = _make_noise_scheduler_factory(
            self.config.noise_scheduler_type,
            num_train_timesteps=self.config.num_train_timesteps,
            beta_start=self.config.beta_start,
            beta_end=self.config.beta_end,
            beta_schedule=self.config.beta_schedule,
            clip_sample=self.config.clip_sample,
            clip_sample_range=self.config.clip_sample_range,
            prediction_type=self.config.prediction_type,
        )
        self.num_inference_steps = self.config.num_inference_steps or self.noise_scheduler.config.num_train_timesteps

        logger.info(
            f"DiffusionIDP3ModelWrapper: tokens_per_camera={self.tokens_per_camera}, vision_seq_len={self.vision_seq_len}, "
            f"use_token_conditioning={self.use_token_conditioning}"
        )

    def _compute_vector_cond_dim(self) -> int:
        """
        Purpose:
            Compute the global conditioning dimension for UNet-style vector conditioning.
        Inputs:
            None.
        Outputs:
            global_cond_dim: int, flattened conditioning dimension.
        """
        per_step_dim = 0
        if self.rgb_encoder is not None:
            per_step_dim += self.cond_feat_dim * self.num_cameras
        if self.use_point_cloud:
            per_step_dim += self.cond_feat_dim * self.num_cameras
        if self.depth_encoder is not None:
            per_step_dim += self.cond_feat_dim * self.num_cameras
        if self.state_token_dim > 0:
            per_step_dim += self.state_token_dim
        return per_step_dim * self.config.n_obs_steps

    def _get_state_tensor(self, batch: Dict[str, Tensor]) -> Optional[Tensor]:
        """
        Purpose:
            Build a state tensor that optionally concatenates env_state.
        Inputs:
            batch: dict of tensors containing OBS_STATE and optional OBS_ENV_STATE.
        Outputs:
            state: Tensor [B, S, D_state] or None.
        """
        if OBS_STATE not in batch:
            return None
        state = batch[OBS_STATE]
        if OBS_ENV_STATE in batch:
            state = torch.cat([state, batch[OBS_ENV_STATE]], dim=-1)
        return state

    def _prepare_token_conditioning(self, batch: Dict[str, Tensor]) -> Tensor:
        """
        Purpose:
            Prepare token-based conditioning for DiT.
        Inputs:
            batch: dict with keys OBS_STATE, OBS_IMAGES (optional), OBS_DEPTH (optional), and point cloud keys.
            Shapes:
                OBS_STATE: [B, S, D_state]
                OBS_IMAGES: [B, S, N_cam, C, H, W]
                OBS_DEPTH: [B, S, N_cam, 1, H, W]
                point cloud: [B, S, N_points, C]
        Outputs:
            global_cond: Tensor [B, S, T_tokens, D_token].
        """
        b = batch[OBS_STATE].shape[0]
        s = batch[OBS_STATE].shape[1]
        tokens_list = []

        # RGB tokens.
        rgb_tokens = None
        if self.rgb_encoder is not None and OBS_IMAGES in batch:
            if isinstance(self.rgb_encoder, nn.ModuleList):
                imgs = einops.rearrange(batch[OBS_IMAGES], "b s n c h w -> n (b s) c h w")
                enc_outs = [enc(im) for enc, im in zip(self.rgb_encoder, imgs, strict=True)]
                rgb_tokens = torch.stack(enc_outs, dim=1)  # [B*S, N_cam, P, D]
            else:
                imgs = einops.rearrange(batch[OBS_IMAGES], "b s n c h w -> (b s n) c h w")
                rgb_tokens = self.rgb_encoder(imgs)
                rgb_tokens = einops.rearrange(rgb_tokens, "(b s n) p d -> (b s) n p d", b=b, s=s, n=self.num_cameras)

        # Depth tokens.
        depth_tokens = None
        if self.depth_encoder is not None and OBS_DEPTH in batch:
            depths = einops.rearrange(batch[OBS_DEPTH], "b s n c h w -> (b s n) c h w")
            depth_tokens = self.depth_encoder(depths)
            depth_tokens = einops.rearrange(depth_tokens, "(b s n) p d -> (b s) n p d", b=b, s=s, n=self.num_cameras)

        # Point cloud tokens.
        pc_tokens = None
        if self.use_point_cloud:
            pc_list = [batch[key] for key in self.point_cloud_keys]
            pc_tokens = self.point_cloud_encoder(pc_list)

        # Fuse RGB + point cloud, then append depth tokens.
        if rgb_tokens is not None or pc_tokens is not None:
            # === [FUSION][REVIEW] RGB + PointCloud tokens are fused here ===
            fused_tokens = self.fusion(rgb_tokens, pc_tokens) if self.fusion is not None else rgb_tokens
            if depth_tokens is not None:
                fused_tokens = torch.cat([fused_tokens, depth_tokens], dim=2)
            fused_tokens = fused_tokens.reshape(b * s, -1, self.cond_feat_dim)
            if self.perceiver is not None:
                fused_tokens = self.perceiver(fused_tokens)
            fused_tokens = einops.rearrange(fused_tokens, "(b s) t d -> b s t d", b=b, s=s)
            tokens_list.append(fused_tokens)

        # State token.
        state_tensor = self._get_state_tensor(batch)
        if state_tensor is not None:
            if self.state_encoder is not None:
                state_emb = self.state_encoder(state_tensor)
            elif self.state_projector is not None:
                state_emb = self.state_projector(state_tensor)
            else:
                state_emb = state_tensor
            tokens_list.append(state_emb.unsqueeze(2))

        if len(tokens_list) == 0:
            raise ValueError("No conditioning tokens available.")
        return torch.cat(tokens_list, dim=2)

    def _prepare_vector_conditioning(self, batch: Dict[str, Tensor]) -> Tensor:
        """
        Purpose:
            Prepare vector-based conditioning for UNet.
        Inputs:
            batch: dict with keys OBS_STATE, OBS_IMAGES (optional), OBS_DEPTH (optional), and point cloud keys.
            Shapes:
                OBS_STATE: [B, S, D_state]
                OBS_IMAGES: [B, S, N_cam, C, H, W]
                OBS_DEPTH: [B, S, N_cam, 1, H, W]
                point cloud: [B, S, N_points, C]
        Outputs:
            global_cond: Tensor [B, global_cond_dim].
        """
        b = batch[OBS_STATE].shape[0]
        s = batch[OBS_STATE].shape[1]
        feats = []

        if self.rgb_encoder is not None and OBS_IMAGES in batch:
            if isinstance(self.rgb_encoder, nn.ModuleList):
                imgs = einops.rearrange(batch[OBS_IMAGES], "b s n c h w -> n (b s) c h w")
                enc_outs = [enc(im) for enc, im in zip(self.rgb_encoder, imgs, strict=True)]
                rgb_tokens = torch.stack(enc_outs, dim=1)  # [B*S, N_cam, P, D]
            else:
                imgs = einops.rearrange(batch[OBS_IMAGES], "b s n c h w -> (b s n) c h w")
                rgb_tokens = self.rgb_encoder(imgs)
                rgb_tokens = einops.rearrange(rgb_tokens, "(b s n) p d -> (b s) n p d", b=b, s=s, n=self.num_cameras)
            rgb_vec = rgb_tokens.mean(dim=2)
            rgb_vec = rgb_vec.reshape(b, s, -1)
            feats.append(rgb_vec)

        if self.depth_encoder is not None and OBS_DEPTH in batch:
            depths = einops.rearrange(batch[OBS_DEPTH], "b s n c h w -> (b s n) c h w")
            depth_tokens = self.depth_encoder(depths)
            depth_tokens = einops.rearrange(depth_tokens, "(b s n) p d -> (b s) n p d", b=b, s=s, n=self.num_cameras)
            depth_vec = depth_tokens.squeeze(2).reshape(b, s, -1)
            feats.append(depth_vec)

        if self.use_point_cloud:
            pc_list = [batch[key] for key in self.point_cloud_keys]
            pc_tokens = self.point_cloud_encoder(pc_list)
            pc_tokens = self.fusion.pc_proj(pc_tokens) if self.fusion is not None else pc_tokens
            pc_vec = pc_tokens.reshape(b, s, -1)
            feats.append(pc_vec)

        state_tensor = self._get_state_tensor(batch)
        if state_tensor is not None:
            if self.state_encoder is not None:
                state_vec = self.state_encoder(state_tensor)
            elif self.state_projector is not None:
                state_vec = self.state_projector(state_tensor)
            else:
                state_vec = state_tensor
            feats.append(state_vec)

        if len(feats) == 0:
            raise ValueError("No conditioning features available for vector conditioning.")
        global_cond = torch.cat(feats, dim=-1).flatten(start_dim=1)
        return global_cond

    def _prepare_global_conditioning(self, batch: Dict[str, Tensor]) -> Tensor:
        """
        Purpose:
            Prepare conditioning in token or vector form depending on denoiser type.
        Inputs:
            batch: dict of tensors with observation keys.
        Outputs:
            global_cond: token tensor [B, S, T, D] or vector tensor [B, D].
        """
        if self.use_token_conditioning:
            return self._prepare_token_conditioning(batch)
        return self._prepare_vector_conditioning(batch)

    def conditional_sample(
        self,
        batch_size: int,
        global_cond: Optional[Tensor] = None,
        generator=None,
        noise: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Purpose:
            Perform diffusion sampling conditioned on global features.
        Inputs:
            batch_size: int.
            global_cond: Tensor [B, ...] or None.
            generator: optional torch.Generator.
            noise: optional Tensor [B, horizon, action_dim].
        Outputs:
            sample: Tensor [B, horizon, action_dim].
        """
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)
        sample = (
            noise
            if noise is not None
            else torch.randn(
                size=(batch_size, self.config.horizon, self.config.action_feature.shape[0]),
                dtype=dtype,
                device=device,
                generator=generator,
            )
        )
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for t in self.noise_scheduler.timesteps:
            model_output = self.unet(
                sample,
                torch.full((batch_size,), t, dtype=torch.long, device=device),
                global_cond=global_cond,
            )
            step_kwargs = {"generator": generator}
            if "eta" in self.noise_scheduler.step.__code__.co_varnames:
                step_kwargs["eta"] = getattr(self.config, "ddim_eta", 0.0)
            step_out = self.noise_scheduler.step(model_output, t, sample, **step_kwargs)
            sample = getattr(step_out, "prev_sample", step_out)
        return sample
