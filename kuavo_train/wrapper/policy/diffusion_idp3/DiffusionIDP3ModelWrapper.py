from __future__ import annotations

from typing import Optional

import einops
import torch
from torch import Tensor, nn

from lerobot.policies.diffusion.modeling_diffusion import DiffusionConditionalUnet1d, DiffusionModel
from lerobot.policies.utils import get_device_from_parameters, get_dtype_from_parameters
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE
from kuavo_train.logger import log_box
from kuavo_train.wrapper.policy.diffusion_idp3.DiffusionIDP3ConfigWrapper import (
    DiffusionIDP3ConfigWrapper,
)
from kuavo_train.wrapper.policy.diffusion_idp3.pointcloud_backbones import (
    PointCloudTokenProjector,
    build_point_cloud_backbone,
)
from kuavo_train.wrapper.policy.diffusion_new.DiT_1D_AdaLN import DiT_S
from kuavo_train.wrapper.policy.diffusion_new.DiffusionModelWrapper import (
    DiffusionRgbEncoder,
    FeatureEncoder,
    PerceiverResampler,
    _make_noise_scheduler_factory,
)
from kuavo_train.wrapper.policy.diffusion_new.transformer_diffusion import TransformerForDiffusion


OBS_DEPTH = "observation.depth"


class DiffusionIDP3ModelWrapper(DiffusionModel):
    """
    Diffusion model with IDP3-style point-cloud tokens.

    It keeps diffusion_new capabilities (vision backbones, LoRA path, Perceiver, and
    UNet/Transformer/DiT denoisers), while adding point-cloud features as extra
    per-step tokens.
    """

    def __init__(self, config: DiffusionIDP3ConfigWrapper):
        # Parent init compatibility hack (same pattern as diffusion_new wrappers).
        original_backbone = config.vision_backbone
        config.vision_backbone = "resnet18"
        original_scheduler = config.noise_scheduler_type
        config.noise_scheduler_type = "DDPM"
        super().__init__(config)
        config.vision_backbone = original_backbone
        config.noise_scheduler_type = original_scheduler
        self.config = config

        self.cond_feat_dim = getattr(self.config, "transformer_n_emb", 384)
        self.use_dit = getattr(self.config, "use_dit", False)
        self.use_transformer = getattr(self.config, "use_transformer", False)
        self.use_unet = getattr(self.config, "use_unet", False)

        # ------------------------------------------------------------------
        # State token encoder (state token must stay last for DiT compatibility)
        # ------------------------------------------------------------------
        self.state_token_encoder: Optional[nn.Module] = None
        self.state_token_count = 0
        if self.config.robot_state_feature is not None:
            state_dim = self.config.robot_state_feature.shape[0]
            if getattr(self.config, "use_state_encoder", True):
                self.state_token_encoder = FeatureEncoder(state_dim, self.cond_feat_dim)
            else:
                self.state_token_encoder = nn.Linear(state_dim, self.cond_feat_dim)
            self.state_token_count = 1

        # ------------------------------------------------------------------
        # Vision encoder path (keeps diffusion_new implementation and LoRA logic)
        # ------------------------------------------------------------------
        self.rgb_encoder: DiffusionRgbEncoder | nn.ModuleList | None = None
        self.vision_feature_dim = 0
        self.raw_vision_tokens_per_step = 0
        if getattr(self.config, "image_features", None):
            num_images = len(self.config.image_features)
            if getattr(self.config, "use_separate_rgb_encoder_per_camera", False):
                encoders = [DiffusionRgbEncoder(config) for _ in range(num_images)]
                self.rgb_encoder = nn.ModuleList(encoders)
                one_encoder = encoders[0].model
                self.vision_feature_dim = encoders[0].feature_dim
            else:
                self.rgb_encoder = DiffusionRgbEncoder(config)
                one_encoder = self.rgb_encoder.model
                self.vision_feature_dim = self.rgb_encoder.feature_dim

            patches_per_image = getattr(one_encoder, "num_patches", 1)
            self.raw_vision_tokens_per_step = num_images * patches_per_image

        self.vision_token_projector: Optional[nn.Module] = None
        if self.vision_feature_dim > 0 and self.vision_feature_dim != self.cond_feat_dim:
            self.vision_token_projector = nn.Sequential(
                nn.Linear(self.vision_feature_dim, self.cond_feat_dim),
                nn.LayerNorm(self.cond_feat_dim),
            )

        # Optional token compression over vision tokens only.
        self.use_perceiver = bool(getattr(self.config, "use_perceiver", False))
        self.perceiver: Optional[PerceiverResampler] = None
        if self.use_perceiver and self.raw_vision_tokens_per_step > 0:
            perceiver_queries = int(getattr(self.config, "perceiver_num_queries", 64))
            self.perceiver = PerceiverResampler(
                dim=self.cond_feat_dim,
                num_queries=perceiver_queries,
                depth=int(getattr(self.config, "perceiver_depth", 2)),
                heads=int(getattr(self.config, "transformer_n_head", 8)),
            )
            self.vision_tokens_per_step = perceiver_queries
        else:
            self.vision_tokens_per_step = self.raw_vision_tokens_per_step

        # ------------------------------------------------------------------
        # Point-cloud token path
        # ------------------------------------------------------------------
        self.use_point_cloud = bool(getattr(self.config, "use_point_cloud", True))
        self.point_cloud_keys = list(getattr(self.config, "point_cloud_keys", []))
        self.point_cloud_backbone = None
        self.point_cloud_projector = None
        self.point_cloud_align = None
        self.point_cloud_tokens_per_step = 0
        if self.use_point_cloud:
            self.point_cloud_backbone = build_point_cloud_backbone(
                name=str(getattr(self.config, "point_cloud_encoder_type", "idp3_multi_stage_pointnet")),
                point_cloud_keys=self.point_cloud_keys,
                num_points=int(getattr(self.config, "point_cloud_num_points", 4096)),
                in_channels=int(getattr(self.config, "point_cloud_in_channels", 6)),
                out_channels=int(getattr(self.config, "point_cloud_out_channels", 128)),
                downsample=bool(getattr(self.config, "point_cloud_downsample", False)),
            )
            pc_project_dim = int(getattr(self.config, "point_cloud_project_dim", self.cond_feat_dim))
            self.point_cloud_projector = PointCloudTokenProjector(
                in_dim=self.point_cloud_backbone.output_dim,
                out_dim=pc_project_dim,
            )
            if pc_project_dim != self.cond_feat_dim:
                self.point_cloud_align = nn.Linear(pc_project_dim, self.cond_feat_dim)
            self.point_cloud_tokens_per_step = self.point_cloud_backbone.token_count

        self.tokens_per_step = (
            self.vision_tokens_per_step + self.point_cloud_tokens_per_step + self.state_token_count
        )
        if self.tokens_per_step <= 0:
            raise ValueError("At least one conditioning token must be present per observation step.")

        if self.use_dit and self.state_token_count == 0:
            raise ValueError("DiT path requires a state token. Please keep `observation.state` as input.")

        # ------------------------------------------------------------------
        # Denoiser backends (UNet / Transformer / DiT)
        # ------------------------------------------------------------------
        if self.use_unet:
            global_cond_dim = self.config.n_obs_steps * self.tokens_per_step * self.cond_feat_dim
            self.unet = DiffusionConditionalUnet1d(config, global_cond_dim=global_cond_dim)
        elif self.use_transformer:
            total_cond_len = self.config.n_obs_steps * self.tokens_per_step
            self.unet = TransformerForDiffusion(
                input_dim=config.output_features["action"].shape[0],
                output_dim=config.output_features["action"].shape[0],
                horizon=config.horizon,
                n_obs_steps=total_cond_len,
                cond_dim=self.cond_feat_dim,
                n_layer=self.config.transformer_n_layer,
                n_head=self.config.transformer_n_head,
                n_emb=self.config.transformer_n_emb,
                p_drop_emb=self.config.transformer_dropout,
                p_drop_attn=self.config.transformer_dropout,
                causal_attn=False,
                time_as_cond=True,
                obs_as_cond=True,
                n_cond_layers=0,
            )
        elif self.use_dit:
            non_state_tokens = max(self.tokens_per_step - self.state_token_count, 1)
            self.unet = DiT_S(
                action_dim=config.output_features["action"].shape[0],
                action_seq_len=config.horizon,
                n_obs_steps=self.config.n_obs_steps,
                token_dim=self.cond_feat_dim,
                max_image_tokens=non_state_tokens,
            )
        else:
            raise ValueError("Either `use_unet`, `use_transformer`, or `use_dit` must be enabled.")

        # ------------------------------------------------------------------
        # Noise scheduler
        # ------------------------------------------------------------------
        self.noise_scheduler = _make_noise_scheduler_factory(
            config.noise_scheduler_type,
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
            prediction_type=config.prediction_type,
        )
        self.num_inference_steps = config.num_inference_steps or self.noise_scheduler.config.num_train_timesteps

        self._log_architecture()

    def _forward_rgb_encoder(
        self,
        encoder: nn.Module,
        images: Tensor,
        depths: Tensor | None,
    ) -> Tensor:
        if depths is not None:
            try:
                return encoder(images, depths)
            except TypeError:
                return encoder(images)
        return encoder(images)

    @staticmethod
    def _ensure_tokens(features: Tensor) -> Tensor:
        # Vector features -> one token; token features are kept as-is.
        if features.dim() == 2:
            return features.unsqueeze(1)
        if features.dim() == 3:
            return features
        raise ValueError(f"Expected 2D or 3D tensor from encoder, got shape {tuple(features.shape)}.")

    def _encode_vision_tokens(self, batch: dict[str, Tensor], batch_size: int, n_obs_steps: int) -> Tensor | None:
        if self.rgb_encoder is None or not getattr(self.config, "image_features", None):
            return None

        if getattr(self.config, "use_separate_rgb_encoder_per_camera", False):
            images_per_camera = einops.rearrange(batch[OBS_IMAGES], "b s n ... -> n (b s) ...")
            depth_per_camera = None
            if OBS_DEPTH in batch:
                depth_per_camera = einops.rearrange(batch[OBS_DEPTH], "b s n c h w -> n (b s) c h w")

            token_list = []
            for camera_idx, (encoder, images) in enumerate(zip(self.rgb_encoder, images_per_camera, strict=True)):
                camera_depth = depth_per_camera[camera_idx] if depth_per_camera is not None else None
                encoded = self._forward_rgb_encoder(encoder, images, camera_depth)
                token_list.append(self._ensure_tokens(encoded))
            vision_tokens = torch.cat(token_list, dim=1)  # (B*S, N_tokens, D_vis)
        else:
            images = einops.rearrange(batch[OBS_IMAGES], "b s n ... -> (b s n) ...")
            depths = None
            if OBS_DEPTH in batch:
                depths = einops.rearrange(batch[OBS_DEPTH], "b s n c h w -> (b s n) c h w")
            encoded = self._forward_rgb_encoder(self.rgb_encoder, images, depths)
            camera_tokens = self._ensure_tokens(encoded)  # (B*S*N, T_cam, D_vis)
            vision_tokens = einops.rearrange(
                camera_tokens,
                "(b s n) t d -> (b s) (n t) d",
                b=batch_size,
                s=n_obs_steps,
            )

        if self.vision_token_projector is not None:
            vision_tokens = self.vision_token_projector(vision_tokens)

        if self.perceiver is not None:
            vision_tokens = self.perceiver(vision_tokens)

        return einops.rearrange(vision_tokens, "(b s) t d -> b s t d", b=batch_size, s=n_obs_steps)

    def _encode_point_cloud_tokens(
        self,
        batch: dict[str, Tensor],
        batch_size: int,
        n_obs_steps: int,
    ) -> Tensor | None:
        if not self.use_point_cloud or self.point_cloud_backbone is None:
            return None

        if getattr(self.config, "strict_point_cloud_keys", True):
            missing = [key for key in self.point_cloud_keys if key not in batch]
            if missing:
                raise KeyError(
                    f"Point-cloud keys missing in batch: {missing}. Available keys: {sorted(batch.keys())}"
                )

        pc_tokens = self.point_cloud_backbone(batch)  # (B*S, T_pc, D_raw)
        pc_tokens = self.point_cloud_projector(pc_tokens)
        if self.point_cloud_align is not None:
            pc_tokens = self.point_cloud_align(pc_tokens)

        return einops.rearrange(pc_tokens, "(b s) t d -> b s t d", b=batch_size, s=n_obs_steps)

    def _encode_state_tokens(self, batch: dict[str, Tensor]) -> Tensor | None:
        if self.state_token_encoder is None:
            return None

        state_tensor = batch[OBS_STATE]  # (B, S, state_dim)
        if isinstance(self.state_token_encoder, FeatureEncoder):
            state_emb = self.state_token_encoder(state_tensor)
        else:
            state_emb = self.state_token_encoder(state_tensor)
        return state_emb.unsqueeze(2)  # (B, S, 1, D)

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]

        token_parts: list[Tensor] = []

        vision_tokens = self._encode_vision_tokens(batch, batch_size, n_obs_steps)
        if vision_tokens is not None:
            token_parts.append(vision_tokens)

        pc_tokens = self._encode_point_cloud_tokens(batch, batch_size, n_obs_steps)
        if pc_tokens is not None:
            token_parts.append(pc_tokens)

        # Keep state token as the last token per step for DiT.
        state_tokens = self._encode_state_tokens(batch)
        if state_tokens is not None:
            token_parts.append(state_tokens)

        if len(token_parts) == 0:
            raise ValueError("No conditioning tokens were produced.")

        combined = torch.cat(token_parts, dim=2)  # (B, S, T, D)

        if self.use_dit:
            return combined
        if self.use_transformer:
            return einops.rearrange(combined, "b s t d -> b (s t) d")
        return einops.rearrange(combined, "b s t d -> b (s t d)")

    def _log_architecture(self) -> None:
        if self.use_dit:
            denoiser = "Transformer (DiT)"
        elif self.use_transformer:
            denoiser = "Standard Transformer"
        else:
            denoiser = "Conditional UNet-1D"

        arch_info = {
            "Denoiser Type": denoiser,
            "Condition Dim": self.cond_feat_dim,
            "Obs Steps (S)": self.config.n_obs_steps,
            "Vision Tokens Per Step": self.vision_tokens_per_step,
            "PointCloud Tokens Per Step": self.point_cloud_tokens_per_step,
            "State Tokens Per Step": self.state_token_count,
            "Total Tokens Per Step": self.tokens_per_step,
            "Noise Scheduler": f"{self.config.noise_scheduler_type} ({self.num_inference_steps} steps)",
        }
        log_box("Diffusion-IDP3 Architecture", arch_info, icon="🧩")

    def conditional_sample(
        self,
        batch_size: int,
        global_cond: Optional[Tensor] = None,
        generator=None,
        noise: Tensor | None = None,
    ) -> Tensor:
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
        for timestep in self.noise_scheduler.timesteps:
            model_output = self.unet(
                sample,
                torch.full((batch_size,), timestep, dtype=torch.long, device=device),
                global_cond=global_cond,
            )

            step_kwargs = {"generator": generator}
            if "eta" in self.noise_scheduler.step.__code__.co_varnames:
                step_kwargs["eta"] = getattr(self.config, "ddim_eta", 0.0)

            step_out = self.noise_scheduler.step(model_output, timestep, sample, **step_kwargs)
            sample = getattr(step_out, "prev_sample", step_out)

        return sample
