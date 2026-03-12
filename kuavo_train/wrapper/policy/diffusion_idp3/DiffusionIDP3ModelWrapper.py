"""Model wrapper for diffusion_idp3 with modular RGB/PointCloud/DiT backbones."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import einops
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from lerobot.policies.diffusion.modeling_diffusion import DiffusionModel
from lerobot.policies.utils import get_device_from_parameters, get_dtype_from_parameters
from lerobot.utils.constants import OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

from kuavo_train.logger import logger
from kuavo_train.wrapper.policy.diffusion_idp3.DiffusionIDP3ConfigWrapper import DiffusionIDP3ConfigWrapper
from kuavo_train.wrapper.policy.diffusion_idp3.action_generator.dit import (
    DiT,
    DiT_B,
    DiT_L,
    DiT_S,
    DiT_XL,
)
from kuavo_train.wrapper.policy.diffusion_idp3.fusion import Fusion
from kuavo_train.wrapper.policy.diffusion_idp3.point_cloud_encoders import IDP3MultiViewPointCloudEncoder
from kuavo_train.wrapper.policy.diffusion_idp3.rgb_encoders import SiglipRGBEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[4]


def _make_noise_scheduler_factory(name: str, **kwargs: dict[str, Any]) -> DDPMScheduler | DDIMScheduler:
    if name == "DDPM":
        return DDPMScheduler(**kwargs)
    if name == "DDIM":
        return DDIMScheduler(**kwargs)
    raise ValueError(f"Unsupported noise scheduler type: {name}")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class StateTokenEncoder(nn.Module):
    """Encode robot/env state into the token dimension."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(inplace=False),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 2:
            return self.net(x)
        if x.dim() == 3:
            batch_size, steps, dim = x.shape
            x_flat = x.reshape(batch_size * steps, dim)
            y = self.net(x_flat)
            return y.reshape(batch_size, steps, -1)
        raise ValueError("StateTokenEncoder expects 2D or 3D tensor.")


class DiffusionIDP3ModelWrapper(DiffusionModel):
    def __init__(self, config: DiffusionIDP3ConfigWrapper) -> None:
        nn.Module.__init__(self)
        self.config = config

        self.cond_feat_dim = int(getattr(config, "transformer_n_emb", 384))
        self.use_point_cloud = bool(getattr(config, "use_point_cloud", True))
        self.point_cloud_keys = list(getattr(config, "point_cloud_keys", []))
        self.strict_point_cloud_keys = bool(getattr(config, "strict_point_cloud_keys", True))
        self.initialize_from_pretrained = bool(getattr(config, "initialize_from_pretrained", True))
        self.pretrained_root = str(getattr(config, "pretrained_root", "dataset/models"))

        self.rgb_module_cfg = self._load_module_cfg(
            attr_name="rgb_encoder",
            default_rel_path="kuavo_train/wrapper/policy/diffusion_idp3/rgb_encoders/siglip/config.yaml",
            legacy=self._legacy_rgb_cfg(),
        )
        self.point_cloud_module_cfg = self._load_module_cfg(
            attr_name="point_cloud_encoder",
            default_rel_path="kuavo_train/wrapper/policy/diffusion_idp3/point_cloud_encoders/idp3/config.yaml",
            legacy=self._legacy_point_cloud_cfg(),
        )
        self.action_module_cfg = self._load_module_cfg(
            attr_name="action_generator",
            default_rel_path="kuavo_train/wrapper/policy/diffusion_idp3/action_generator/dit/config.yaml",
            legacy=self._legacy_action_cfg(),
        )
        self.cond_feat_dim = int(
            self.action_module_cfg.get("token_dim", self.rgb_module_cfg.get("target_dim", self.cond_feat_dim))
        )

        self.num_cameras = len(self.config.image_features)
        if self.use_point_cloud and len(self.point_cloud_keys) != self.num_cameras:
            raise ValueError(
                "For view-wise fusion, point_cloud_keys length must match RGB camera count. "
                f"Got {len(self.point_cloud_keys)} vs {self.num_cameras}."
            )

        self.rgb_encoder = self._build_rgb_encoder()
        self.rgb_patches_per_img = self.rgb_encoder.num_patches

        self.point_cloud_encoder = self._build_point_cloud_encoder() if self.use_point_cloud else None
        self.point_feature_dim = int(self.point_cloud_module_cfg["out_channels"]) if self.use_point_cloud else 0
        self.fusion = self._build_fusion_module() if self.use_point_cloud else None

        self.state_encoder = self._build_state_encoder()
        self.state_token_dim = self.cond_feat_dim

        # Basic FiLM keeps the original patch token count per view.
        self.tokens_per_camera = self.rgb_patches_per_img if self.rgb_encoder is not None else 1
        self.vision_seq_len = self.num_cameras * self.tokens_per_camera

        self.unet = self._build_action_generator()

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
            "DiffusionIDP3ModelWrapper initialized: "
            f"num_cameras={self.num_cameras}, rgb_patches_per_camera={self.rgb_patches_per_img}, "
            f"vision_seq_len={self.vision_seq_len}, initialize_from_pretrained={self.initialize_from_pretrained}"
        )

    def _legacy_rgb_cfg(self) -> dict[str, Any]:
        image_h, image_w = self._resolve_image_size()
        return {
            "type": "siglip",
            "overrides": {
                "pretrained_model_path": getattr(
                    self.config,
                    "siglip_model_path",
                    getattr(self.config, "siglip_model_name", "siglip2-base-patch16-224"),
                ),
                "pretrained_processor_path": getattr(
                    self.config,
                    "siglip_model_path",
                    getattr(self.config, "siglip_model_name", "siglip2-base-patch16-224"),
                ),
                "vision_encoder_mode": getattr(self.config, "vision_encoder_mode", "patches"),
                "target_dim": self.cond_feat_dim,
                "image_size": [image_h, image_w],
                "use_lora": bool(getattr(self.config, "use_lora", True)),
                "lora_rank": int(getattr(self.config, "lora_rank", 16)),
                "lora_alpha": int(getattr(self.config, "lora_alpha", 32)),
                "lora_dropout": float(getattr(self.config, "lora_dropout", 0.05)),
                "vision_freeze": bool(getattr(self.config, "vision_freeze", False)),
            },
        }

    def _legacy_point_cloud_cfg(self) -> dict[str, Any]:
        return {
            "type": "idp3_multi_stage_pointnet",
            "overrides": {
                "num_points": int(getattr(self.config, "point_cloud_num_points", 1024)),
                "in_channels": int(getattr(self.config, "point_cloud_in_channels", 6)),
                "out_channels": int(getattr(self.config, "point_cloud_out_channels", self.cond_feat_dim)),
                "hidden_dim": int(getattr(self.config, "point_cloud_hidden_dim", 128)),
                "num_layers": int(getattr(self.config, "point_cloud_num_layers", 4)),
                "downsample": bool(getattr(self.config, "point_cloud_downsample", True)),
            },
        }

    def _legacy_action_cfg(self) -> dict[str, Any]:
        return {
            "type": "diffusion_new_dit",
            "overrides": {
                "token_dim": int(getattr(self.config, "transformer_n_emb", 384)),
                "variant": getattr(self.config, "dit_variant", "DiT_S"),
                "hidden_size": int(getattr(self.config, "transformer_n_emb", 384)),
                "depth": int(getattr(self.config, "transformer_n_layer", 12)),
                "num_heads": int(getattr(self.config, "transformer_n_head", 6)),
                "mlp_ratio": float(getattr(self.config, "dit_mlp_ratio", 4.0)),
            },
        }

    def _resolve_image_size(self) -> tuple[int, int]:
        if self.config.resize_shape is not None:
            return int(self.config.resize_shape[0]), int(self.config.resize_shape[1])

        if self.config.crop_shape is not None:
            if isinstance(self.config.crop_shape[0], (list, tuple)):
                (x_start, x_end), (y_start, y_end) = self.config.crop_shape
                return int(x_end - x_start), int(y_end - y_start)
            return int(self.config.crop_shape[0]), int(self.config.crop_shape[1])

        first_feature = next(iter(self.config.image_features.values()))
        return int(first_feature.shape[1]), int(first_feature.shape[2])

    def _resolve_config_path(self, raw_path: str | None, default_rel_path: str) -> Path:
        if raw_path is None:
            return PROJECT_ROOT / default_rel_path
        path = Path(raw_path)
        return path if path.is_absolute() else PROJECT_ROOT / path

    def _load_module_cfg(self, attr_name: str, default_rel_path: str, legacy: dict[str, Any]) -> dict[str, Any]:
        raw_section = getattr(self.config, attr_name, {}) if hasattr(self.config, attr_name) else {}
        if isinstance(raw_section, DictConfig):
            section = OmegaConf.to_container(raw_section, resolve=True)
        elif isinstance(raw_section, dict):
            section = copy.deepcopy(raw_section)
        else:
            section = {}
        if not isinstance(section, dict):
            section = {}
        use_legacy_override = len(section) == 0

        config_path = self._resolve_config_path(section.get("config_path"), default_rel_path)
        if not config_path.exists():
            raise FileNotFoundError(f"{attr_name} config not found: {config_path}")

        file_cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
        merged = copy.deepcopy(file_cfg)
        if use_legacy_override:
            merged = _deep_merge(merged, legacy.get("overrides", {}))

        direct_overrides = {k: v for k, v in section.items() if k not in {"config_path", "overrides", "type"}}
        merged = _deep_merge(merged, direct_overrides)
        merged = _deep_merge(merged, section.get("overrides", {}) or {})

        if "type" in section:
            merged["type"] = section["type"]
        elif "type" in legacy:
            merged["type"] = legacy["type"]

        return merged

    def _build_rgb_encoder(self) -> SiglipRGBEncoder:
        encoder_type = self.rgb_module_cfg.get("type", "siglip")
        if encoder_type != "siglip":
            raise ValueError(f"Unsupported RGB encoder: {encoder_type}")

        rgb_cfg = copy.deepcopy(self.rgb_module_cfg)
        rgb_cfg["target_dim"] = self.cond_feat_dim
        image_h, image_w = self._resolve_image_size()
        rgb_cfg["image_size"] = [image_h, image_w]
        rgb_cfg["initialize_from_pretrained"] = self.initialize_from_pretrained
        rgb_cfg["pretrained_root"] = self.pretrained_root
        return SiglipRGBEncoder(rgb_cfg, target_dim=self.cond_feat_dim)

    def _build_point_cloud_encoder(self) -> IDP3MultiViewPointCloudEncoder:
        encoder_type = self.point_cloud_module_cfg.get("type", "idp3_multi_stage_pointnet")
        if encoder_type != "idp3_multi_stage_pointnet":
            raise ValueError(f"Unsupported point cloud encoder: {encoder_type}")

        point_cloud_encoder = IDP3MultiViewPointCloudEncoder(
            num_views=len(self.point_cloud_keys),
            num_points=int(self.point_cloud_module_cfg["num_points"]),
            in_channels=int(self.point_cloud_module_cfg["in_channels"]),
            out_channels=int(self.point_cloud_module_cfg["out_channels"]),
            hidden_dim=int(self.point_cloud_module_cfg.get("hidden_dim", 128)),
            num_layers=int(self.point_cloud_module_cfg.get("num_layers", 4)),
            downsample=bool(self.point_cloud_module_cfg.get("downsample", True)),
        )
        self._maybe_load_module_pretrained(
            module=point_cloud_encoder,
            module_cfg=self.point_cloud_module_cfg,
            module_name="point_cloud_encoder",
        )
        return point_cloud_encoder

    def _build_fusion_module(self) -> Fusion:
        return Fusion(token_dim=self.cond_feat_dim, point_dim=self.point_feature_dim)

    def _build_state_encoder(self) -> nn.Module:
        state_dim = int(self.config.robot_state_feature.shape[0])
        if self.config.env_state_feature is not None:
            state_dim += int(self.config.env_state_feature.shape[0])

        if bool(getattr(self.config, "use_state_encoder", True)):
            return StateTokenEncoder(state_dim, self.cond_feat_dim)

        if state_dim == self.cond_feat_dim:
            return nn.Identity()
        return nn.Linear(state_dim, self.cond_feat_dim)

    def _build_action_generator(self) -> nn.Module:
        generator_type = self.action_module_cfg.get("type", "diffusion_new_dit")
        if generator_type != "diffusion_new_dit":
            raise ValueError(f"Unsupported action generator: {generator_type}")

        common_kwargs = {
            "action_dim": self.config.output_features["action"].shape[0],
            "action_seq_len": self.config.horizon,
            "n_obs_steps": self.config.n_obs_steps,
            "token_dim": self.cond_feat_dim,
            "max_image_tokens": self.vision_seq_len,
            "num_cameras": self.num_cameras,
        }

        variant = self.action_module_cfg.get("variant", "DiT_S")
        factory = {
            "DiT_XL": DiT_XL,
            "DiT_L": DiT_L,
            "DiT_B": DiT_B,
            "DiT_S": DiT_S,
        }

        if variant in factory:
            action_generator = factory[variant](**common_kwargs)
            self._maybe_load_module_pretrained(
                module=action_generator,
                module_cfg=self.action_module_cfg,
                module_name="action_generator",
            )
            return action_generator

        if variant == "custom":
            action_generator = DiT(
                hidden_size=int(self.action_module_cfg.get("hidden_size", self.cond_feat_dim)),
                depth=int(self.action_module_cfg.get("depth", 12)),
                num_heads=int(self.action_module_cfg.get("num_heads", 6)),
                mlp_ratio=float(self.action_module_cfg.get("mlp_ratio", 4.0)),
                **common_kwargs,
            )
            self._maybe_load_module_pretrained(
                module=action_generator,
                module_cfg=self.action_module_cfg,
                module_name="action_generator",
            )
            return action_generator

        raise ValueError(f"Unknown DiT variant: {variant}")

    def _resolve_local_pretrained_path(self, raw_path: str | None) -> Path | None:
        if raw_path is None or str(raw_path).strip() == "":
            return None

        path = Path(raw_path)
        if path.is_absolute():
            return path

        root = Path(self.pretrained_root)
        if not root.is_absolute():
            root = PROJECT_ROOT / root

        if str(path).startswith("dataset/models"):
            return PROJECT_ROOT / path
        return root / path

    def _maybe_load_module_pretrained(self, module: nn.Module, module_cfg: dict[str, Any], module_name: str) -> None:
        pretrained_path = self._resolve_local_pretrained_path(module_cfg.get("pretrained_weight_path"))

        if pretrained_path is None:
            logger.info(f"{module_name}: no pretrained_weight_path configured, skip loading.")
            return

        if not self.initialize_from_pretrained:
            logger.info(f"{module_name}: initialize_from_pretrained=False, skip loading {pretrained_path}.")
            return

        if not pretrained_path.exists():
            raise FileNotFoundError(f"{module_name} pretrained weight not found: {pretrained_path}")

        state_dict_or_ckpt = torch.load(pretrained_path, map_location="cpu")
        if isinstance(state_dict_or_ckpt, dict):
            if isinstance(state_dict_or_ckpt.get("state_dict"), dict):
                state_dict = state_dict_or_ckpt["state_dict"]
            elif isinstance(state_dict_or_ckpt.get("model_state_dict"), dict):
                state_dict = state_dict_or_ckpt["model_state_dict"]
            else:
                state_dict = state_dict_or_ckpt
        else:
            raise ValueError(f"{module_name} pretrained file must contain a dict: {pretrained_path}")

        missing_keys, unexpected_keys = module.load_state_dict(state_dict, strict=False)
        logger.info(
            f"{module_name}: loaded pretrained weights from {pretrained_path} "
            f"(missing={len(missing_keys)}, unexpected={len(unexpected_keys)})."
        )

    def _get_state_tensor(self, batch: dict[str, Tensor]) -> Tensor:
        state = batch[OBS_STATE]
        if OBS_ENV_STATE in batch:
            state = torch.cat([state, batch[OBS_ENV_STATE]], dim=-1)
        return state

    def _encode_rgb_tokens(self, images: Tensor) -> Tensor:
        """Args: images [B,S,V,C,H,W]. Returns: [B*S,V,P,D]."""
        batch_size, obs_steps, num_views = images.shape[:3]
        if num_views != self.num_cameras:
            raise ValueError(f"Image view count {num_views} != expected {self.num_cameras}")

        rgb_flat = einops.rearrange(images, "b s v c h w -> (b s v) c h w")
        rgb_tokens = self.rgb_encoder(rgb_flat)
        return einops.rearrange(
            rgb_tokens,
            "(b s v) p d -> (b s) v p d",
            b=batch_size,
            s=obs_steps,
            v=self.num_cameras,
        )

    def _encode_point_tokens(self, batch: dict[str, Tensor]) -> Tensor | None:
        if not self.use_point_cloud:
            return None

        if self.strict_point_cloud_keys:
            for key in self.point_cloud_keys:
                if key not in batch:
                    raise KeyError(f"Missing point cloud key in batch: {key}")

        point_clouds = [batch[key] for key in self.point_cloud_keys]
        return self.point_cloud_encoder(point_clouds)

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        """Return token conditioning [B, S, T, D]."""
        batch_size, obs_steps = batch[OBS_STATE].shape[:2]

        if OBS_IMAGES not in batch:
            raise ValueError(f"Missing required key `{OBS_IMAGES}` in batch.")

        rgb_tokens = self._encode_rgb_tokens(batch[OBS_IMAGES])
        point_tokens = self._encode_point_tokens(batch)

        if self.use_point_cloud:
            if self.fusion is None:
                raise RuntimeError("Point-cloud fusion is enabled but fusion module is not initialized.")
            fused_view_tokens = self.fusion(rgb_tokens, point_tokens)  # [BS, V, P, D]
        else:
            fused_view_tokens = rgb_tokens

        fused_tokens = fused_view_tokens.reshape(batch_size, obs_steps, -1, self.cond_feat_dim)

        state_token = self.state_encoder(self._get_state_tensor(batch)).unsqueeze(2)
        return torch.cat([fused_tokens, state_token], dim=2)

    def conditional_sample(
        self,
        batch_size: int,
        global_cond: Tensor | None = None,
        generator: torch.Generator | None = None,
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

            step_kwargs: dict[str, Any] = {"generator": generator}
            if "eta" in self.noise_scheduler.step.__code__.co_varnames:
                step_kwargs["eta"] = float(getattr(self.config, "ddim_eta", 0.0))

            step_out = self.noise_scheduler.step(model_output, timestep, sample, **step_kwargs)
            sample = getattr(step_out, "prev_sample", step_out)

        return sample
