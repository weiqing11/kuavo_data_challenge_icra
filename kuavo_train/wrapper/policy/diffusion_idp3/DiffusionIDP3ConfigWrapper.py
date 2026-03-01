"""Configuration wrapper for the diffusion_idp3 policy."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, TypeVar

import draccus
from huggingface_hub.constants import CONFIG_NAME
from omegaconf import DictConfig, ListConfig, OmegaConf

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamConfig, AdamWConfig
from kuavo_train.logger import logger

T = TypeVar("T", bound="DiffusionIDP3ConfigWrapper")


@PreTrainedConfig.register_subclass("diffusion_idp3")
@dataclass
class DiffusionIDP3ConfigWrapper(DiffusionConfig):
    """
    Purpose:
        Extend the base DiffusionConfig with custom fields for RGB + point cloud fusion.
    Inputs (constructor):
        Uses DiffusionConfig fields plus a `custom` dictionary.
    Outputs (constructor):
        None.
    """

    custom: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Purpose:
            Initialize config defaults, merge normalization mappings, and expand custom fields.
        Inputs:
            None.
        Outputs:
            None.
        """
        vision_backbone = self.vision_backbone
        noise_scheduler = self.noise_scheduler_type
        # Temporary override for parent checks.
        self.vision_backbone = "resnet18"
        self.noise_scheduler_type = "DDPM"
        super().__post_init__()
        self.vision_backbone = vision_backbone
        self.noise_scheduler_type = noise_scheduler

        default_map = {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
        merged = copy.deepcopy(default_map)
        merged.update(self.normalization_mapping)
        self.normalization_mapping = merged

        if isinstance(self.custom, (DictConfig, dict)):
            for k, v in self.custom.items():
                if not hasattr(self, k):
                    setattr(self, k, v)
                else:
                    raise ValueError(
                        f"Custom setting '{k}: {v}' conflicts with base config. Remove it from 'custom'."
                    )
        self._convert_omegaconf_fields()

    def _convert_omegaconf_fields(self) -> None:
        """
        Purpose:
            Convert OmegaConf containers to native Python types.
        Inputs:
            None.
        Outputs:
            None.
        """
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, (ListConfig, DictConfig)):
                converted = OmegaConf.to_container(val, resolve=True)
                setattr(self, f.name, converted)

    @property
    def image_features(self) -> dict[str, PolicyFeature]:
        """
        Purpose:
            Return RGB/visual input features.
        Inputs:
            None.
        Outputs:
            image_features: dict of PolicyFeature entries keyed by observation name.
        """
        return {
            key: ft
            for key, ft in self.input_features.items()
            if ft.type in (FeatureType.VISUAL, getattr(FeatureType, "RGB", FeatureType.VISUAL))
        }

    @property
    def depth_features(self) -> dict[str, PolicyFeature]:
        """
        Purpose:
            Return depth input features (typed as VISUAL in this project).
        Inputs:
            None.
        Outputs:
            depth_features: dict of PolicyFeature entries keyed by observation name.
        """
        return {
            key: ft
            for key, ft in self.input_features.items()
            if ft.type is getattr(FeatureType, "DEPTH", FeatureType.VISUAL)
        }

    def validate_features(self) -> None:
        """
        Purpose:
            Validate that required inputs (RGB, env state, or point cloud) are present and consistent.
        Inputs:
            None.
        Outputs:
            None.
        """
        use_point_cloud = getattr(self, "use_point_cloud", False)
        if len(self.image_features) == 0 and self.env_state_feature is None and not use_point_cloud:
            raise ValueError("You must provide at least one image, environment state, or point cloud input.")

        # Validate image shapes are consistent.
        if len(self.image_features) > 0:
            first_image_key, first_image_ft = next(iter(self.image_features.items()))
            if self.crop_shape is not None:
                if isinstance(self.crop_shape[0], (list, tuple)):
                    (x_start, x_end), (y_start, y_end) = self.crop_shape
                    for key, image_ft in self.image_features.items():
                        if x_start < 0 or x_end > image_ft.shape[1] or y_start < 0 or y_end > image_ft.shape[2]:
                            raise ValueError(
                                f"crop_shape {self.crop_shape} must fit within image shape {image_ft.shape} for {key}."
                            )
                else:
                    for key, image_ft in self.image_features.items():
                        if self.crop_shape[0] > image_ft.shape[1] or self.crop_shape[1] > image_ft.shape[2]:
                            raise ValueError(
                                f"crop_shape {self.crop_shape} must fit within image shape {image_ft.shape} for {key}."
                            )
            for key, image_ft in self.image_features.items():
                if image_ft.shape != first_image_ft.shape:
                    raise ValueError(f"Image shape mismatch: {key} vs {first_image_key}.")

        # Validate depth shapes when enabled.
        if getattr(self, "use_depth", False) and len(self.depth_features) > 0:
            first_depth_key, first_depth_ft = next(iter(self.depth_features.items()))
            for key, image_ft in self.depth_features.items():
                if image_ft.shape != first_depth_ft.shape:
                    raise ValueError(f"Depth shape mismatch: {key} vs {first_depth_key}.")

        # Validate point cloud keys if strict.
        if use_point_cloud and getattr(self, "strict_point_cloud_keys", False):
            pc_keys = getattr(self, "point_cloud_keys", [])
            for key in pc_keys:
                if key not in self.input_features:
                    raise ValueError(f"Point cloud key not found in input_features: {key}")

    def _save_pretrained(self, save_directory: Path) -> None:
        """
        Purpose:
            Save config to disk while removing expanded custom fields.
        Inputs:
            save_directory: Path to write config.
        Outputs:
            None.
        """
        cfg_copy = copy.deepcopy(self)
        if isinstance(cfg_copy.custom, dict):
            for k in list(cfg_copy.custom.keys()):
                if hasattr(cfg_copy, k):
                    delattr(cfg_copy, k)
        with open(save_directory / CONFIG_NAME, "w") as f, draccus.config_type("json"):
            draccus.dump(cfg_copy, f, indent=4)

    @classmethod
    def from_pretrained(
        cls: type[T],
        pretrained_name_or_path: str | Path,
        *,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **policy_kwargs,
    ) -> T:
        """
        Purpose:
            Load config from a pretrained directory or repo.
        Inputs:
            pretrained_name_or_path: path or repo ID.
        Outputs:
            config: DiffusionIDP3ConfigWrapper instance.
        """
        parent_cls = PreTrainedConfig
        return parent_cls.from_pretrained(
            pretrained_name_or_path,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
            **policy_kwargs,
        )

    def get_optimizer_preset(self):
        """
        Purpose:
            Return optimizer preset (Adam or AdamW) based on denoiser selection.
        Inputs:
            None.
        Outputs:
            optimizer_config: AdamConfig or AdamWConfig.
        """
        if getattr(self, "use_unet", False):
            logger.info("Using Adam optimizer for UNet.")
            return AdamConfig(
                lr=self.optimizer_lr,
                betas=self.optimizer_betas,
                eps=self.optimizer_eps,
                weight_decay=self.optimizer_weight_decay,
            )
        logger.info("Using AdamW optimizer for transformer/DiT.")
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )
