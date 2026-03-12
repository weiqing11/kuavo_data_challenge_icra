"""Configuration wrapper for diffusion_idp3 policy."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

import draccus
from huggingface_hub.constants import CONFIG_NAME
from omegaconf import DictConfig, ListConfig, OmegaConf

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig

T = TypeVar("T", bound="DiffusionIDP3ConfigWrapper")


@PreTrainedConfig.register_subclass("diffusion_idp3")
@dataclass
class DiffusionIDP3ConfigWrapper(DiffusionConfig):
    """Diffusion config with RGB + point-cloud + DiT extensions."""

    custom: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Bypass parent strict checks; this policy uses custom backbones.
        vision_backbone = self.vision_backbone
        noise_scheduler = self.noise_scheduler_type
        self.vision_backbone = "resnet18"
        self.noise_scheduler_type = "DDPM"
        super().__post_init__()
        self.vision_backbone = vision_backbone
        self.noise_scheduler_type = noise_scheduler

        default_norm = {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
        merged = copy.deepcopy(default_norm)
        merged.update(self.normalization_mapping)
        self.normalization_mapping = merged

        if isinstance(self.custom, (DictConfig, dict)):
            for key, value in self.custom.items():
                if hasattr(self, key):
                    raise ValueError(f"Custom setting '{key}' conflicts with base config field.")
                setattr(self, key, value)

        self._convert_omegaconf_fields()

    def _convert_omegaconf_fields(self) -> None:
        for field_def in fields(self):
            value = getattr(self, field_def.name)
            if isinstance(value, (ListConfig, DictConfig)):
                setattr(self, field_def.name, OmegaConf.to_container(value, resolve=True))

    @property
    def image_features(self) -> dict[str, PolicyFeature]:
        return {
            key: feature
            for key, feature in self.input_features.items()
            if feature.type in (FeatureType.VISUAL, getattr(FeatureType, "RGB", FeatureType.VISUAL))
        }

    def validate_features(self) -> None:
        if len(self.image_features) == 0:
            raise ValueError("diffusion_idp3 requires RGB inputs.")

        use_point_cloud = bool(getattr(self, "use_point_cloud", True))
        if not use_point_cloud:
            raise ValueError("diffusion_idp3 requires point cloud input (`use_point_cloud=true`).")

        point_cloud_keys = list(getattr(self, "point_cloud_keys", []))
        if len(point_cloud_keys) == 0:
            raise ValueError("`point_cloud_keys` must not be empty for diffusion_idp3.")

        if len(point_cloud_keys) != len(self.image_features):
            raise ValueError(
                "For view-wise fusion, `point_cloud_keys` length must match number of RGB cameras. "
                f"Got {len(point_cloud_keys)} vs {len(self.image_features)}."
            )

        if bool(getattr(self, "strict_point_cloud_keys", True)):
            for key in point_cloud_keys:
                if key not in self.input_features:
                    raise ValueError(f"Point cloud key not found in input_features: {key}")

        first_key, first_feature = next(iter(self.image_features.items()))
        for key, feature in self.image_features.items():
            if feature.shape != first_feature.shape:
                raise ValueError(f"Image shape mismatch: {key} vs {first_key}")

        if self.crop_shape is not None:
            if isinstance(self.crop_shape[0], (list, tuple)):
                (x_start, x_end), (y_start, y_end) = self.crop_shape
                for key, feature in self.image_features.items():
                    if x_start < 0 or x_end > feature.shape[1] or y_start < 0 or y_end > feature.shape[2]:
                        raise ValueError(
                            f"crop_shape {self.crop_shape} must fit image shape {feature.shape} for {key}."
                        )
            else:
                for key, feature in self.image_features.items():
                    if self.crop_shape[0] > feature.shape[1] or self.crop_shape[1] > feature.shape[2]:
                        raise ValueError(
                            f"crop_shape {self.crop_shape} must fit image shape {feature.shape} for {key}."
                        )

    def _save_pretrained(self, save_directory: Path) -> None:
        cfg_copy = copy.deepcopy(self)
        if isinstance(cfg_copy.custom, dict):
            for key in list(cfg_copy.custom.keys()):
                if hasattr(cfg_copy, key):
                    delattr(cfg_copy, key)
        with open(save_directory / CONFIG_NAME, "w") as file, draccus.config_type("json"):
            draccus.dump(cfg_copy, file, indent=4)

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
        local_path = Path(pretrained_name_or_path)
        if not local_path.exists():
            raise ValueError(
                "diffusion_idp3 config can only be loaded from local paths. "
                f"Path not found: {pretrained_name_or_path}"
            )

        return PreTrainedConfig.from_pretrained(
            local_path,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            token=token,
            cache_dir=cache_dir,
            local_files_only=True,
            revision=revision,
            **policy_kwargs,
        )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )
