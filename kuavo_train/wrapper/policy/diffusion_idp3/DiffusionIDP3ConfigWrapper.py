from __future__ import annotations

import copy
from copy import deepcopy
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

import draccus
from huggingface_hub.constants import CONFIG_NAME
from omegaconf import DictConfig, ListConfig, OmegaConf

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamConfig, AdamWConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig


T = TypeVar("T", bound="DiffusionIDP3ConfigWrapper")


@PreTrainedConfig.register_subclass("diffusion_idp3")
@dataclass
class DiffusionIDP3ConfigWrapper(DiffusionConfig):
    custom: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Keep parent validations while bypassing strict backbone/scheduler checks in parent constructor.
        vision_backbone = self.vision_backbone
        self.vision_backbone = "resnet18"
        noise_scheduler = self.noise_scheduler_type
        self.noise_scheduler_type = "DDPM"
        super().__post_init__()
        self.noise_scheduler_type = noise_scheduler
        self.vision_backbone = vision_backbone

        default_map = {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
        merged = copy.deepcopy(default_map)
        merged.update(self.normalization_mapping)
        self.normalization_mapping = merged

        if isinstance(self.custom, (DictConfig, dict)):
            for key, value in self.custom.items():
                if hasattr(self, key):
                    raise ValueError(
                        f"Custom setting `{key}: {value}` conflicts with base configuration fields."
                    )
                setattr(self, key, value)

        self._set_point_cloud_defaults()
        self._convert_omegaconf_fields()

    def _set_point_cloud_defaults(self) -> None:
        if not hasattr(self, "use_point_cloud"):
            self.use_point_cloud = True
        if not hasattr(self, "point_cloud_keys"):
            self.point_cloud_keys = ["observation.pc_h", "observation.pc_l", "observation.pc_r"]
        if not hasattr(self, "point_cloud_encoder_type"):
            self.point_cloud_encoder_type = "idp3_multi_stage_pointnet"
        if not hasattr(self, "strict_point_cloud_keys"):
            self.strict_point_cloud_keys = True
        if not hasattr(self, "point_cloud_downsample"):
            self.point_cloud_downsample = False
        if not hasattr(self, "point_cloud_out_channels"):
            self.point_cloud_out_channels = 128
        if not hasattr(self, "point_cloud_project_dim"):
            self.point_cloud_project_dim = getattr(self, "transformer_n_emb", 384)

        keys = list(getattr(self, "point_cloud_keys", []))
        if len(keys) > 0:
            first_key = keys[0]
            if first_key in self.input_features:
                first_ft = self.input_features[first_key]
                if not hasattr(self, "point_cloud_num_points"):
                    self.point_cloud_num_points = first_ft.shape[0]
                if not hasattr(self, "point_cloud_in_channels"):
                    self.point_cloud_in_channels = first_ft.shape[1]

        if not hasattr(self, "point_cloud_num_points"):
            self.point_cloud_num_points = 4096
        if not hasattr(self, "point_cloud_in_channels"):
            self.point_cloud_in_channels = 6

    def _convert_omegaconf_fields(self):
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, (ListConfig, DictConfig)):
                setattr(self, f.name, OmegaConf.to_container(value, resolve=True))

        # convert dynamically injected custom fields as well
        for key, value in list(self.__dict__.items()):
            if isinstance(value, (ListConfig, DictConfig)):
                setattr(self, key, OmegaConf.to_container(value, resolve=True))

    @property
    def image_features(self) -> dict[str, PolicyFeature]:
        rgb_type = getattr(FeatureType, "RGB", None)
        return {
            key: ft
            for key, ft in self.input_features.items()
            if (ft.type is FeatureType.VISUAL) or (rgb_type is not None and ft.type is rgb_type)
        }

    @property
    def depth_features(self) -> dict[str, PolicyFeature]:
        depth_type = getattr(FeatureType, "DEPTH", None)
        if depth_type is None:
            return {}
        return {key: ft for key, ft in self.input_features.items() if ft.type is depth_type}

    def validate_features(self) -> None:
        has_point_cloud = bool(getattr(self, "use_point_cloud", True) and len(getattr(self, "point_cloud_keys", [])) > 0)
        if len(self.image_features) == 0 and self.env_state_feature is None and not has_point_cloud:
            raise ValueError("You must provide image features, environment state, or point-cloud features.")

        if self.crop_shape is not None:
            if isinstance(self.crop_shape[0], (list, tuple)):
                (x_start, x_end), (y_start, y_end) = self.crop_shape
                for key, image_ft in self.image_features.items():
                    if x_start < 0 or x_end > image_ft.shape[1] or y_start < 0 or y_end > image_ft.shape[2]:
                        raise ValueError(
                            f"`crop_shape` {self.crop_shape} must fit image shape {image_ft.shape} ({key})."
                        )
            else:
                for key, image_ft in self.image_features.items():
                    if self.crop_shape[0] > image_ft.shape[1] or self.crop_shape[1] > image_ft.shape[2]:
                        raise ValueError(
                            f"`crop_shape` {self.crop_shape} must fit image shape {image_ft.shape} ({key})."
                        )

        if len(self.image_features) > 0:
            first_image_key, first_image_ft = next(iter(self.image_features.items()))
            for key, image_ft in self.image_features.items():
                if image_ft.shape != first_image_ft.shape:
                    raise ValueError(
                        f"`{key}` shape {image_ft.shape} does not match `{first_image_key}` shape {first_image_ft.shape}."
                    )

        if len(self.depth_features) > 0:
            first_depth_key, first_depth_ft = next(iter(self.depth_features.items()))
            for key, depth_ft in self.depth_features.items():
                if depth_ft.shape != first_depth_ft.shape:
                    raise ValueError(
                        f"`{key}` shape {depth_ft.shape} does not match `{first_depth_key}` shape {first_depth_ft.shape}."
                    )

        if getattr(self, "use_point_cloud", True):
            pc_keys = list(getattr(self, "point_cloud_keys", []))
            if len(pc_keys) == 0:
                raise ValueError("`use_point_cloud=True` requires non-empty `point_cloud_keys`.")

            missing = [key for key in pc_keys if key not in self.input_features]
            if missing and getattr(self, "strict_point_cloud_keys", True):
                raise ValueError(
                    f"Point-cloud keys missing in input features: {missing}. "
                    f"Available keys: {sorted(self.input_features.keys())}"
                )

            if not missing:
                for key in pc_keys:
                    shape = self.input_features[key].shape
                    if len(shape) != 2:
                        raise ValueError(
                            f"Point-cloud key `{key}` must have shape `(N, C)`, got {shape}."
                        )

    def _save_pretrained(self, save_directory: Path) -> None:
        cfg_copy = deepcopy(self)
        if isinstance(cfg_copy.custom, dict):
            for key in list(cfg_copy.custom.keys()):
                if hasattr(cfg_copy, key):
                    delattr(cfg_copy, key)

        with open(save_directory / CONFIG_NAME, "w") as handle, draccus.config_type("json"):
            draccus.dump(cfg_copy, handle, indent=4)

    @classmethod
    def from_pretrained(
        cls: type[T],
        pretrained_name_or_path: str | Path,
        *,
        force_download: bool = False,
        resume_download: bool = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **policy_kwargs,
    ) -> T:
        return PreTrainedConfig.from_pretrained(
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
        if getattr(self, "use_unet", False):
            return AdamConfig(
                lr=self.optimizer_lr,
                betas=self.optimizer_betas,
                eps=self.optimizer_eps,
                weight_decay=self.optimizer_weight_decay,
            )

        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )
