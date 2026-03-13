"""Policy wrapper for diffusion_idp3."""

from __future__ import annotations

import builtins
from collections import deque
from pathlib import Path
from typing import TypeVar

import torch
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from torch import Tensor

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

from kuavo_train.utils.augmenter import crop_image, resize_image
from kuavo_train.wrapper.policy.diffusion_idp3.DiffusionIDP3ConfigWrapper import DiffusionIDP3ConfigWrapper
from kuavo_train.wrapper.policy.diffusion_idp3.DiffusionIDP3ModelWrapper import DiffusionIDP3ModelWrapper

T = TypeVar("T", bound="DiffusionIDP3PolicyWrapper")


class DiffusionIDP3PolicyWrapper(DiffusionPolicy):
    config_class = DiffusionIDP3ConfigWrapper
    name = "diffusion_idp3"

    def __init__(self, config: DiffusionIDP3ConfigWrapper):
        # Build parent class safely, then replace its diffusion model.
        vision_backbone = config.vision_backbone
        noise_scheduler = config.noise_scheduler_type
        config.vision_backbone = "resnet18"
        config.noise_scheduler_type = "DDPM"
        super().__init__(config)
        config.vision_backbone = vision_backbone
        config.noise_scheduler_type = noise_scheduler

        self.diffusion = DiffusionIDP3ModelWrapper(config)
        self.config = config

    def reset(self) -> None:
        self._queues = {
            OBS_STATE: deque(maxlen=self.config.n_obs_steps),
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

        if self.config.image_features:
            self._queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)

        if self.config.env_state_feature:
            self._queues[OBS_ENV_STATE] = deque(maxlen=self.config.n_obs_steps)

        if bool(getattr(self.config, "use_point_cloud", True)):
            for key in getattr(self.config, "point_cloud_keys", []):
                self._queues[key] = deque(maxlen=self.config.n_obs_steps)

    def _prepare_rgb(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        random_crop = self.config.crop_is_random and self.training
        prepared = dict(batch)

        for key in self.config.image_features:
            prepared[key], _ = crop_image(
                prepared[key],
                target_range=self.config.crop_shape,
                random_crop=random_crop,
            )
            prepared[key] = resize_image(
                prepared[key],
                target_size=self.config.resize_shape,
                image_type="rgb",
            )

        prepared[OBS_IMAGES] = torch.stack([prepared[key] for key in self.config.image_features], dim=-4)
        return prepared

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None, episode: int = 0, step: int = 0) -> Tensor:
        if ACTION in batch:
            batch.pop(ACTION)

        batch = self._prepare_rgb(batch)
        self._queues = populate_queues(self._queues, batch)

        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch, noise=noise)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        return self._queues[ACTION].popleft()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        batch = self._prepare_rgb(batch)
        loss = self.diffusion.compute_loss(batch)
        return loss, None

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: DiffusionIDP3ConfigWrapper | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = False,
        initialize_from_pretrained: bool | None = None,
        **kwargs,
    ) -> T:
        checkpoint_path = Path(pretrained_name_or_path)
        if not checkpoint_path.is_dir():
            raise ValueError(
                "diffusion_idp3 only supports loading from local checkpoint directories. "
                f"Got: {pretrained_name_or_path}"
            )

        if config is None:
            config = DiffusionIDP3ConfigWrapper.from_pretrained(
                pretrained_name_or_path=checkpoint_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=True,
                revision=revision,
                **kwargs,
            )

        if initialize_from_pretrained is None:
            # Default inference behavior: checkpoint-only mode.
            if getattr(config, "initialize_from_pretrained", False):
                config.initialize_from_pretrained = False
        else:
            config.initialize_from_pretrained = bool(initialize_from_pretrained)

        instance = cls(config, **kwargs)
        model_file = checkpoint_path / SAFETENSORS_SINGLE_FILE
        if not model_file.exists():
            raise FileNotFoundError(f"Checkpoint weight file not found: {model_file}")
        policy = cls._load_as_safetensor(instance, str(model_file), config.device, strict)

        policy.eval()
        return policy
