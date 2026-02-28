from __future__ import annotations

import builtins
import os
from collections import deque
from pathlib import Path
from typing import TypeVar

import torch
import torchvision.transforms.functional
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from huggingface_hub.errors import HfHubHTTPError
from torch import Tensor

from kuavo_train.utils.augmenter import crop_image, resize_image
from kuavo_train.wrapper.policy.diffusion_idp3.DiffusionIDP3ConfigWrapper import (
    DiffusionIDP3ConfigWrapper,
)
from kuavo_train.wrapper.policy.diffusion_idp3.DiffusionIDP3ModelWrapper import (
    DiffusionIDP3ModelWrapper,
)
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import ACTION, OBS_IMAGES


OBS_DEPTH = "observation.depth"
T = TypeVar("T", bound="DiffusionIDP3PolicyWrapper")


class DiffusionIDP3PolicyWrapper(DiffusionPolicy):
    def __init__(self, config: DiffusionIDP3ConfigWrapper):
        # `DiffusionPolicy.__init__` calls `self.reset()`, so subclass fields used in `reset`
        # must be initialized before `super().__init__`.
        self.point_cloud_keys = list(getattr(config, "point_cloud_keys", []))

        # Parent init compatibility hack (same pattern as diffusion_new wrapper).
        vision_backbone = config.vision_backbone
        config.vision_backbone = "resnet18"
        noise_scheduler = config.noise_scheduler_type
        config.noise_scheduler_type = "DDPM"
        super().__init__(config)
        config.vision_backbone = vision_backbone
        config.noise_scheduler_type = noise_scheduler

        self.diffusion = DiffusionIDP3ModelWrapper(config)
        self.point_cloud_keys = list(getattr(self.config, "point_cloud_keys", []))

    def reset(self):
        """Clear observation/action queues. Should be called on env reset."""
        self._queues = {
            "observation.state": deque(maxlen=self.config.n_obs_steps),
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)
        if getattr(self.config, "use_depth", False) and self.config.depth_features:
            self._queues[OBS_DEPTH] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues["observation.environment_state"] = deque(maxlen=self.config.n_obs_steps)
        if getattr(self.config, "use_point_cloud", True):
            point_cloud_keys = getattr(self, "point_cloud_keys", list(getattr(self.config, "point_cloud_keys", [])))
            for key in point_cloud_keys:
                self._queues[key] = deque(maxlen=self.config.n_obs_steps)

    @torch.no_grad()
    def select_action(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
        episode: int = 0,
        step: int = 0,
    ) -> Tensor:
        if ACTION in batch:
            batch = dict(batch)
            batch.pop(ACTION)

        random_crop = self.config.crop_is_random and self.training
        crop_position_list = []

        if self.config.image_features:
            batch = dict(batch)
            for key in self.config.image_features:
                batch[key], crop_position = crop_image(
                    batch[key],
                    target_range=self.config.crop_shape,
                    random_crop=random_crop,
                )
                crop_position_list.append(crop_position)
                batch[key] = resize_image(batch[key], target_size=self.config.resize_shape, image_type="rgb")
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)

        if getattr(self.config, "use_depth", False) and self.config.depth_features:
            batch = dict(batch)
            if len(crop_position_list) == 0:
                for key in self.config.depth_features:
                    batch[key] = resize_image(
                        batch[key],
                        target_size=self.config.resize_shape,
                        image_type="depth",
                    )
            else:
                for key, crop_position in zip(self.config.depth_features, crop_position_list, strict=True):
                    if len(crop_position) == 4:
                        batch[key] = torchvision.transforms.functional.crop(batch[key], *crop_position)
                    else:
                        batch[key] = torchvision.transforms.functional.center_crop(batch[key], crop_position)
                    batch[key] = resize_image(batch[key], target_size=self.config.resize_shape, image_type="depth")
            batch[OBS_DEPTH] = torch.stack([batch[key] for key in self.config.depth_features], dim=-4)

        self._queues = populate_queues(self._queues, batch)

        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch, noise=noise)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        return self._queues[ACTION].popleft()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        random_crop = self.config.crop_is_random and self.training
        crop_position = None

        if self.config.image_features:
            batch = dict(batch)
            for key in self.config.image_features:
                batch[key], crop_position = crop_image(
                    batch[key],
                    target_range=self.config.crop_shape,
                    random_crop=random_crop,
                )
                batch[key] = resize_image(batch[key], target_size=self.config.resize_shape, image_type="rgb")

        if getattr(self.config, "use_depth", False) and self.config.depth_features:
            batch = dict(batch)
            for key in self.config.depth_features:
                if crop_position is not None:
                    if len(crop_position) == 4:
                        batch[key] = torchvision.transforms.functional.crop(batch[key], *crop_position)
                    else:
                        batch[key] = torchvision.transforms.functional.center_crop(batch[key], crop_position)
                batch[key] = resize_image(batch[key], target_size=self.config.resize_shape, image_type="depth")

        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)

        if getattr(self.config, "use_depth", False) and self.config.depth_features:
            batch = dict(batch)
            batch[OBS_DEPTH] = torch.stack([batch[key] for key in self.config.depth_features], dim=-4)

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
        **kwargs,
    ) -> T:
        if config is None:
            config = DiffusionIDP3ConfigWrapper.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )

        model_id = str(pretrained_name_or_path)
        instance = cls(config, **kwargs)
        if os.path.isdir(model_id):
            model_file = os.path.join(model_id, SAFETENSORS_SINGLE_FILE)
            policy = cls._load_as_safetensor(instance, model_file, config.device, strict)
        else:
            try:
                model_file = hf_hub_download(
                    repo_id=model_id,
                    filename=SAFETENSORS_SINGLE_FILE,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
                policy = cls._load_as_safetensor(instance, model_file, config.device, strict)
            except HfHubHTTPError as error:
                raise FileNotFoundError(
                    f"{SAFETENSORS_SINGLE_FILE} not found on the HuggingFace Hub in {model_id}"
                ) from error

        policy.to(config.device)
        policy.eval()
        return policy
