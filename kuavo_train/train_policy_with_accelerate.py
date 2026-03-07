import json

import lerobot_patches.custom_patches  # Ensure custom patches are applied, DON'T REMOVE THIS LINE!
from lerobot.configs.policies import PolicyFeature
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
from pathlib import Path
from functools import partial

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import shutil
import accelerate
from accelerate.utils import DistributedDataParallelKwargs
from hydra.utils import instantiate
# from diffusers.optimization import get_scheduler

from lerobot.configs.types import FeatureType, NormalizationMode
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata, LeRobotDataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.utils.random_utils import set_seed
from lerobot.policies.factory import make_pre_post_processors
from kuavo_train.wrapper.policy.diffusion_new.DiffusionPolicyWrapper import CustomDiffusionPolicyWrapper
from kuavo_train.wrapper.policy.diffusion_idp3.DiffusionIDP3PolicyWrapper import (
    DiffusionIDP3PolicyWrapper,
)
#from kuavo_train.wrapper.policy.diffusion.DiffusionPolicyWrapper import CustomDiffusionPolicyWrapper
from kuavo_train.wrapper.policy.act.ACTPolicyWrapper import CustomACTPolicyWrapper
from kuavo_train.wrapper.policy.idp3.modeling_idp3 import IDP3Policy
from kuavo_train.wrapper.dataset.LeRobotDatasetWrapper import CustomLeRobotDataset
from kuavo_train.utils.augmenter import crop_image, resize_image, DeterministicAugmenterColor
from kuavo_train.utils.utils import save_rng_state, load_rng_state
from lerobot.policies.act.modeling_act import ACTPolicy
from diffusers.optimization import get_scheduler
from utils.transforms import ImageTransforms, ImageTransformsConfig, ImageTransformConfig

from functools import partial
from contextlib import nullcontext
from lerobot.processor import ProcessorStep, NormalizerProcessorStep
from lerobot.processor.core import TransitionKey
from lerobot.configs.types import PipelineFeatureType, PolicyFeature

from logger import logger, log_box, Progress
# 禁用安全检查
_original_load = torch.load

def unsafe_load(*args, **kwargs):
    kwargs['weights_only'] = False 
    return _original_load(*args, **kwargs)

torch.load = unsafe_load

def build_augmenter(cfg):
    """Since operations such as cropping and resizing in LeRobot are implemented at the model level 
    rather than at the data level, we provide only RGB image augmentations on the data side here, 
    with support for customization. For more details, refer to configs/policy/diffusion_config.yaml. 
    To define custom transformations, please see utils.transforms.py."""

    img_tf_cfg = ImageTransformsConfig(
        enable=cfg.get("enable", False),
        max_num_transforms=cfg.get("max_num_transforms", 3),
        random_order=cfg.get("random_order", False),
        tfs={}
    )

    # deal tfs part
    if "tfs" in cfg:
        for name, tf_dict in cfg["tfs"].items():
            img_tf_cfg.tfs[name] = ImageTransformConfig(
                weight=tf_dict.get("weight", 1.0),
                type=tf_dict.get("type", "Identity"),
                kwargs=tf_dict.get("kwargs", {}),
            )
    return ImageTransforms(img_tf_cfg)


def build_delta_timestamps(dataset_metadata, policy_cfg):
    """Build delta timestamps for observations and actions."""
    obs_indices = getattr(policy_cfg, "observation_delta_indices", None)
    act_indices = getattr(policy_cfg, "action_delta_indices", None)
    if obs_indices is None and act_indices is None:
        return None

    delta_timestamps = {}
    for key in dataset_metadata.info["features"]:
        if "observation" in key and obs_indices is not None:
            delta_timestamps[key] = [i / dataset_metadata.fps for i in obs_indices]
        elif "action" in key and act_indices is not None:
            delta_timestamps[key] = [i / dataset_metadata.fps for i in act_indices]

    return delta_timestamps if delta_timestamps else None


def build_optimizer_and_scheduler(policy, cfg, total_frames, accelerator):
    """Return optimizer and scheduler."""
    optimizer = policy.config.get_optimizer_preset().build([p for p in policy.parameters() if p.requires_grad])
    # If `max_training_step` is specified, it takes precedence; 
    # otherwise, the value is automatically determined based on `max_epoch`.
    if cfg.training.max_training_step is None:
        effective_batch_size = cfg.training.batch_size * accelerator.num_processes
        # updates_per_epoch = (total_frames // (cfg.training.batch_size * cfg.training.accumulation_steps)) + 1
        updates_per_epoch = max(1, total_frames // (effective_batch_size * cfg.training.accumulation_steps))
        num_training_steps = cfg.training.max_epoch * updates_per_epoch
    else:
        num_training_steps = cfg.training.max_training_step
    lr_scheduler = policy.config.get_scheduler_preset()
    if lr_scheduler is not None:
        lr_scheduler = lr_scheduler.build(optimizer, num_training_steps)
    else:
        lr_scheduler = get_scheduler(
            name=cfg.training.scheduler_name,
            optimizer=optimizer,
            num_warmup_steps=cfg.training.scheduler_warmup_steps,
            num_training_steps=num_training_steps,
        )

    # or you can set your optimizer and lr_scheduler here and replace it.
    return optimizer, lr_scheduler

def build_policy(name, policy_cfg):
    policy = {
        "diffusion": CustomDiffusionPolicyWrapper,
        "diffusion_idp3": DiffusionIDP3PolicyWrapper,
        "act": CustomACTPolicyWrapper,
        "idp3": IDP3Policy,
    }[name](policy_cfg)
    return policy

def build_policy_config(cfg, input_features, output_features):
    def _normalize_feature_dict(d: Any) -> dict[str, PolicyFeature]:
        if isinstance(d, DictConfig):
            d = OmegaConf.to_container(d, resolve=True)
        if not isinstance(d, dict):
            raise TypeError(f"Expected dict or DictConfig, got {type(d)}")

        return {
            k: PolicyFeature(**v) if isinstance(v, dict) and not isinstance(v, PolicyFeature) else v
            for k, v in d.items()
        }

    policy_cfg = instantiate(
        cfg.policy,
        input_features=input_features,
        output_features=output_features,
        device=cfg.training.device,
    )
                
    policy_cfg.input_features = _normalize_feature_dict(policy_cfg.input_features)
    policy_cfg.output_features = _normalize_feature_dict(policy_cfg.output_features)
    return policy_cfg

def sanitize_policy_config(policy):
    """
    在保存前，递归地将 policy.config 中的 OmegaConf 对象转换为原生 Python 类型 (list, dict, tuple)。
    解决 json.dump 或 draccus 无法序列化 ListConfig/DictConfig 的问题。
    """
    if not hasattr(policy, 'config') or policy.config is None:
        return

    # 遍历 config 的所有属性
    for key, value in policy.config.__dict__.items():
        if isinstance(value, (DictConfig, ListConfig)):
            native_value = OmegaConf.to_container(value, resolve=True) 
            if isinstance(value, ListConfig) and key in ["down_dims", "optimizer_betas"]:
                native_value = tuple(native_value)     
            # 写回 config
            setattr(policy.config, key, native_value)
            # logger.info(f"Auto-sanitized config field '{key}': {type(value)} -> {type(native_value)}")


class DeltaActionProcessorStep(ProcessorStep):
    def __init__(self, action_key="action", state_key="state"):
        super().__init__()
        self.action_key = action_key
        self.state_key = state_key

    def __call__(self, transition):
        # 与 AugmentationProcessorStep 一致：先拷贝，再改写
        new_transition = transition.copy()

        # 1) 取 action（兼容 TransitionKey.ACTION 与字符串 key）
        action = new_transition.get(TransitionKey.ACTION, None)
        # logger.info(f"action_shape={tuple(action.shape)}")

        # 2) 取 observation dict，再取 state
        obs_dict = new_transition.get(TransitionKey.OBSERVATION, None)

        state = obs_dict.get(self.state_key, None)
        # logger.info(f"state_shape={tuple(state.shape)}")
        if state is None:
            # 兼容传入 "observation.state" 的情况
            if self.state_key == "observation.state":
                state = obs_dict.get("state", None)
            if state is None:
                return new_transition

        # logger.info("[DEBUG]use delta")
        # 3) 计算 delta action: action - state_base
        action_dim = action.shape[-1]
        if state.dim() == 3:
            # [B, T, S] -> 取当前时刻（最后一帧）
            state_base = state[:, -1, :action_dim]  # [B, A]
        else:
            # [B, S]
            state_base = state[:, :action_dim]      # [B, A]
        
        # logger.info(
        #         f"[DeltaDebug] applied: action_shape={tuple(action.shape)}, "
        #         f"state_shape={tuple(state.shape)}, "
        #         f"delta_action={action[0][0] - state[0][1]}, "
        #     )
        
        new_transition[TransitionKey.ACTION] = action - state_base.unsqueeze(1)
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


class AugmentationProcessorStep(ProcessorStep):
    def __init__(self, transform, cam_keys):
        super().__init__()
        self.transform = transform
        self.cam_keys = [k for k in cam_keys if "depth" not in k]  # list of keys in the transition dict to augment

    def __call__(self, transition):
        # Store the current transition (required by ProcessorStep)
        new_transition = transition.copy()

        # Apply transform to each camera key
        data_dict = new_transition.get(TransitionKey.OBSERVATION)
        if data_dict is not None:
            # new_data_dict = {
            #     k: self.transform(v) if k in self.cam_keys else v
            #     for k, v in data_dict.items()
            # }
            new_data_dict = {}
            for k, v in data_dict.items():
                
                if k in self.cam_keys:
                    # print(k)
                    new_data_dict[k] = self.transform(v)
                else:
                    new_data_dict[k] = v
            # print(new_data_dict['observation.images.head_cam_h'].device)
            new_transition[TransitionKey.OBSERVATION] = new_data_dict
            return new_transition
        else:
            return new_transition
        

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Returns the input features unchanged.

        Device and dtype transformations do not alter the fundamental definition of the features (e.g., shape).

        Args:
            features: A dictionary of policy features.

        Returns:
            The original dictionary of policy features.
        """
        return features


class StateAugmentationProcessorStep(ProcessorStep):
    """
    状态增强处理器 - 对state添加噪声，提高模型对state误差的鲁棒性

    原理：
    - 训练时：state加噪声，模型学习在noisy_state下预测正确的action
    - 对于delta action：delta = action - noisy_state
    - 推理时：即使state有误差，模型也能预测正确的delta

    注意：此步骤仅用于训练，不应保存到preprocessor中
    """

    def __init__(self, config, state_key="observation.state"):
        super().__init__()
        self.enable = config.get("enable", False)
        self.noise_std = config.get("noise_std", 0.01)
        self.apply_prob = config.get("apply_prob", 0.7)
        self.state_key = state_key

    def __call__(self, transition):
        if not self.enable or torch.rand(1).item() > self.apply_prob:
            return transition

        # 与 DeltaActionProcessorStep 一致：先拷贝，再改写
        new_transition = transition.copy()

        # 取 observation dict，再取 state（模仿DeltaActionProcessorStep的方式）
        obs_dict = new_transition.get(TransitionKey.OBSERVATION, None)
        if obs_dict is None:
            return new_transition

        state = obs_dict.get(self.state_key, None)
        if state is None:
            # 兼容传入 "observation.state" 的情况
            if self.state_key == "observation.state":
                state = obs_dict.get("state", None)
            if state is None:
                return new_transition

        # 添加高斯噪声
        noise = torch.randn_like(state) * self.noise_std
        noisy_state = state + noise

        # 更新state（使用相同的key）
        obs_dict[self.state_key if self.state_key in obs_dict else "state"] = noisy_state
        new_transition[TransitionKey.OBSERVATION] = obs_dict
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Returns the input features unchanged.

        State augmentation does not alter the fundamental definition of the features.

        Args:
            features: A dictionary of policy features.

        Returns:
            The original dictionary of policy features.
        """
        return features


def insert_before_normalizer(pipeline, new_step):
    """
    Insert a processor step before the first NormalizerProcessorStep.
    If no NormalizerProcessorStep is found, append at the end.
    """
    for i, step in enumerate(pipeline.steps):
        if isinstance(step, NormalizerProcessorStep):
            pipeline.steps.insert(i, new_step)
            logger.info(f"Inserted {new_step.__class__.__name__} before NormalizerProcessorStep at index {i}")
            return new_step
    pipeline.steps.append(new_step)
    logger.info(f"No NormalizerProcessorStep found, appended {new_step.__class__.__name__} at the end")
    return new_step

def insert_before_step(pipeline, new_step, target_step_class):
    """
    Insert a processor step before a specific step type.
    If target step is not found, insert before NormalizerProcessorStep.

    Args:
        pipeline: The processor pipeline
        new_step: The step to insert
        target_step_class: The class type to insert before (e.g., DeltaActionProcessorStep)
    """
    for i, step in enumerate(pipeline.steps):
        if isinstance(step, target_step_class):
            pipeline.steps.insert(i, new_step)
            logger.info(f"Inserted {new_step.__class__.__name__} before {target_step_class.__name__} at index {i}")
            return new_step
    # If target not found, insert before normalizer
    return insert_before_normalizer(pipeline, new_step)

def remove_aug_step(pipeline, step_to_remove):
    """
    Remove the given step from the pipeline if it exists.
    """
    if step_to_remove in pipeline.steps:
        pipeline.steps.remove(step_to_remove)
        logger.info(f"Removed {step_to_remove.__class__.__name__}")
    else:
        logger.warning(f"Step {step_to_remove.__class__.__name__} not found in pipeline")

@hydra.main(config_path="../configs/policy/", config_name="diffusion_config", version_base=None)
def main(cfg: DictConfig):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    # Initialize Accelerator
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=cfg.training.accumulation_steps,
        log_with=None,                        # Disable logging
        # log_with="tensorboard",             # Disable logging
        device_placement=True,                # Explicitly enable device placement
        step_scheduler_with_optimizer=False,  # A fix to the stepping logic as accelerate might make this thread-unsafe.
        mixed_precision="fp16" if cfg.policy.get("use_amp", False) else "no",
        kwargs_handlers=[ddp_kwargs]          # transfer DDP kwargs
    )

    # With Accelerate, we get the device from accelerator
    device = accelerator.device

    # set_seed(cfg.training.seed)
    accelerate.utils.set_seed(cfg.training.seed)

    # mkdir and output TensorBoard only in the main process
    output_directory = None
    if accelerator.is_main_process:
        output_directory = Path(cfg.training.output_directory) / f"run_{cfg.timestamp}"
        output_directory.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(output_directory))

    # Dataset metadata and features
    dataset_metadata = LeRobotDatasetMetadata(cfg.repoid, root=cfg.root)
    features = dataset_to_policy_features(dataset_metadata.features)
    input_features = {k: ft for k, ft in features.items() if ft.type is not FeatureType.ACTION}
    output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}

    # instantiate the policy
    policy_cfg = build_policy_config(cfg, input_features, output_features)
    logger.info(f"policy_cfg: {policy_cfg}")

    # =========================================================================
    # [新增]: 加载 Delta Action 的统计参数并替换到 dataset_metadata 中
    # =========================================================================
    if cfg.policy.get("custom", {}).get("delta_action", {}).get("enable", False):
        stats_path = cfg.policy.custom.delta_action.stats_path
        logger.info(f"🔄 Delta Action enabled! Loading stats from {stats_path}")
        
        with open(stats_path, "r") as f:
            delta_stats = json.load(f)
            
        action_key = "action"
        if action_key in dataset_metadata.stats:
            # 覆盖原有的均值和方差
            dataset_metadata.stats[action_key]["mean"] = torch.tensor(delta_stats["mean"], dtype=torch.float32)
            dataset_metadata.stats[action_key]["std"] = torch.tensor(delta_stats["std"], dtype=torch.float32)
            dataset_metadata.stats[action_key]["min"] = torch.tensor(delta_stats["min"], dtype=torch.float32)
            dataset_metadata.stats[action_key]["max"] = torch.tensor(delta_stats["max"], dtype=torch.float32)
    # =========================================================================

    # Build policy
    policy = build_policy(cfg.policy_name, policy_cfg)
    accelerator.wait_for_everyone()

    # 创建preprocessor和postprocessor（此时还没有训练专用的增强步骤）
    preprocessor, postprocessor = make_pre_post_processors(policy_cfg, dataset_stats=dataset_metadata.stats)

    # 先保存干净的preprocessor（不包含训练时的增强步骤）
    if accelerator.is_main_process:
        preprocessor.save_pretrained(output_directory)
        postprocessor.save_pretrained(output_directory)
        logger.info("💾 Saved clean preprocessor and postprocessor (without training augmentation steps)")

    # Initialize optimizer and lr scheduler
    optimizer, lr_scheduler = build_optimizer_and_scheduler(policy, cfg, dataset_metadata.info["total_frames"], accelerator)

    # print only in main process
    training_info = {
        "Policy Name": cfg.policy_name,
        "Method": cfg.method,
        "Batch Size": f"{cfg.training.batch_size} (Global: {cfg.training.batch_size * accelerator.num_processes})",
        "Max Epochs": cfg.training.max_epoch,
        "Num Workers": cfg.training.num_workers,
        "Output Dir": str(output_directory) if output_directory else "N/A (Sub-process)",
        "Mixed Precision": accelerator.mixed_precision,
    }
    log_box("Training Configuration", training_info, icon="⚙️")

    dataset_info = {
        "Repo ID": cfg.repoid,
        "Total Frames": dataset_metadata.info["total_frames"],
        "FPS": dataset_metadata.fps
    }
    log_box("Dataset Information (used)", dataset_info, icon="💾")

    # 打印模型参数量
    num_total_params = sum(p.numel() for p in policy.parameters())
    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    trainable_ratio = num_learnable_params / num_total_params

    model_stats = {
        "Total Parameters": f"{num_total_params:,}",
        "Learnable Params": f"{num_learnable_params:,}",
        "Frozen Params": f"{num_total_params - num_learnable_params:,}",
        "Learnable Ratio": f"{trainable_ratio * 100:.2f}%"
    }
    
    # 根据是否冻结显示不同的图标
    if trainable_ratio <= 0:
        status_icon = "❄️"  # 完全冻结 (Frozen)
        status_text = "Frozen (Inference Only)"
    elif trainable_ratio < 1:  
        status_icon = "⚡"  # vision部分微调 (LoRA / PEFT) + DiT全量训练
        status_text = f"PEFT/LoRA (Trainable: {trainable_ratio:.2%})"
    else:
        status_icon = "🔥"  # 全量训练 (Full Finetune)
        status_text = "Full Fine-Tuning"
    
    log_box(status_text, model_stats, icon=status_icon)

    # Build dataset and dataloader
    delta_timestamps = build_delta_timestamps(dataset_metadata, policy_cfg)

    image_transforms = build_augmenter(cfg.training.RGB_Augmenter)
    dataset = LeRobotDataset(
        cfg.repoid,
        delta_timestamps=delta_timestamps,
        root=cfg.root,
        image_transforms=None,
    )
    accelerator.wait_for_everyone()
    # Training loop
    aug_step = insert_before_normalizer(preprocessor, AugmentationProcessorStep(image_transforms, dataset.meta.camera_keys))  # just for training

    # =========================================================================
    # [新增]: 状态增强 - 提高对state噪声的鲁棒性
    # 注意：必须在 Delta Action 计算之前插入！
    # =========================================================================
    state_aug_step = None
    if cfg.training.get("State_Augmenter", {}).get("enable", False):
        state_aug_step = StateAugmentationProcessorStep(cfg.training.State_Augmenter, state_key="observation.state")
        insert_before_normalizer(preprocessor, state_aug_step)
        logger.info("✅ Inserted StateAugmentationProcessorStep (will be applied before delta action)")
    # =========================================================================

    # =========================================================================
    # [新增]: 将 Delta Action 转换步骤插入到 normalizer 之前
    # 注意：Delta Action 计算会使用加噪后的 state
    # =========================================================================
    delta_step = None
    if cfg.policy.get("custom", {}).get("delta_action", {}).get("enable", False):
        delta_step = DeltaActionProcessorStep(action_key="action", state_key="observation.state")
        insert_before_normalizer(preprocessor, delta_step)
        logger.info("✅ Inserted DeltaActionProcessorStep before NormalizerProcessorStep")
    # =========================================================================

    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    dataloader = DataLoader(
        dataset,
        num_workers=cfg.training.num_workers,
        batch_size=cfg.training.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=(device.type != "cpu"),
        drop_last=cfg.training.drop_last,
        prefetch_factor=2 if cfg.training.num_workers > 0 else None,
    )
    # Use accelerator to prepare data, model, and optimizer
    accelerator.wait_for_everyone()

    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, dataloader, lr_scheduler
    )


    # Initialize training state variables
    start_epoch = 0
    steps = 0
    best_loss = float('inf')

    # # ===== Resume logic (perfect resume for AMP & RNG) =====
    if cfg.training.resume and cfg.training.resume_timestamp:
        resume_path = Path(cfg.training.output_directory) / cfg.training.resume_timestamp
        logger.info(f"🔄 Resuming from: {resume_path}")
        try:
            # Load state
            accelerator.load_state(resume_path / "epochlatest")
            if accelerator.is_main_process:
                latest_training_state = torch.load(resume_path / "training_latest_state.pth", map_location='cpu')
                steps = latest_training_state["steps"]
                start_epoch = latest_training_state["epoch"]
                best_loss = latest_training_state["best_loss"]
                logger.info(f"✅ Resumed training from epoch {start_epoch}, step {steps}, best_loss {best_loss}")
        except Exception as e:
            logger.error(f"❌ Failed to load checkpoint: {e}")
            logger.warning("🔄 Starting training from scratch.")
    else:
        logger.info("🚀 Training from scratch!")

    
    # Training loop
    for epoch in range(start_epoch, cfg.training.max_epoch):
        policy.train()

        # Use tqdm only on main process
        epoch_bar = Progress(
            dataloader, 
            desc=f"Epoch {epoch+1}/{cfg.training.max_epoch}",
            disable=not accelerator.is_main_process
        )

        total_loss = 0.0
        batch_count = 0
        for batch in epoch_bar:
            batch = preprocessor(batch)

            # if accelerator.is_main_process and steps == 0:
            #     # Delta action 检查
            #     if cfg.policy.get("custom", {}).get("delta_action", {}).get("enable", False):
            #         if "action" in batch and "observation.state" in batch:
            #             logger.info(f"[DeltaCheck] action shape: {tuple(batch['action'].shape)}")
            #             logger.info(f"[DeltaCheck] state shape: {tuple(batch['observation.state'].shape)}")
            #             logger.info(f"[DeltaCheck] action mean after preprocessor: {batch['action'].mean().item():.6f}")
            #         else:
            #             logger.warning("[DeltaCheck] Missing keys: 'action' or 'observation.state'")

            with accelerator.accumulate(policy):
                # batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
                with accelerator.autocast():
                    loss, _ = policy.forward(batch)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(policy.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()
                    if accelerator.is_main_process:
                        if steps % cfg.training.log_freq == 0:
                            current_lr = lr_scheduler.get_last_lr()[0]
                            writer.add_scalar("train/loss", loss.item(), steps)
                            writer.add_scalar("train/lr", current_lr, steps)
                            epoch_bar.set_postfix(loss=loss.item(), lr=current_lr)
                    steps += 1
                    batch_count += 1
                    total_loss += accelerator.gather(loss).mean().item()

        total_loss = total_loss / batch_count if batch_count > 0 else total_loss
        
        # Log, save, and eval flags
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            # Update best loss
            if total_loss < best_loss:
                best_loss = total_loss
                unwrapped_policy = accelerator.unwrap_model(policy)
                sanitize_policy_config(unwrapped_policy)
                unwrapped_policy.save_pretrained(output_directory / "epochbest")

            # Save checkpoint every N epochs
            if (epoch + 1) % cfg.training.save_freq_epoch == 0:
                unwrapped_policy = accelerator.unwrap_model(policy)
                sanitize_policy_config(unwrapped_policy)
                unwrapped_policy.save_pretrained(output_directory / f"epoch{epoch+1}")

                # save latest epoch training state based on accelerator save_state
            logger.warning("💾 Saving latest epoch training state... DON'T CTRL+C EXIT!!!!!!")
            accelerator.save_state(output_directory / "epochlatest")
            training_state = {
                "epoch": epoch+1, 
                "steps": steps,
                "best_loss": best_loss
            }
            torch.save(training_state, output_directory / "training_latest_state.pth")
            logger.info(f"🎉 Epoch {epoch+1} completed. Avg Loss: {total_loss:.4f}. Best Loss: {best_loss:.4f}")
        accelerator.wait_for_everyone()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        writer.close()
    
    accelerator.end_training()


if __name__ == "__main__":
    main()
