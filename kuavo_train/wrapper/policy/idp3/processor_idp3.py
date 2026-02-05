#!/usr/bin/env python

from typing import Any
import random
from dataclasses import dataclass, field
import numpy as np
import torch

# 根据你的文件结构，这里使用相对导入
from .configuration_idp3 import IDP3Config
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.processor.pipeline import ObservationProcessorStep
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME
from dataclasses import dataclass
from kuavo_data.common.pcd_utils_cpu import choose_method, augmentation

@dataclass
class PointCloudVariantSelectorAndAugmentationStep(ObservationProcessorStep):
    # 从 config 中提取的需要处理的点云前缀列表
    pc_prefixes: list[str] = field(default_factory=list)
    target_points: int = 4096
    task_id = 1

    def observation(self, observation: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if not self.pc_prefixes:
            raise ValueError("pc_prefixes list is empty. At least one point cloud prefix must be provided.")

        prefixes_to_process = self.pc_prefixes
        
        for prefix in prefixes_to_process:
            # 使用 choose_method() 高效选择变体
            selected_method = choose_method()
            selected_variant_key = f"{prefix}_{selected_method}"
            
            if selected_variant_key not in observation:
                raise ValueError(
                    f"Expected point cloud variant key '{selected_variant_key}' not found in observation. "
                    f"Available keys: {list(observation.keys())}"
                )
            
            point_cloud = observation[selected_variant_key]
            camera_id = self._get_camera_id(prefix)

            # 暂时不做数据增强
            # point_cloud = self._augment_point_cloud(point_cloud, self.task_id, camera_id)

            observation[prefix] = point_cloud
            
            # 删除其它变体，只保留pc_h，pc_r，或者pc_l
            # keys_to_remove = [k for k in observation.keys() if k.startswith(f"{prefix}_")]
            # for k in keys_to_remove:
            #     del observation[k]
        
        return observation
    
    def _get_camera_id(self, prefix: str) -> str:
        if "pc_h" in prefix:
            return "cam_h"
        elif "pc_l" in prefix:
            return "cam_l"
        elif "pc_r" in prefix:
            return "cam_r"
        else:
            raise ValueError(f"Unknown point cloud prefix '{prefix}' for camera ID extraction.")

    def _augment_point_cloud(self, point_cloud: torch.Tensor | np.ndarray, task_id: int, camera_id: str) -> torch.Tensor:
        """
        封装的数据增强函数。
        支持处理 (N, C), (T, N, C), (B, T, N, C) 等任意维度的输入。
        """
        is_tensor = isinstance(point_cloud, torch.Tensor)
        original_device = point_cloud.device if is_tensor else None
        
        # 转换为 Numpy
        if is_tensor:
            pc_numpy = point_cloud.cpu().numpy()
        else:
            pc_numpy = point_cloud
        
        # 保存原始形状信息
        original_shape = pc_numpy.shape
        # 假设最后两维总是 (Num_Points, Channels)
        # 例如: (64, 2, 8192, 6) -> batch_dims=(64, 2), num_points=8192, channels=6
        batch_dims = original_shape[:-2]
        num_points_in = original_shape[-2]
        channels = original_shape[-1]
        
        # 展平前面的维度，变成 list of (N, C)
        # (64, 2, 8192, 6) -> (128, 8192, 6)
        if len(batch_dims) > 0:
            flattened_pc = pc_numpy.reshape(-1, num_points_in, channels)
        else:
            flattened_pc = pc_numpy[np.newaxis, ...] # 处理单个样本的情况
            
        processed_list = []
        for i in range(flattened_pc.shape[0]):
            # 处理单个点云 (8192, 6) -> (Target, 6)
            single_pc = flattened_pc[i]
            
            # 调用外部工具函数
            # 注意：这步是在 CPU 上进行的，对于大 Batch 可能会比较慢
            aug_pc = augmentation(
                point_cloud=single_pc,
                task_id=task_id,
                camera_id=camera_id,
                target_points=self.target_points
            )
            processed_list.append(aug_pc)
            
        # 重新堆叠
        processed_array = np.stack(processed_list, axis=0)
        
        # 恢复维度
        # (128, 4096, 6) -> (64, 2, 4096, 6)
        new_shape = batch_dims + (self.target_points, channels)
        processed_array = processed_array.reshape(new_shape)
        
        # 转换回 Tensor 并移回原设备
        if is_tensor:
            return torch.from_numpy(processed_array).to(original_device)
        
        return processed_array

    def transform_features(self, features):
        """返回特征不变（此步骤不改变特征定义）"""
        return features


def make_idp3_pre_post_processors(
    config: IDP3Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for an IDP3 policy.

    The pre-processing pipeline prepares the input data for the model by:
    1. Selecting point cloud variants (data augmentation).
    2. Renaming features.
    3. Normalizing the input and output features based on dataset statistics.
    4. Adding a batch dimension.
    5. Moving the data to the specified device.

    The post-processing pipeline handles the model's output by:
    1. Moving the data to the CPU.
    2. Unnormalizing the output features to their original scale.

    Args:
        config: The configuration object for the IDP3 policy,
            containing feature definitions, normalization mappings, and device information.
        dataset_stats: A dictionary of statistics used for normalization.
            Defaults to None.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """

    # 输入预处理步骤
    input_steps = [
        # 点云变体选择和增强（集成在一个步骤中）
        # 从 config.obs_dict 中提取需要处理的点云前缀
        PointCloudVariantSelectorAndAugmentationStep(
            pc_prefixes=[key for key in config.obs_dict.keys() if "pc_" in key]
        ),
        # 如果需要重命名观察空间键值（例如将 observation.image.front 重命名为 observation.images），在这里配置
        RenameObservationsProcessorStep(rename_map={}),
        # 增加 Batch 维度 (C, H, W) -> (1, C, H, W)
        AddBatchDimensionProcessorStep(),
        # 移动数据到计算设备 (CPU/GPU)
        DeviceProcessorStep(device=config.device),
        # 归一化：根据 config.normalization_mapping 和 dataset_stats 对输入特征进行归一化
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
    ]

    # 输出后处理步骤
    output_steps = [
        # 反归一化：将模型输出的 Action 还原到原始物理空间数值
        UnnormalizerProcessorStep(
            features=config.output_features, 
            norm_map=config.normalization_mapping, 
            stats=dataset_stats
        ),
        # 将数据移回 CPU
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
