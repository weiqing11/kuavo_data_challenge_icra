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
from kuavo_data.common.pcd_utils_cpu import augmentation


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
    1. Selecting point cloud variants 暂时取消
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
