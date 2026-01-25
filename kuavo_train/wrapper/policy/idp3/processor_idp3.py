import torch
from lerobot.processor import PolicyProcessorPipeline, NormalizerProcessorStep, UnnormalizerProcessorStep

def make_idp3_pre_post_processors(policy_cfg, dataset_stats=None):
    """
    为 IDP3 策略创建预处理器和后处理器。
    函数名必须匹配 make_{policy_name}_pre_post_processors 格式。
    """
    # 合并所有特征定义，因为 Normalizer/Unnormalizer 需要知道所有 Key 的统计数据
    features = {**policy_cfg.input_features, **policy_cfg.output_features}

    # 1. 创建预处理器 (Preprocessor)
    # 使用 NormalizerProcessorStep 将输入 (observation) 归一化
    normalization_step = NormalizerProcessorStep(
        features=features,
        norm_map=policy_cfg.normalization_mapping,
        stats=dataset_stats
    )
    preprocessor = PolicyProcessorPipeline([normalization_step])

    # 2. 创建后处理器 (Postprocessor)
    # 使用 UnnormalizerProcessorStep 将输出 (action) 反归一化
    unnormalization_step = UnnormalizerProcessorStep(
        features=features,
        norm_map=policy_cfg.normalization_mapping,
        stats=dataset_stats
    )
    postprocessor = PolicyProcessorPipeline([unnormalization_step])

    return preprocessor, postprocessor