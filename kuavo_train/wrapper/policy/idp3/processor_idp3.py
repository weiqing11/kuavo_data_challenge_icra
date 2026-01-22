import torch
from lerobot.processor import PolicyProcessorPipeline, NormalizerProcessorStep

def make_idp3_pre_post_processors(policy_cfg, dataset_stats=None):
    """
    为 IDP3 策略创建预处理器和后处理器。
    """
    # NormalizerProcessorStep 需要知道所有特征（输入和输出）的定义
    # 所以我们需要合并 input_features 和 output_features
    features = {**policy_cfg.input_features, **policy_cfg.output_features}

    # 1. 创建归一化步骤 (NormalizerProcessorStep)
    # 关键修正：参数名必须是 features, norm_map, stats
    normalization_step = NormalizerProcessorStep(
        features=features,
        norm_map=policy_cfg.normalization_mapping,
        stats=dataset_stats
    )

    steps = [normalization_step]

    # 2. 创建预处理管道 (Preprocessor)
    preprocessor = PolicyProcessorPipeline(steps)

    # 3. 创建后处理管道 (Postprocessor)
    postprocessor = PolicyProcessorPipeline(steps)

    return preprocessor, postprocessor