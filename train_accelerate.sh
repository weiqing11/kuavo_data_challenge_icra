#!/bin/bash

# 设置脚本在遇到错误时立即退出
set -e

# 设置项目根目录为当前目录（可选，确保 python 路径正确）
export PYTHONPATH="${PYTHONPATH}:."

# 配置文件路径变量
ACCELERATE_CONFIG="configs/accelerate/ac_915.yaml"
POLICY_CONFIG_PATH="../configs/policy"
POLICY_CONFIG_NAME="diffusion_new_config.yaml"
TRAIN_SCRIPT="kuavo_train/train_policy_with_accelerate.py"

echo "Starting training with Accelerate..."
echo "Accelerate Config: $ACCELERATE_CONFIG"
echo "Policy Config: $POLICY_CONFIG_NAME"

# 执行命令
accelerate launch --config_file "$ACCELERATE_CONFIG" \
    "$TRAIN_SCRIPT" \
    --config-path="$POLICY_CONFIG_PATH" \
    --config-name="$POLICY_CONFIG_NAME"