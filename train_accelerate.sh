
accelerate launch \
    --config_file configs/accelerate/accelerate_config.yaml \
    kuavo_train/train_policy_with_accelerate.py \
    --config-path=../configs/policy \
    --config-name=diffusion_new_config.yaml