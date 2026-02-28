#!/bin/bash

SSH_OPT="ssh -p 3072"
REMOTE_HOST="agilex@121.48.164.165"

# 更改参数
MODE=${1:-"2"}

case $MODE in
    "1")
        echo "使用配置 [1]: IDP3 (rgb)"
        SRC="/home/wushihan/data/Codes/kuavo_data_challenge_icra/outputs/train/diffusion_idp3/"
        DEST="/home/agilex/liangweiqing/kdc/kuavo_data_challenge_icra/outputs/train/diffusion_idp3"
        ;;
    "2")
        echo "使用配置 [2]: Task1 (baseline)"
        SRC="/home/liangweiqing/data/code/kuavo_data_challenge_icra/outputs/train/Task1/siglip_DFormer_S_train_1"
        DEST="/home/agilex/liangweiqing/kdc/kuavo_data_challenge_icra/outputs/train/Task1"
        ;;
    *)
        echo "未知模式: $MODE"
        exit 1
        ;;
esac

echo "正在同步: $SRC -> $DEST"
rsync -avzP -e "$SSH_OPT" --rsync-path="mkdir -p ${DEST} && rsync" "$SRC" "${REMOTE_HOST}:${DEST}"