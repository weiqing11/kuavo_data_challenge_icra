#!/bin/bash

SSH_OPT="ssh -p 307"
REMOTE_HOST="humanoid@121.48.164.165"

# 更改参数
MODE=${1:-"2"}

case $MODE in
    "1")
        echo "使用配置 [1]: IDP3 (no_rgb)"
        SRC="/home/wushihan/data/Codes/kuavo_data_challenge_icra/outputs/train/idp3_v2_no_rgb/idp3"
        DEST="/home/humanoid/liangweiqing/code/kuavo_data_challenge_icra/outputs/train/idp3_v2_no_rgb"
        ;;
    "2")
        echo "使用配置 [2]: Task1 (baseline)"
        SRC="/home/liangweiqing/data/code/kuavo_data_challenge_icra/outputs/train/Task1/siglip_local_depth"
        DEST="/home/humanoid/liangweiqing/code/kuavo_data_challenge_icra/outputs/train/Task1"
        ;;
    *)
        exit 1
        ;;
esac

echo "正在同步: $SRC -> $DEST"
rsync -avzP -e "$SSH_OPT" --rsync-path="mkdir -p ${DEST} && rsync" "$SRC" "${REMOTE_HOST}:${DEST}"