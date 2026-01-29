#!/bin/bash

SSH_OPT="ssh -p 307"
#SRC="/home/wushihan/data/Codes/kuavo_data_challenge_icra/outputs/train/idp3_v2_no_rgb/idp3/run_20260125_184949"
SRC="/home/wushihan/data/Codes/kuavo_data_challenge_icra/outputs/train/Task1/siglip+global"
REMOTE_HOST="humanoid@121.48.164.165"
DEST="/home/humanoid/liangweiqing/code/kuavo_data_challenge_icra/outputs/train/Task1"

rsync -avzP -e "$SSH_OPT" --rsync-path="mkdir -p ${DEST} && rsync" "$SRC" "${REMOTE_HOST}:${DEST}"