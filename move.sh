#!/bin/bash

SSH_OPT="ssh -p 307"
SRC="/home/wushihan/data/Codes/kuavo_data_challenge_icra/outputs/train/Task1/dino+siglip/run_20260122_225025"
REMOTE_HOST="humanoid@121.48.164.165"
DEST="/home/humanoid/liangweiqing/code/kuavo_data_challenge_icra/outputs/train/Task1/dino+siglip/"

rsync -avzP -e "$SSH_OPT" "$SRC" "${REMOTE_HOST}:${DEST}"