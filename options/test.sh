#!/bin/bash

# 获取当前脚本的路径
SCRIPT_PATH=$(realpath "$0")
SCRIPT_NAME=$(basename "$0")

# 获取name参数指定的文件夹名称
NAME="test"
DEST_DIR="./runs/$NAME"

# 检查目标文件夹是否存在，不存在则创建
if [ ! -d "$DEST_DIR" ]; then
  mkdir -p "$DEST_DIR"
fi

# 拷贝当前脚本到目标文件夹
cp "$SCRIPT_PATH" "$DEST_DIR"

# 运行python命令
python -u core/train.py \
    --name "$NAME" \
    --stage chairs \
    --validation chairs \
    --gpus 0 1\
    --num_steps 150000 \
    --batch_size 5 \
    --lr 0.0002 \
    --image_size 256 256 \
    --wdecay 0.0001 \
    # --raft_ckpt raft-chairs.pth \
    # --former_ckpt LOL_v1.pth
