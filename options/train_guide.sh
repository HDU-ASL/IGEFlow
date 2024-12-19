#!/bin/bash
# 延迟 16 小时执行
# sleep 57600
# 获取当前脚本的路径
SCRIPT_PATH=$(realpath "$0")
SCRIPT_NAME=$(basename "$0")

# 获取name参数指定的文件夹名称
NAME="train_guide"
DEST_DIR="./runs/$NAME"

# 检查目标文件夹是否存在，不存在则创建
if [ ! -d "$DEST_DIR" ]; then
  mkdir -p "$DEST_DIR"
fi

# 拷贝当前脚本到目标文件夹
cp "$SCRIPT_PATH" "$DEST_DIR"

# 运行python命令
python -u core/train_hidden.py \
    --name "$NAME" \
    --stage chairs-4img \
    --validation chairs \
    --gpus 0 1\
    --num_steps 200000 \
    --batch_size 16 \
    --lr 0.00025 \
    --image_size 368 496 \
    --wdecay 0.0001 \
    --restore_ckpt raft-chairs.pth \
    --feature_guide_ckpt raft-chairs.pth \
    --use_enhance \
    --lamda 0.1
