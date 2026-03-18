#!/bin/bash
# VoxCPM 声音克隆训练脚本
# 训练数据：同一说话人的多段音频+文本
# 训练后：模型学会该说话人的音色，推理时无需提供参考音频
#
# 用法: ./train_clone.sh <训练数据.jsonl> [额外参数...]
# 示例:
#   ./train_clone.sh data/my_voice.jsonl
#   ./train_clone.sh data/my_voice.jsonl --num_iters 2000

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

if [ $# -lt 1 ]; then
    echo "用法: $0 <训练数据.jsonl> [额外参数...]"
    echo "示例: $0 data/my_voice.jsonl"
    exit 1
fi

TRAIN_MANIFEST="$1"
shift

if [ ! -f "$TRAIN_MANIFEST" ]; then
    echo "错误: 训练数据文件不存在: $TRAIN_MANIFEST"
    exit 1
fi

CONF_FILE="$PROJECT_ROOT/conf/voxcpm_v1.5/voxcpm_finetune_lora.yaml"

echo "=========================================="
echo "VoxCPM 声音克隆训练"
echo "=========================================="
echo "训练数据: $TRAIN_MANIFEST"
echo "=========================================="

cd "$PROJECT_ROOT"

# macOS 自动单设备
if [ "$(uname)" == "Darwin" ]; then
    python scripts/train_voxcpm_finetune.py \
        --config_path "$CONF_FILE" \
        --train_manifest "$TRAIN_MANIFEST" \
        --distributed false \
        "$@"
else
    python scripts/train_voxcpm_finetune.py \
        --config_path "$CONF_FILE" \
        --train_manifest "$TRAIN_MANIFEST" \
        "$@"
fi
