#!/bin/bash
# VoxCPM 通用训练脚本（标准 TTS 微调）
# 用于提升通用 TTS 能力，不针对特定音色优化
#
# 用法: ./train.sh [lora|all] [配置文件路径(可选)] [额外参数...]
#
# 示例:
#   ./train.sh lora                                    # LoRA 微调（默认）
#   ./train.sh all                                    # 全参数微调
#   ./train.sh /path/to/custom.yaml                   # 自定义配置
#   ./train.sh lora --train_manifest my_data.jsonl    # 指定训练数据

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

CONF_DIR="$PROJECT_ROOT/conf/voxcpm_v1.5"
CONFIG_FILE="$CONF_DIR/voxcpm_finetune_lora.yaml"

# 解析参数
if [ $# -ge 1 ]; then
    case "$1" in
        lora)
            CONFIG_FILE="$CONF_DIR/voxcpm_finetune_lora.yaml"
            ;;
        all)
            CONFIG_FILE="$CONF_DIR/voxcpm_finetune_all.yaml"
            ;;
        *)
            if [ -f "$1" ]; then
                CONFIG_FILE="$1"
            else
                echo "错误: 配置文件不存在: $1"
                exit 1
            fi
            ;;
    esac
fi

EXTRA_ARGS="${@:2}"

echo "=========================================="
echo "VoxCPM 训练启动"
echo "=========================================="
echo "配置: $CONFIG_FILE"
echo "额外参数: $EXTRA_ARGS"
echo "=========================================="

cd "$PROJECT_ROOT"

# macOS 自动单设备
if [ "$(uname)" == "Darwin" ]; then
    EXTRA_ARGS="--distributed false $EXTRA_ARGS"
fi

python scripts/train_voxcpm_finetune.py \
    --config_path "$CONFIG_FILE" \
    $EXTRA_ARGS
