#!/bin/bash
# VoxCPM 推理脚本（支持参考音频声音克隆）
# 推理时提供参考音频，模型提取音色特征后生成
#
# 用法:
#   ./infer.sh --lora_ckpt <checkpoint> --text "文本"
#   ./infer.sh --lora_ckpt <checkpoint> --text "文本" --prompt_audio ref.wav --prompt_text "参考文本"
#
# 示例:
#   ./infer.sh --lora_ckpt checkpoints/lora/step_0001000 --text "你好世界"
#   ./infer.sh --lora_ckpt checkpoints/lora/step_0001000 --text "你好" --prompt_audio ref.wav --prompt_text "这是参考"

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

cd "$PROJECT_ROOT"

python scripts/test_voxcpm_lora_infer.py "$@"
