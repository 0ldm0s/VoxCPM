#!/bin/bash
# VoxCPM Web UI 启动脚本

# 设置项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# 设置 Python 路径
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

# 解决 macOS M1/M2/M3 系列芯片上 OpenMP 冲突问题
export KMP_DUPLICATE_LIB_OK=TRUE

# 禁用 tokenizer 并行
export TOKENIZERS_PARALLELISM=false

# 设置模型路径（如果使用本地模型）
# app.py 默认查找 ./models/VoxCPM1.5/，但我们的模型在 ./models/openbmb__VoxCPM1.5/
# 创建符号链接或使用 HF_REPO_ID 环境变量
if [ -d "models/openbmb__VoxCPM1.5" ] && [ ! -d "models/VoxCPM1.5" ]; then
    echo "创建模型目录符号链接: models/VoxCPM1.5 -> models/openbmb__VoxCPM1.5"
    ln -s "openbmb__VoxCPM1.5" "models/VoxCPM1.5"
fi

# 或者使用 HF_REPO_ID 指定模型
# export HF_REPO_ID="openbmb/VoxCPM1.5"

# 运行 Gradio 应用
echo "启动 VoxCPM Web UI..."
python app.py "$@"
