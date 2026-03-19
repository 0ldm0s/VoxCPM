#!/bin/bash
#
# VoxCPM 依赖安装脚本
#
# 用法:
#   ./scripts/install_dependencies.sh           # 使用官方 PyPI 源
#   ./scripts/install_dependencies.sh --mirror  # 使用国内镜像源
#   ./scripts/install_dependencies.sh -m       # 使用国内镜像源 (简写)
#

set -e

# 默认不使用镜像
USE_MIRROR=false

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --mirror|-m)
            USE_MIRROR=true
            shift
            ;;
        --help|-h)
            echo "用法: $0 [--mirror|-m]"
            echo ""
            echo "选项:"
            echo "  --mirror, -m  使用国内镜像源安装 (清华)"
            echo "  --help, -h    显示此帮助信息"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 设置 pip 源
if [ "$USE_MIRROR" = true ]; then
    echo "使用国内镜像源: 清华 PyPI"
    PIP_EXTRA_INDEX_URL="--index-url https://pypi.tuna.tsinghua.edu.cn/simple"
    # 额外添加官方源作为备用（某些包国内镜像可能没有）
    PIP_FALLBACK="--extra-index-url https://pypi.org/simple"
else
    echo "使用官方 PyPI 源"
    PIP_EXTRA_INDEX_URL=""
    PIP_FALLBACK=""
fi

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 核心依赖
CORE_DEPS=(
    "torch>=2.5.0"
    "torchaudio>=2.5.0"
    "torchcodec"
    "transformers>=4.36.2"
    "einops"
    "gradio<6"
    "inflect"
    "addict"
    "modelscope>=1.22.0"
    "datasets>=3,<4"
    "huggingface-hub"
    "pydantic"
    "tqdm"
    "simplejson"
    "sortedcontainers"
    "soundfile"
    "librosa"
    "matplotlib"
    "funasr"
    "spaces"
    "argbind"
    "safetensors"
)

# ONNX 相关依赖
ONNX_DEPS=(
    "onnx"
    "onnxruntime"
)

# 开发依赖
DEV_DEPS=(
    "pytest"
    "black"
    "flake8"
    "mypy"
    "pre-commit"
)

echo "=========================================="
echo "安装 VoxCPM 核心依赖"
echo "=========================================="

pip install $PIP_EXTRA_INDEX_URL $PIP_FALLBACK "${CORE_DEPS[@]}"

echo ""
echo "=========================================="
echo "安装 ONNX 相关依赖"
echo "=========================================="

pip install $PIP_EXTRA_INDEX_URL $PIP_FALLBACK "${ONNX_DEPS[@]}"

echo ""
echo "=========================================="
echo "安装开发依赖 (可选)"
echo "=========================================="
read -p "是否安装开发依赖? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install $PIP_EXTRA_INDEX_URL $PIP_FALLBACK "${DEV_DEPS[@]}"
fi

echo ""
echo "=========================================="
echo "安装完成!"
echo "=========================================="

# 检查关键包是否安装成功
echo ""
echo "验证关键依赖..."
python -c "import torch; print(f'  torch: {torch.__version__}')"
python -c "import transformers; print(f'  transformers: {transformers.__version__}')"
python -c "import addict; print(f'  addict: OK')"
python -c "import funasr; print(f'  funasr: OK')"
python -c "import onnx; print(f'  onnx: {onnx.__version__}')"
python -c "import onnxruntime; print(f'  onnxruntime: {onnxruntime.__version__}')"
python -c "import spaces; print(f'  spaces: OK')"
python -c "import soundfile; print(f'  soundfile: OK')"
python -c "import librosa; print(f'  librosa: {librosa.__version__}')"

echo ""
echo "所有依赖安装成功!"
