#!/usr/bin/env python3
"""
VoxCPM MPS 推理性能优化验证脚本

用于验证 torch.compile 和 float16 dtype 优化是否生效
"""
import os
import sys

# 必须在导入 torch 之前设置环境变量
if sys.platform == "darwin":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
import torch

# 设置项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

# 解决 macOS M1/M2/M3 系列芯片上 OpenMP 冲突问题
if sys.platform == "darwin":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 禁用 tokenizer 并行
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import voxcpm
from voxcpm.model.utils import get_dtype


def test_optimization():
    """测试优化效果"""

    print("=" * 60)
    print("VoxCPM MPS 推理性能优化验证")
    print("=" * 60)

    # 检查设备
    if torch.backends.mps.is_available():
        device = "mps"
        print(f"\n设备: MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = "cuda"
        print(f"\n设备: CUDA (NVIDIA GPU)")
    else:
        device = "cpu"
        print(f"\n设备: CPU")

    # 检查 dtype
    dtype = get_dtype("bfloat16")
    print(f"bfloat16 映射到: {dtype}")

    # 查找模型目录
    model_dirs = [
        "./models/VoxCPM1.5",
        "./models/openbmb__VoxCPM1.5",
    ]

    model_dir = None
    for d in model_dirs:
        if os.path.isdir(d):
            model_dir = d
            break

    if model_dir is None:
        print("\n错误: 找不到模型目录")
        print("请确保模型已下载到以下位置之一:")
        for d in model_dirs:
            print(f"  - {d}")
        return

    print(f"\n模型目录: {model_dir}")

    # 加载模型（会触发 optimize）
    print("\n" + "-" * 60)
    print("加载模型...")
    print("-" * 60)

    start_load = time.time()
    model = voxcpm.VoxCPM(
        voxcpm_model_path=model_dir,
        optimize=True,
        enable_denoiser=False,  # 禁用降噪器以加快加载
        device=device
    )
    load_time = time.time() - start_load

    print(f"\n模型加载完成，耗时: {load_time:.2f} 秒")

    # 测试推理
    print("\n" + "-" * 60)
    print("测试推理性能...")
    print("-" * 60)

    test_text = "VoxCPM 是一个端到端的文本转语音模型。"
    print(f"\n测试文本: {test_text}")

    # 预热
    print("\n预热运行...")
    model.generate(
        text=test_text,
        max_len=10,
        inference_timesteps=4
    )
    print("预热完成")

    # 正式测试
    print("\n正式测试 (3 次运行)...")
    times = []

    for i in range(3):
        print(f"  运行 {i+1}/3...", end=" ", flush=True)
        start = time.time()
        audio = model.generate(
            text=test_text,
            max_len=50,
            inference_timesteps=10
        )
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"完成 ({elapsed:.2f} 秒)")

    avg_time = sum(times) / len(times)
    print(f"\n平均推理时间: {avg_time:.2f} 秒")
    print(f"生成音频长度: {audio.shape[0]} 样本")

    # 计算实时率
    sample_rate = model.tts_model.sample_rate
    audio_duration = audio.shape[0] / sample_rate
    rtf = avg_time / audio_duration
    print(f"音频时长: {audio_duration:.2f} 秒")
    print(f"实时率 (RTF): {rtf:.2f} (数值越小越好)")

    # 优化建议
    print("\n" + "=" * 60)
    print("性能评估")
    print("=" * 60)

    if rtf < 0.5:
        status = "优秀"
    elif rtf < 1.0:
        status = "良好"
    elif rtf < 2.0:
        status = "一般"
    else:
        status = "较慢"

    print(f"\n实时率: {rtf:.2f} - {status}")

    if rtf > 1.0:
        print("\n优化建议:")
        print("  1. 检查 torch.compile 是否成功启用（查看启动日志）")
        print("  2. 尝试降低 inference_timesteps 参数（当前: 10）")
        print("  3. 对于长文本，使用 streaming 模式")
    else:
        print("\n优化效果良好！")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_optimization()
