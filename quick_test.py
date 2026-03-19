#!/usr/bin/env python3
"""
快速测试 VoxCPM 在 MPS 上的性能
"""
import os
import sys
import time

# 必须在导入 torch 之前设置
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import voxcpm


def quick_test():
    print("=" * 50)
    print("VoxCPM 快速性能测试")
    print("=" * 50)

    # 检查模型
    model_dir = "./models/VoxCPM1.5"
    if not os.path.isdir(model_dir):
        print(f"\n错误: 找不到模型目录 {model_dir}")
        return

    print(f"\n模型目录: {model_dir}")
    print("加载模型（首次运行会较慢）...")

    start = time.time()
    model = voxcpm.VoxCPM(
        voxcpm_model_path=model_dir,
        enable_denoiser=False,
    )
    load_time = time.time() - start
    print(f"模型加载完成，耗时: {load_time:.1f} 秒")

    # 测试推理
    test_text = "这是一个测试句子，用于验证 VoxCPM 的性能。"
    print(f"\n测试文本: {test_text}")
    print("开始推理...")

    start = time.time()
    audio = model.generate(text=test_text, max_len=50)
    elapsed = time.time() - start

    # 计算性能指标
    audio_duration = len(audio) / model.tts_model.sample_rate
    rtf = elapsed / audio_duration

    print(f"\n结果:")
    print(f"  推理时间: {elapsed:.1f} 秒")
    print(f"  音频时长: {audio_duration:.1f} 秒")
    print(f"  RTF: {rtf:.2f}")

    # 性能评估
    if rtf < 0.5:
        status = "优秀 ✅"
    elif rtf < 1.0:
        status = "良好 ✅"
    elif rtf < 3.0:
        status = "正常 ✅"
    elif rtf < 10.0:
        status = "较慢 ⚠️"
    else:
        status = "异常 ❌"

    print(f"\n性能评估: {status}")

    if rtf >= 10.0:
        print("\n提示:")
        print("  - 如果是首次运行，请再次测试")
        print("  - 检查是否有其他程序占用 GPU")
        print("  - 尝试重启系统后再次测试")

    print("=" * 50)


if __name__ == "__main__":
    quick_test()
