#!/usr/bin/env python3
"""
通过 Gradio API 测试 VoxCPM Web UI 的性能
"""
import os
import sys
import time
import requests

# API 端点
API_URL = "http://localhost:7860"

# 测试参数
test_text = "VoxCPM 是一个端到端的文本转语音模型，专为生成高度真实的语音而设计。"


def test_api():
    print("=" * 50)
    print("VoxCPM Web API 性能测试")
    print("=" * 50)

    # 检查服务是否运行
    try:
        response = requests.get(f"{API_URL}", timeout=5)
        print(f"\n✓ Web UI 正在运行: {API_URL}")
    except requests.exceptions.RequestException as e:
        print(f"\n✗ 无法连接到 Web UI: {e}")
        print("请先启动 Web UI: python app.py")
        return

    # 获取 API 配置
    try:
        config = requests.get(f"{API_URL}/config", timeout=5).json()
        print(f"✓ API 配置加载成功")
    except:
        print("⚠ 无法获取 API 配置，继续测试...")

    # 准备请求数据
    # 根据 Gradio API 格式构造请求
    payload = {
        "data": [
            test_text,                    # text
            None,                         # prompt_wav (使用默认示例)
            "每天只需聆听几分钟，你就能通过积极的心态消除负面想法。",  # prompt_text
            2.0,                          # cfg_value
            10,                           # inference_timesteps
            False,                        # do_normalize
            False,                        # denoise
        ]
    }

    print(f"\n测试文本: {test_text}")
    print(f"\n发送推理请求...")

    # 发送推理请求
    start = time.time()
    try:
        response = requests.post(
            f"{API_URL}/api/generate",
            json=payload,
            timeout=300,  # 5 分钟超时
        )
        elapsed = time.time() - start

        if response.status_code == 200:
            result = response.json()

            # 检查返回结果
            if "data" in result and len(result["data"]) > 0:
                audio_data = result["data"][0]
                if audio_data:
                    # 解析音频数据
                    import base64
                    import json

                    # Gradio 返回的数据可能是 base64 编码的
                    sample_rate = 44100  # VoxCPM 默认采样率

                    print(f"\n结果:")
                    print(f"  状态: ✓ 成功")
                    print(f"  推理时间: {elapsed:.1f} 秒")

                    # 计算 RTF
                    # 假设返回的是音频数据，计算音频长度
                    if isinstance(audio_data, dict) and "audio" in audio_data:
                        # 可能是 {"audio": base64_data, "sample_rate": rate} 格式
                        audio_bytes = base64.b64decode(audio_data["audio"])
                        audio_duration = len(audio_bytes) / (sample_rate * 2)  # 假设 16-bit PCM
                    elif isinstance(audio_data, list):
                        # 可能是采样点数组
                        audio_duration = len(audio_data) / sample_rate
                    else:
                        audio_duration = 5.0  # 估算

                    rtf = elapsed / audio_duration if audio_duration > 0 else float('inf')

                    print(f"  估算音频时长: {audio_duration:.1f} 秒")
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
                        print("  - 尝试降低 inference_timesteps 参数")

                else:
                    print(f"\n⚠ 返回数据为空")
            else:
                print(f"\n⚠ 返回格式异常: {result.keys()}")
        else:
            print(f"\n✗ API 请求失败: HTTP {response.status_code}")
            print(f"  响应: {response.text[:200]}")

    except requests.exceptions.Timeout:
        print(f"\n✗ 请求超时（{300} 秒）")
        print("  这表明推理速度非常慢，请检查:")
        print("  1. 是否有其他程序占用 GPU 资源")
        print("  2. 系统内存是否充足")
        print("  3. 尝试重启 Web UI")

    except requests.exceptions.RequestException as e:
        print(f"\n✗ 请求失败: {e}")

    print("\n" + "=" * 50)


if __name__ == "__main__":
    test_api()
