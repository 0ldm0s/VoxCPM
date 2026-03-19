# VoxCPM 改进版

本项目基于 [VoxCPM](https://github.com/OpenBMB/VoxCPM) 进行改进，原始版本说明请参阅 [ORIG_README.md](./ORIG_README.md)。

---

## 本项目改进内容

### 1. ONNX 模型导出支持

本项目支持将 VoxCPM 模型导出为 ONNX 格式，可用于 **Rust + ONNX Runtime** 推理。

#### 目录结构

```
src/voxcpm/onnx/
├── __init__.py
├── voxcpm/
│   ├── model/                          # ONNX 专用模型封装
│   │   ├── VoxCPM_Prefill.py          # 预填充阶段
│   │   ├── VoxCPM_Deocde.py           # 解码阶段
│   │   ├── VoxCPM_Audio_VAE_Encoder.py
│   │   ├── VoxCPM_Audio_VAE_Decoder.py
│   │   └── utils.py
│   └── modules/
│       └── minicpm4/                  # ONNX 兼容的 MiniCPM4
│           ├── model.py                 # 包含 ONNX 友好的 forward_step
│           ├── cache.py
│           ├── config.py
│           └── sdpa_gqa.py
└── export/
    ├── export_audio_vae_encoder.py
    ├── export_audio_vae_decoder.py
    ├── export_voxcpm_prefill.py
    ├── export_voxcpm_decode.py
    └── utils.py

scripts/
├── export_onnx.py                      # 统一导出脚本
└── install_dependencies.sh             # 依赖安装脚本
```

#### 导出模型说明

| 模型 | 输入 | 输出 | 用途 |
|------|------|------|------|
| `voxcpm_prefill.onnx` | text_tokens, text_mask, feat, feat_mask | dit_hidden, kv_caches, prefix_feat_cond | 预填充阶段 |
| `voxcpm_decode_step.onnx` | dit_hidden, kv_caches, noise, cfg_value | pred_feat, new_kv_caches, stop_flag | 单步解码 |
| `audio_vae_encoder.onnx` | audio, sample_rate | latent | 音频编码 |
| `audio_vae_decoder.onnx` | latent | audio | 音频解码 |

#### 使用方法

```bash
# 安装依赖
./scripts/install_dependencies.sh --mirror   # 使用国内镜像

# 导出所有 ONNX 模型
python scripts/export_onnx.py \
    --model_path /path/to/VoxCPM \
    --output_dir ./onnx_models

# 导出并验证
python scripts/export_onnx.py \
    --model_path /path/to/VoxCPM \
    --output_dir ./onnx_models \
    --validate

# 仅导出特定组件
python scripts/export_onnx.py \
    --model_path /path/to/VoxCPM \
    --export prefill \
    --export decode
```

### 2. 依赖安装脚本

提供统一的依赖安装脚本，支持国内镜像源：

```bash
# 使用官方 PyPI 源
./scripts/install_dependencies.sh

# 使用国内镜像（清华）
./scripts/install_dependencies.sh --mirror
./scripts/install_dependencies.sh -m
```

### 3. Apple Silicon (M1/M2/M3) 推理支持

推理代码已适配 Apple Silicon Mac，可在 M 系列芯片上运行。

### 4. 训练脚本 Apple Silicon 支持

训练脚本 `scripts/train_voxcpm_finetune.py` 已适配 Apple Silicon：
- 自动检测 MPS 设备
- 支持 Gloo 后端分布式训练（替代 NCCL）

---

## 快速开始

```bash
# 1. 安装依赖
./scripts/install_dependencies.sh --mirror

# 2. 推理
voxcpm --text "你好，世界" --output out.wav

# 3. 导出 ONNX 模型
python scripts/export_onnx.py --model_path /path/to/model --output_dir ./onnx
```

---

## 文档

- [ORIG_README.md](./ORIG_README.md) - 原始 VoxCPM 项目说明
- [docs/usage_guide.md](./docs/usage_guide.md) - 使用指南
- [docs/finetune.md](./docs/finetune.md) - 微调指南

---

## License

基于 Apache-2.0 许可证，详见 [LICENSE](./LICENSE)。
