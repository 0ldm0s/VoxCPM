# VoxCPM 快速训练与推理指南

本文档介绍如何使用项目提供的 Shell 脚本快速进行模型训练和推理。

## 脚本概览

| 脚本 | 用途 | 是否需要训练数据 |
|------|------|----------------|
| `train.sh` | 通用 TTS 微调 | 需要（任意音频+文本） |
| `train_clone.sh` | 声音克隆训练 | 需要（**同一说话人**的多段音频+文本） |
| `infer.sh` | 推理 | 可选（支持参考音频声音克隆） |

---

## 1. train.sh - 通用 TTS 微调

提升通用 TTS 能力，不针对特定音色优化。

### 训练数据格式

```json
{"audio": "path/to/audio1.wav", "text": "这是第一段音频的文本"}
{"audio": "path/to/audio2.wav", "text": "这是第二段音频的文本"}
```

### 使用方法

```bash
# LoRA 微调（默认）
./train.sh lora

# 全参数微调
./train.sh all

# 指定训练数据
./train.sh lora --train_manifest data/my_data.jsonl

# 自定义配置
./train.sh /path/to/custom.yaml

# 传递额外参数
./train.sh lora --num_iters 2000 --batch_size 8
```

---

## 2. train_clone.sh - 声音克隆训练

用**同一说话人**的多段音频微调，让模型学会该音色。

### 训练数据格式

```json
{"audio": "speaker1_clip1.wav", "text": "这是该说话人的第一段话"}
{"audio": "speaker1_clip2.wav", "text": "这是该说话人的第二段话"}
{"audio": "speaker1_clip3.wav", "text": "这是该说话人的第三段话"}
```

### 使用方法

```bash
# 基本用法（必需参数：训练数据文件）
./train_clone.sh data/my_voice.jsonl

# 指定迭代次数
./train_clone.sh data/my_voice.jsonl --num_iters 2000

# 指定保存路径
./train_clone.sh data/my_voice.jsonl --save_path checkpoints/my_voice_lora
```

### 注意事项

- 训练数据建议包含该说话人 **5-10 分钟**以上的音频
- 每段音频长度建议 **3-10 秒**
- 数据量越大、音频质量越好，学到的音色越准确

---

## 3. infer.sh - 推理

支持两种推理模式：

### 模式 A：基础推理（无参考音频）

```bash
./infer.sh --lora_ckpt checkpoints/lora/step_0001000 --text "你好世界"
```

### 模式 B：参考音频声音克隆

```bash
./infer.sh --lora_ckpt checkpoints/lora/step_0001000 \
    --text "你好" \
    --prompt_audio ref.wav \
    --prompt_text "这是参考音频的文本"
```

### 常用参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--lora_ckpt` | LoRA  checkpoint 路径 | `--lora_ckpt checkpoints/lora/step_0001000` |
| `--text` | 要合成的文本 | `--text "你好世界"` |
| `--prompt_audio` | 参考音频路径 | `--prompt_audio ref.wav` |
| `--prompt_text` | 参考音频对应文本 | `--prompt_text "这是参考"` |
| `--output` | 输出文件名 | `--output result.wav` |

### 使用预训练模型（无需训练）

```bash
# 不做任何训练，直接用参考音频克隆
./infer.sh --lora_ckpt openbmb/VoxCPM1.5 \
    --text "你好" \
    --prompt_audio ref.wav \
    --prompt_text "参考文本"
```

---

## 平台支持

### macOS (Apple Silicon)

脚本会自动检测并添加 `--distributed false` 参数，禁用分布式训练。

```bash
./train.sh lora
./train_clone.sh data/my_voice.jsonl
```

### Linux (CUDA)

```bash
# 单 GPU
./train.sh lora

# 多 GPU（需安装 NCCL）
torchrun --nproc_per_node=2 ./train.sh lora
```

---

## 数据格式说明

### JSONL 文件格式

每行一个 JSON 对象，字段说明：

| 字段 | 必需 | 说明 |
|------|------|------|
| `audio` | 是 | 音频文件路径（绝对或相对路径） |
| `text` | 是 | 音频对应的文本 |
| `duration` | 否 | 音频时长（秒），可加速数据过滤 |
| `dataset_id` | 否 | 数据集 ID，用于多数据集训练 |

### 示例文件

参考 `examples/train_data_example.jsonl`：

```json
{"audio": "examples/example.wav", "text": "This is an example audio transcript for training."}
{"audio": "data/audio1.wav", "text": "Another training sample."}
```

---

## 常见问题

### Q: 训练时报内存不足 (OOM)

```bash
# 减小 batch_size
./train.sh lora --batch_size 4

# 增大梯度累积
./train.sh lora --batch_size 1 --grad_accum_steps 8

# 减小最大 token 数
./train.sh lora --max_batch_tokens 4096
```

### Q: macOS 上训练失败

确保使用 `train_clone.sh` 或 `train.sh`，脚本会自动处理分布式设置。

### Q: 声音克隆效果不理想

- 增加训练数据量（建议 5 分钟以上）
- 确保参考音频和训练数据来自同一人
- 尝试更多迭代步数（`--num_iters 3000`）

---

## 完整示例

### 1. 准备数据

```bash
# 创建数据目录
mkdir -p data

# 创建训练数据（同一说话人的多段音频）
cat > data/my_voice.jsonl << 'EOF'
{"audio": "data/clip1.wav", "text": "今天天气真不错"}
{"audio": "data/clip2.wav", "text": "我想去公园散步"}
{"audio": "data/clip3.wav", "text": "这本书很有意思"}
EOF
```

### 2. 训练

```bash
./train_clone.sh data/my_voice.jsonl --num_iters 2000
```

### 3. 推理

```bash
# 用学到的音色生成
./infer.sh --lora_ckpt checkpoints/lora/step_0020000 --text "你好世界"
```
