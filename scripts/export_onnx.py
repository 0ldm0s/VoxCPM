#!/usr/bin/env python3
"""
VoxCPM ONNX 统一导出脚本

该脚本整合了 VoxCPM 模型各组件的 ONNX 导出功能：
- voxcpm_prefill.onnx: 预填充阶段
- voxcpm_decode_step.onnx: 单步解码阶段
- audio_vae_encoder.onnx: 音频编码器
- audio_vae_decoder.onnx: 音频解码器

用法:
    python scripts/export_onnx.py --model_path /path/to/VoxCPM --output_dir ./onnx_models
    python scripts/export_onnx.py --model_path /path/to/VoxCPM --output_dir ./onnx_models --export prefill --export decode --export audiovae
"""

import os
import sys
import platform
import argparse
import logging
from typing import Tuple, Optional, List

# Apple Silicon (M1/M2/M3) 检测：自动设置 OpenMP 环境变量
if platform.system() == "Darwin" and platform.machine() == "arm64":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.onnx
from torch.export import Dim
import random
import numpy as np

# 设置项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from voxcpm.model.voxcpm import VoxCPMModel
from voxcpm.onnx.voxcpm.model.VoxCPM_Prefill import VoxCPMPrefill
from voxcpm.onnx.voxcpm.model.VoxCPM_Deocde import VoxCPMDecode
from voxcpm.onnx.voxcpm.model.VoxCPM_Audio_VAE_Encoder import VoxCPMAudioVAEEncoder
from voxcpm.onnx.voxcpm.model.VoxCPM_Audio_VAE_Decoder import VoxCPMAudioVAEDecoder
from voxcpm.onnx.export.utils import validate_onnx_model_with_torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """设置所有随机种子以保证可重复性"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# AudioVAE 编码器导出
# ============================================================================

def create_dummy_inputs_encoder(audio_vae, batch_size: int = 2, audio_length: int = 16000) -> Tuple[torch.Tensor]:
    """创建 AudioVAE 编码器的虚拟输入"""
    audio_data = torch.randn(batch_size, 1, audio_length, dtype=torch.float32)
    return (audio_data,)


def export_audio_vae_encoder(
    voxcpm_model: VoxCPMModel,
    output_path: str,
    opset_version: int = 20,
    batch_size: int = 2,
    audio_length: int = 16000,
    fix_batch1: bool = False,
):
    """导出 AudioVAE 编码器到 ONNX"""
    logger.info("正在导出 AudioVAE 编码器...")

    voxcpm_model = voxcpm_model.to(torch.float32).cpu()
    voxcpm_model.eval()
    set_seed(42)

    audio_vae = voxcpm_model.audio_vae
    wrapper = VoxCPMAudioVAEEncoder(voxcpm_model, sample_rate=audio_vae.sample_rate)
    wrapper.eval()

    dummy_inputs = create_dummy_inputs_encoder(
        audio_vae,
        batch_size=1 if fix_batch1 else batch_size,
        audio_length=audio_length,
    )

    audio_length_dim = Dim("audio_length", min=int(audio_vae.hop_length), max=int(audio_vae.sample_rate) * 30)

    if fix_batch1:
        dynamic_shapes = {"audio_data": {2: audio_length_dim}}
    else:
        dim_batch = Dim("batch_size", min=1, max=32)
        dynamic_shapes = {"audio_data": {0: dim_batch, 2: audio_length_dim}}

    try:
        onnx_program = torch.onnx.export(
            wrapper,
            dummy_inputs,
            f=None,
            dynamo=True,
            opset_version=opset_version,
            input_names=["audio_data"],
            output_names=["z"],
            dynamic_shapes=dynamic_shapes,
            verbose=False,
            external_data=True,
        )
        onnx_program.save(output_path)
        logger.info(f"AudioVAE 编码器已导出至 {output_path}")
    except Exception as e:
        logger.error(f"导出 AudioVAE 编码器失败: {e}")
        raise


# ============================================================================
# AudioVAE 解码器导出
# ============================================================================

def create_dummy_inputs_decoder(audio_vae, batch_size: int = 2, latent_length: int = 100) -> Tuple[torch.Tensor]:
    """创建 AudioVAE 解码器的虚拟输入"""
    latent_dim = getattr(audio_vae, 'latent_dim', None) or 128
    z = torch.randn(batch_size, latent_dim, latent_length, dtype=torch.float32)
    return (z,)


def export_audio_vae_decoder(
    voxcpm_model: VoxCPMModel,
    output_path: str,
    opset_version: int = 20,
    batch_size: int = 2,
    latent_length: int = 100,
    fix_batch1: bool = False,
):
    """导出 AudioVAE 解码器到 ONNX"""
    logger.info("正在导出 AudioVAE 解码器...")

    voxcpm_model = voxcpm_model.to(torch.float32).cpu()
    voxcpm_model.eval()
    set_seed(42)

    audio_vae = voxcpm_model.audio_vae
    wrapper = VoxCPMAudioVAEDecoder(voxcpm_model)
    wrapper.eval()

    dummy_inputs = create_dummy_inputs_decoder(
        audio_vae,
        batch_size=1 if fix_batch1 else batch_size,
        latent_length=latent_length,
    )

    max_latent = int(getattr(audio_vae, 'max_latent_length', 4096))
    latent_length_dim = Dim("latent_length", min=1, max=max_latent)

    if fix_batch1:
        dynamic_shapes = {"z": {2: latent_length_dim}}
    else:
        dim_batch = Dim("batch_size", min=1, max=32)
        dynamic_shapes = {"z": {0: dim_batch, 2: latent_length_dim}}

    try:
        onnx_program = torch.onnx.export(
            wrapper,
            dummy_inputs,
            f=None,
            dynamo=True,
            opset_version=opset_version,
            input_names=["z"],
            output_names=["audio"],
            dynamic_shapes=dynamic_shapes,
            verbose=False,
            external_data=True,
        )
        onnx_program.save(output_path)
        logger.info(f"AudioVAE 解码器已导出至 {output_path}")
    except Exception as e:
        logger.error(f"导出 AudioVAE 解码器失败: {e}")
        raise


# ============================================================================
# Prefill 阶段导出
# ============================================================================

def create_dummy_inputs_prefill(model: VoxCPMModel, batch_size: int = 1, seq_length: int = 8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """创建 Prefill 阶段的虚拟输入"""
    patch_size = model.config.patch_size
    feat_dim = model.config.feat_dim

    text = torch.randint(low=0, high=10000, size=(batch_size, seq_length), dtype=torch.int64)
    text_mask = torch.ones(batch_size, seq_length, dtype=torch.int32)
    feat = torch.randn(batch_size, seq_length, patch_size, feat_dim, dtype=torch.float32)
    feat_mask = torch.ones(batch_size, seq_length, dtype=torch.int32)
    return (text, text_mask, feat, feat_mask)


def export_voxcpm_prefill(
    model: VoxCPMModel,
    output_path: str,
    opset_version: int = 20,
    batch_size: int = 2,
    seq_length: int = 8,
    fix_batch1: bool = False,
):
    """导出 Prefill 阶段到 ONNX"""
    logger.info("正在导出 VoxCPM Prefill 阶段...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(torch.float32).to(device)
    model.eval()
    set_seed(42)

    wrapper = VoxCPMPrefill(model)
    wrapper.eval()

    dummy_batch_size = 1 if fix_batch1 else batch_size
    dummy_inputs = create_dummy_inputs_prefill(model, batch_size=dummy_batch_size, seq_length=seq_length)
    dummy_inputs = tuple(inp.to(device) for inp in dummy_inputs)

    dim_seq_length = Dim("seq_length", min=2, max=model.config.max_length)

    if fix_batch1:
        dynamic_shapes = {
            "text": {1: dim_seq_length},
            "text_mask": {1: dim_seq_length},
            "feat": {1: dim_seq_length},
            "feat_mask": {1: dim_seq_length},
        }
    else:
        dim_batch = Dim("batch_size", min=1, max=32)
        dynamic_shapes = {
            "text": {0: dim_batch, 1: dim_seq_length},
            "text_mask": {0: dim_batch, 1: dim_seq_length},
            "feat": {0: dim_batch, 1: dim_seq_length},
            "feat_mask": {0: dim_batch, 1: dim_seq_length},
        }

    try:
        onnx_program = torch.onnx.export(
            wrapper,
            dummy_inputs,
            f=None,
            dynamo=True,
            opset_version=opset_version,
            do_constant_folding=False,
            input_names=["text", "text_mask", "feat", "feat_mask"],
            output_names=[
                "dit_hidden",
                "base_next_keys",
                "base_next_values",
                "residual_next_keys",
                "residual_next_values",
                "prefix_feat_cond",
            ],
            dynamic_shapes=dynamic_shapes,
            verbose=False,
            external_data=True,
        )
        onnx_program.save(output_path)
        logger.info(f"Prefill 阶段已导出至 {output_path}")
    except Exception as e:
        logger.error(f"导出 Prefill 阶段失败: {e}")
        raise


# ============================================================================
# Decode 阶段导出
# ============================================================================

class VoxCPMDecodeFixedTimestepsWrapper(torch.nn.Module):
    """固定推理步数的 Decode 封装器，用于 ONNX 导出"""

    def __init__(self, decode_module: VoxCPMDecode, timesteps: int):
        super().__init__()
        self.decode = decode_module
        self.timesteps = int(timesteps)

    def forward(
        self,
        dit_hidden: torch.Tensor,
        base_next_keys: torch.Tensor,
        base_next_values: torch.Tensor,
        residual_next_keys: torch.Tensor,
        residual_next_values: torch.Tensor,
        prefix_feat_cond: torch.Tensor,
        noise: torch.Tensor,
        cfg_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.decode(
            dit_hidden,
            base_next_keys,
            base_next_values,
            residual_next_keys,
            residual_next_values,
            prefix_feat_cond,
            noise=noise,
            inference_timesteps=self.timesteps,
            cfg_value=cfg_value,
        )


def create_dummy_inputs_decode(
    model: VoxCPMModel,
    batch_size: int = 2,
    past_seq_length: int = 8,
    cfg_value: float = 2.0,
) -> Tuple[torch.Tensor, ...]:
    """创建 Decode 阶段的虚拟输入"""
    h_dit = model.config.dit_config.hidden_dim
    hidden_size = model.config.lm_config.hidden_size

    num_layers_base = model.config.lm_config.num_hidden_layers
    num_heads_base = model.config.lm_config.num_key_value_heads
    head_dim_base = model.base_lm.config.kv_channels or (hidden_size // model.base_lm.config.num_attention_heads)

    num_layers_res = model.residual_lm.config.num_hidden_layers
    num_heads_res = model.residual_lm.config.num_key_value_heads
    head_dim_res = model.residual_lm.config.kv_channels or (hidden_size // model.residual_lm.config.num_attention_heads)

    patch_size = model.config.patch_size
    feat_dim = model.config.feat_dim

    dit_hidden = torch.randn(batch_size, h_dit, dtype=torch.float32)
    base_next_keys = torch.randn(batch_size, num_layers_base, num_heads_base, past_seq_length, head_dim_base, dtype=torch.float32)
    base_next_values = torch.randn(batch_size, num_layers_base, num_heads_base, past_seq_length, head_dim_base, dtype=torch.float32)
    residual_next_keys = torch.randn(batch_size, num_layers_res, num_heads_res, past_seq_length, head_dim_res, dtype=torch.float32)
    residual_next_values = torch.randn(batch_size, num_layers_res, num_heads_res, past_seq_length, head_dim_res, dtype=torch.float32)
    prefix_feat_cond = torch.randn(batch_size, patch_size, feat_dim, dtype=torch.float32)
    cfg_value_tensor = torch.tensor(cfg_value, dtype=torch.float32)
    noise = torch.randn(batch_size, patch_size, feat_dim, dtype=torch.float32)

    return (
        dit_hidden,
        base_next_keys,
        base_next_values,
        residual_next_keys,
        residual_next_values,
        prefix_feat_cond,
        noise,
        cfg_value_tensor,
    )


def export_voxcpm_decode(
    model: VoxCPMModel,
    output_path: str,
    timesteps: int,
    cfg_value: float,
    opset_version: int = 20,
    batch_size: int = 2,
    past_seq_length: int = 8,
    fix_batch1: bool = False,
):
    """导出 Decode 阶段到 ONNX"""
    logger.info("正在导出 VoxCPM Decode 阶段...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(torch.float32).to(device)
    model.eval()
    set_seed(42)

    decode = VoxCPMDecode(model)
    wrapper = VoxCPMDecodeFixedTimestepsWrapper(decode, timesteps=timesteps)
    wrapper.eval()

    dummy_inputs = create_dummy_inputs_decode(
        model,
        batch_size=1 if fix_batch1 else batch_size,
        past_seq_length=past_seq_length,
        cfg_value=cfg_value,
    )
    dummy_inputs = tuple(inp.to(device) for inp in dummy_inputs)

    dim_past_seq_length = Dim("past_seq_length", min=1, max=2048)

    if fix_batch1:
        dynamic_shapes = {
            "dit_hidden": {},
            "base_next_keys": {3: dim_past_seq_length},
            "base_next_values": {3: dim_past_seq_length},
            "residual_next_keys": {3: dim_past_seq_length},
            "residual_next_values": {3: dim_past_seq_length},
            "prefix_feat_cond": {},
            "noise": {},
            "cfg_value": None,
        }
    else:
        dim_batch = Dim("batch_size", min=2, max=64)
        dynamic_shapes = {
            "dit_hidden": {0: dim_batch},
            "base_next_keys": {0: dim_batch, 3: dim_past_seq_length},
            "base_next_values": {0: dim_batch, 3: dim_past_seq_length},
            "residual_next_keys": {0: dim_batch, 3: dim_past_seq_length},
            "residual_next_values": {0: dim_batch, 3: dim_past_seq_length},
            "prefix_feat_cond": {0: dim_batch},
            "noise": {0: dim_batch},
            "cfg_value": None,
        }

    try:
        onnx_program = torch.onnx.export(
            wrapper,
            dummy_inputs,
            f=None,
            dynamo=True,
            opset_version=opset_version,
            do_constant_folding=False,
            input_names=[
                "dit_hidden",
                "base_next_keys",
                "base_next_values",
                "residual_next_keys",
                "residual_next_values",
                "prefix_feat_cond",
                "noise",
                "cfg_value",
            ],
            output_names=[
                "pred_feat",
                "new_dit_hidden",
                "new_base_next_keys",
                "new_base_next_values",
                "new_residual_next_keys",
                "new_residual_next_values",
                "stop_flag",
            ],
            dynamic_shapes=dynamic_shapes,
            verbose=False,
            external_data=True,
        )
        onnx_program.save(output_path)
        logger.info(f"Decode 阶段已导出至 {output_path}")
    except Exception as e:
        logger.error(f"导出 Decode 阶段失败: {e}")
        raise


# ============================================================================
# 验证函数
# ============================================================================

def validate_model(
    torch_model: torch.nn.Module,
    onnx_path: str,
    test_inputs: Tuple[torch.Tensor, ...],
    input_names: List[str],
    output_names: List[str],
    rtol: float = 1e-3,
    atol: float = 1e-4,
    num_tests: int = 3,
):
    """验证导出的 ONNX 模型"""
    logger.info(f"正在验证 ONNX 模型: {onnx_path}")

    validation_results = validate_onnx_model_with_torch(
        torch_model=torch_model,
        onnx_path=onnx_path,
        test_inputs=test_inputs,
        input_names=input_names,
        output_names=output_names,
        rtol=rtol,
        atol=atol,
        num_tests=num_tests,
        verbose=True
    )

    if 'error' in validation_results:
        logger.error(f"验证失败: {validation_results['error']}")
        return False
    else:
        logger.info(f"验证完成: {validation_results['successful_tests']}/{validation_results['total_tests']} 次测试成功")
        return True


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="VoxCPM ONNX 统一导出脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
导出的 ONNX 模型:

  voxcpm_prefill.onnx    - 预填充阶段: text + feat → dit_hidden + kv_caches + prefix_feat_cond
  voxcpm_decode_step.onnx - 单步解码: dit_hidden + kv_caches + noise + cfg_value → pred_feat + new_kv_caches + stop_flag
  audio_vae_encoder.onnx - 音频编码: audio → latent
  audio_vae_decoder.onnx - 音频解码: latent → audio

示例:
  # 导出所有模型
  python scripts/export_onnx.py --model_path /path/to/VoxCPM --output_dir ./onnx_models

  # 仅导出 Prefill 和 Decode
  python scripts/export_onnx.py --model_path /path/to/VoxCPM --output_dir ./onnx_models --export prefill --export decode

  # 导出并验证
  python scripts/export_onnx.py --model_path /path/to/VoxCPM --output_dir ./onnx_models --validate
        """
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="VoxCPM 模型目录路径",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./onnx_models",
        help="ONNX 模型输出目录",
    )
    parser.add_argument(
        "--export",
        type=str,
        action="append",
        choices=["prefill", "decode", "audiovae_encoder", "audiovae_decoder", "all"],
        default=["all"],
        help="指定要导出的模型组件 (可多次使用)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="虚拟输入的批次大小",
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=8,
        help="Prefill 阶段的序列长度",
    )
    parser.add_argument(
        "--past_seq_length",
        type=int,
        default=8,
        help="Decode 阶段的过去序列长度",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=10,
        help="扩散推理步数 (Decode 阶段)",
    )
    parser.add_argument(
        "--cfg_value",
        type=float,
        default=2.0,
        help="Classifier-Free Guidance 值",
    )
    parser.add_argument(
        "--audio_length",
        type=int,
        default=16000,
        help="AudioVAE 编码器的音频长度",
    )
    parser.add_argument(
        "--latent_length",
        type=int,
        default=100,
        help="AudioVAE 解码器的隐向量长度",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=64,
        help="AudioVAE 解码器的隐向量维度",
    )
    parser.add_argument(
        "--opset_version",
        type=int,
        default=23,
        help="ONNX opset 版本",
    )
    parser.add_argument(
        "--fix_batch1",
        action="store_true",
        help="将批次维度固定为 1 (静态)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="导出后验证 ONNX 模型",
    )
    parser.add_argument(
        "--num_tests",
        type=int,
        default=3,
        help="验证测试次数",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="验证相对容差",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-4,
        help="验证绝对容差",
    )

    args = parser.parse_args()

    # 处理 "all" 选项
    if "all" in args.export:
        exports = ["prefill", "decode", "audiovae_encoder", "audiovae_decoder"]
    else:
        exports = args.export

    # macOS Apple Silicon 导出 ONNX 时使用 MPS
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        logger.info("macOS Apple Silicon 检测到，使用 MPS 导出 ONNX")
        force_device = "mps"
    else:
        force_device = None

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        logger.info(f"正在加载 VoxCPM 模型: {args.model_path}")
        voxcpm_model = VoxCPMModel.from_local(args.model_path, optimize=False, device=force_device)
        logger.info("模型加载成功")

        # 如果指定了 latent_dim，设置到 audio_vae
        if args.latent_dim is not None:
            setattr(voxcpm_model.audio_vae, 'latent_dim', int(args.latent_dim))

        # 导出 AudioVAE 编码器
        if "audiovae_encoder" in exports:
            output_path = os.path.join(args.output_dir, "audio_vae_encoder.onnx")
            export_audio_vae_encoder(
                voxcpm_model,
                output_path,
                opset_version=args.opset_version,
                batch_size=args.batch_size,
                audio_length=args.audio_length,
                fix_batch1=args.fix_batch1,
            )

            if args.validate:
                wrapper = VoxCPMAudioVAEEncoder(
                    voxcpm_model.to(torch.float32).cpu(),
                    sample_rate=voxcpm_model.audio_vae.sample_rate
                )
                wrapper.eval()
                test_inputs = create_dummy_inputs_encoder(
                    voxcpm_model.audio_vae,
                    batch_size=1 if args.fix_batch1 else args.batch_size,
                    audio_length=args.audio_length,
                )
                validate_model(
                    torch_model=wrapper,
                    onnx_path=output_path,
                    test_inputs=test_inputs,
                    input_names=["audio_data"],
                    output_names=["z"],
                    rtol=args.rtol,
                    atol=args.atol,
                    num_tests=args.num_tests,
                )

        # 导出 AudioVAE 解码器
        if "audiovae_decoder" in exports:
            output_path = os.path.join(args.output_dir, "audio_vae_decoder.onnx")
            export_audio_vae_decoder(
                voxcpm_model,
                output_path,
                opset_version=args.opset_version,
                batch_size=args.batch_size,
                latent_length=args.latent_length,
                fix_batch1=args.fix_batch1,
            )

            if args.validate:
                wrapper = VoxCPMAudioVAEDecoder(voxcpm_model.to(torch.float32).cpu())
                wrapper.eval()
                test_inputs = create_dummy_inputs_decoder(
                    voxcpm_model.audio_vae,
                    batch_size=1 if args.fix_batch1 else args.batch_size,
                    latent_length=args.latent_length,
                )
                validate_model(
                    torch_model=wrapper,
                    onnx_path=output_path,
                    test_inputs=test_inputs,
                    input_names=["z"],
                    output_names=["audio"],
                    rtol=args.rtol,
                    atol=args.atol,
                    num_tests=args.num_tests,
                )

        # 导出 Prefill 阶段
        if "prefill" in exports:
            output_path = os.path.join(args.output_dir, "voxcpm_prefill.onnx")
            export_voxcpm_prefill(
                voxcpm_model,
                output_path,
                opset_version=args.opset_version,
                batch_size=args.batch_size,
                seq_length=args.seq_length,
                fix_batch1=args.fix_batch1,
            )

            if args.validate:
                wrapper = VoxCPMPrefill(voxcpm_model.to(torch.float32).cpu())
                wrapper.eval()
                test_inputs = create_dummy_inputs_prefill(
                    voxcpm_model,
                    batch_size=1 if args.fix_batch1 else args.batch_size,
                    seq_length=args.seq_length,
                )
                validate_model(
                    torch_model=wrapper,
                    onnx_path=output_path,
                    test_inputs=test_inputs,
                    input_names=["text", "text_mask", "feat", "feat_mask"],
                    output_names=[
                        "dit_hidden",
                        "base_next_keys",
                        "base_next_values",
                        "residual_next_keys",
                        "residual_next_values",
                        "prefix_feat_cond",
                    ],
                    rtol=args.rtol,
                    atol=args.atol,
                    num_tests=args.num_tests,
                )

        # 导出 Decode 阶段
        if "decode" in exports:
            output_path = os.path.join(args.output_dir, "voxcpm_decode_step.onnx")
            export_voxcpm_decode(
                voxcpm_model,
                output_path,
                timesteps=args.timesteps,
                cfg_value=args.cfg_value,
                opset_version=args.opset_version,
                batch_size=args.batch_size,
                past_seq_length=args.past_seq_length,
                fix_batch1=args.fix_batch1,
            )

            if args.validate:
                decode = VoxCPMDecode(voxcpm_model.to(torch.float32).cpu())
                wrapper = VoxCPMDecodeFixedTimestepsWrapper(decode, timesteps=args.timesteps)
                wrapper.eval()
                test_inputs = create_dummy_inputs_decode(
                    voxcpm_model,
                    batch_size=1 if args.fix_batch1 else args.batch_size,
                    past_seq_length=args.past_seq_length,
                    cfg_value=args.cfg_value,
                )
                validate_model(
                    torch_model=wrapper,
                    onnx_path=output_path,
                    test_inputs=test_inputs,
                    input_names=[
                        "dit_hidden",
                        "base_next_keys",
                        "base_next_values",
                        "residual_next_keys",
                        "residual_next_values",
                        "prefix_feat_cond",
                        "noise",
                        "cfg_value",
                    ],
                    output_names=[
                        "pred_feat",
                        "new_dit_hidden",
                        "new_base_next_keys",
                        "new_base_next_values",
                        "new_residual_next_keys",
                        "new_residual_next_values",
                        "stop_flag",
                    ],
                    rtol=args.rtol,
                    atol=args.atol,
                    num_tests=args.num_tests,
                )

        logger.info("=" * 60)
        logger.info("ONNX 导出完成！")
        logger.info(f"输出目录: {args.output_dir}")
        logger.info("=" * 60)

        # 列出导出的文件
        logger.info("导出的文件:")
        for filename in os.listdir(args.output_dir):
            if filename.endswith(".onnx"):
                filepath = os.path.join(args.output_dir, filename)
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                logger.info(f"  - {filename} ({size_mb:.2f} MB)")

    except Exception as e:
        logger.error(f"导出失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
