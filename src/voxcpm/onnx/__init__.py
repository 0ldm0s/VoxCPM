"""
VoxCPM ONNX 导出模块

该模块提供了 VoxCPM 模型的 ONNX 导出功能，包括：
- VoxCPMPrefill: Prefill 阶段导出
- VoxCPMDecode: Decode 阶段导出
- VoxCPMAudioVAEEncoder: AudioVAE 编码器导出
- VoxCPMAudioVAEDecoder: AudioVAE 解码器导出

使用方法:
    使用 scripts/export_onnx.py 统一导出脚本：
    python scripts/export_onnx.py --model_path /path/to/VoxCPM --output_dir ./onnx_models

或者单独导出各个组件：
    - python src/voxcpm/onnx/export/export_voxcpm_prefill.py --model_path /path/to/VoxCPM
    - python src/voxcpm/onnx/export/export_voxcpm_decode.py --model_path /path/to/VoxCPM
    - python src/voxcpm/onnx/export/export_audio_vae_encoder.py --model_path /path/to/VoxCPM
    - python src/voxcpm/onnx/export/export_audio_vae_decoder.py --model_path /path/to/VoxCPM
"""

from .voxcpm.model.VoxCPM_Prefill import VoxCPMPrefill
from .voxcpm.model.VoxCPM_Deocde import VoxCPMDecode
from .voxcpm.model.VoxCPM_Audio_VAE_Encoder import VoxCPMAudioVAEEncoder
from .voxcpm.model.VoxCPM_Audio_VAE_Decoder import VoxCPMAudioVAEDecoder

__all__ = [
    "VoxCPMPrefill",
    "VoxCPMDecode",
    "VoxCPMAudioVAEEncoder",
    "VoxCPMAudioVAEDecoder",
]
