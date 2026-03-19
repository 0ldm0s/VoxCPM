"""
VoxCPM Decode wrapper

This module factors out the non-streaming autoregressive decode step so it can be
exported as a static computation graph (e.g., ONNX). It consumes the outputs
from the Prefill stage and generates one patch of audio features per step,
updating hidden states and KV caches.

The DiT hidden state is now received as input (computed in Prefill stage) and
updated at the end of each decode step for the next iteration. This avoids
creating separate small models for lm_to_dit_proj and res_to_dit_proj operations.

It places the stop prediction logic at the end of the step, as requested.
"""
from typing import Tuple

import torch
import torch.nn as nn

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .voxcpm import VoxCPMModel

# 导入 ONNX 版本的 MiniCPMModel（返回 3 个值）
from ..modules.minicpm4.model import MiniCPMModel as ONNXMiniCPMModel


def _convert_to_onnx_model(base_lm):
    """
    将主项目的 MiniCPMModel（返回 2 个值）转换为 ONNX 版本（返回 3 个值）

    主项目的 MiniCPMModel.forward() 返回 (hidden_states, next_decoder_cache)
    ONNX 版本的 MiniCPMModel.forward() 返回 (hidden_states, next_keys, next_values)

    此函数创建一个新的 ONNX 版本模型并复制权重。
    """
    # 获取主项目模型的配置
    if hasattr(base_lm, 'config'):
        config = base_lm.config
    else:
        # 如果没有 config 属性，创建一个默认配置
        raise ValueError("无法获取 base_lm 的配置")

    # 创建 ONNX 版本的模型
    onnx_model = ONNXMiniCPMModel(config)
    onnx_model.eval()

    # 复制权重
    onnx_model.load_state_dict(base_lm.state_dict())

    return onnx_model


class _UnifiedCFMWithoutInferenceMode(nn.Module):
    """
    UnifiedCFM 的包装器，移除 @torch.inference_mode() 装饰器以支持 ONNX 导出。

    原始的 UnifiedCFM.forward() 使用 @torch.inference_mode()，这会在 ONNX 导出时导致错误：
    "Inference tensors cannot be saved for backward"

    这个包装器直接调用内部的 _forward_impl 方法，绕过 inference_mode 装饰器。
    """

    def __init__(self, original_cfm):
        super().__init__()
        self.original_cfm = original_cfm
        # 复制所有属性
        self.solver = original_cfm.solver
        self.sigma_min = original_cfm.sigma_min
        self.t_scheduler = original_cfm.t_scheduler
        self.training_cfg_rate = original_cfm.training_cfg_rate
        self.inference_cfg_rate = original_cfm.inference_cfg_rate
        self.reg_loss_type = original_cfm.reg_loss_type
        self.ratio_r_neq_t_range = original_cfm.ratio_r_neq_t_range
        self.noise_cond_prob_range = original_cfm.noise_cond_prob_range
        self.noise_cond_scale = original_cfm.noise_cond_scale
        self.in_channels = original_cfm.in_channels
        self.mean_mode = original_cfm.mean_mode
        self.estimator = original_cfm.estimator

    def forward(
        self,
        mu: torch.Tensor,
        n_timesteps: int,
        patch_size: int,
        cond: torch.Tensor,
        temperature: float = 1.0,
        cfg_value: float = 1.0,
        sway_sampling_coef: float = 1.0,
        use_cfg_zero_star: bool = True,
    ):
        """不使用 inference_mode 的 forward 方法"""
        b, _ = mu.shape
        t = patch_size
        z = torch.randn((b, self.in_channels, t), device=mu.device, dtype=mu.dtype) * temperature

        t_span = torch.linspace(1, 0, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        t_span = t_span + sway_sampling_coef * (torch.cos(torch.pi / 2 * t_span) - 1 + t_span)

        return self.original_cfm.solve_euler(
            x=z,
            t_span=t_span,
            mu=mu,
            cond=cond,
            cfg_value=cfg_value,
            use_cfg_zero_star=use_cfg_zero_star,
        )


class VoxCPMDecode(nn.Module):
    """
    Single-step autoregressive decode wrapper.

    Inputs:
      - dit_hidden: [b, h_dit] DiT hidden state from previous step (computed in Prefill or previous Decode)
      - base_next_keys, base_next_values: KV cache tensors from base LM
      - residual_next_keys, residual_next_values: KV cache tensors from residual LM
      - prefix_feat_cond: [b, p, d] conditioning feature from last patch
      - inference_timesteps: scalar tensor for diffusion steps
      - cfg_value: scalar tensor for CFG strength
      - noise: [b, p, d] external noise for diffusion (optional for ONNX export)

    Outputs:
      - pred_feat: [b, p, d] predicted patch features
      - new_dit_hidden: [b, h_dit] updated DiT hidden state for next decode step
      - new_base_next_keys, new_base_next_values
      - new_residual_next_keys, new_residual_next_values
      - stop_flag: bool tensor (True stop, False continue) computed at step end
    """

    def __init__(self, model: 'VoxCPMModel'):
        super().__init__()
        # submodules
        # 注意：使用 ONNX 版本的 MiniCPMModel（返回 3 个值），而不是主项目的版本
        self.base_lm = _convert_to_onnx_model(model.base_lm)
        self.residual_lm = _convert_to_onnx_model(model.residual_lm)
        self.feat_encoder = model.feat_encoder
        self.enc_to_lm_proj = model.enc_to_lm_proj
        self.lm_to_dit_proj = model.lm_to_dit_proj
        self.res_to_dit_proj = model.res_to_dit_proj
        # 使用包装器替换 feat_decoder，移除 inference_mode 装饰器以支持 ONNX 导出
        self.feat_decoder = _UnifiedCFMWithoutInferenceMode(model.feat_decoder)
        self.stop_proj = model.stop_proj
        self.stop_actn = model.stop_actn
        self.stop_head = model.stop_head
        self.fsq_layer = model.fsq_layer
        self.patch_size = model.patch_size

    def forward(
        self,
        dit_hidden: torch.Tensor,
        base_next_keys: torch.Tensor,
        base_next_values: torch.Tensor,
        residual_next_keys: torch.Tensor,
        residual_next_values: torch.Tensor,
        prefix_feat_cond: torch.Tensor,
        noise: torch.Tensor,
        inference_timesteps: torch.Tensor,
        cfg_value: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,  # pred_feat [b, p, d]
        torch.Tensor,  # new_dit_hidden [b, h_dit] - Updated DiT hidden for next decode step
        torch.Tensor,  # new_base_next_keys
        torch.Tensor,  # new_base_next_values
        torch.Tensor,  # new_residual_next_keys
        torch.Tensor,  # new_residual_next_values
        torch.Tensor,  # stop_flag (bool [b])
    ]:
        # # DiT hidden computation is now moved to prefill stage and end of decode step
        # # to avoid creating separate small models for ONNX export

        # 2) Diffusion decoder to predict next patch features using input DiT hidden
        # 注意：noise 参数在 ONNX 导出中不使用，因为 UnifiedCFM 内部会生成噪声
        pred_feat = self.feat_decoder(
            mu=dit_hidden,
            patch_size=self.patch_size,
            cond=prefix_feat_cond.transpose(1, 2).contiguous(),
            n_timesteps=inference_timesteps,
            cfg_value=cfg_value,
        ).transpose(1, 2)  # [b, p, d]
        batch_size = dit_hidden.shape[0]

        # 3) Encode predicted patch back to LM space (one step)
        curr_embed = self.feat_encoder(pred_feat.unsqueeze(1))  # [b, 1, c]
        curr_embed = self.enc_to_lm_proj(curr_embed)            # [b, 1, h]

        # 抽取设备
        device = curr_embed.device

        # 4) Base LM forward_step to update hidden and KV cache (ONNX-friendly mask)
        current_seq_len = base_next_keys.shape[3]
        # 优化：明确指定数据类型，提高GPU fp16环境下的精度一致性
        position_id = torch.tensor([current_seq_len], dtype=torch.int32, device=device)
        # 优化：使用浮点掩码替代布尔掩码，提高GPU fp16稳定性
        attn_mask = torch.ones((batch_size, 1, 1, current_seq_len + 1), dtype=torch.bool, device=device)

        # 将curr_embed从 [b, 1, h] 展平为 [b, h]，避免重复计算
        curr_embed_flat = curr_embed.squeeze(1)

        new_base_hidden, new_base_keys, new_base_values = self.base_lm.forward_step(
            curr_embed_flat, position_id, base_next_keys, base_next_values, attn_mask
        )

        # Concatenate new KV to existing cache
        new_base_next_keys = torch.cat([base_next_keys, new_base_keys], dim=3)
        new_base_next_values = torch.cat([base_next_values, new_base_values], dim=3)

        # Apply FSQ to base hidden for residual conditioning
        new_base_hidden = self.fsq_layer(new_base_hidden)

        # 5) Residual LM forward_step to update hidden and KV cache (ONNX-friendly mask)
        current_residual_seq_len = residual_next_keys.shape[3]
        # 优化：明确指定数据类型，提高GPU fp16环境下的精度一致性
        residual_position_id = torch.tensor([current_residual_seq_len], dtype=torch.int32, device=device)
        # 优化：使用浮点掩码替代布尔掩码，提高GPU fp16稳定性
        residual_attn_mask = torch.ones((batch_size, 1, 1, current_residual_seq_len + 1), dtype=torch.bool, device=device)

        new_residual_hidden, new_residual_keys, new_residual_values = self.residual_lm.forward_step(
            new_base_hidden + curr_embed_flat, residual_position_id, residual_next_keys, residual_next_values, residual_attn_mask
        )
        new_residual_next_keys = torch.cat([residual_next_keys, new_residual_keys], dim=3)
        new_residual_next_values = torch.cat([residual_next_values, new_residual_values], dim=3)

        # 6) Stop prediction placed at the end of the step (use updated base hidden)
        # 优化：使用温度缩放和阈值判断替代argmax，提高fp16稳定性
        stop_logits = self.stop_head(self.stop_actn(self.stop_proj(new_base_hidden)))
        # 温度缩放提高数值稳定性，温度值0.1使分布更尖锐
        stop_probs = torch.softmax(stop_logits / 0.1, dim=-1)
        # 使用阈值判断替代硬性argmax，减少fp16精度问题
        stop_flag = (stop_probs[:, 1] > 0.5)  # bool [b]

        # 7) Compute updated DiT hidden state for next decode step
        dit_hidden_1 = self.lm_to_dit_proj(new_base_hidden)  # [b, h_dit]
        dit_hidden_2 = self.res_to_dit_proj(new_residual_hidden)  # [b, h_dit]
        new_dit_hidden = dit_hidden_1 + dit_hidden_2

        return (
            pred_feat,
            new_dit_hidden,
            new_base_next_keys,
            new_base_next_values,
            new_residual_next_keys,
            new_residual_next_values,
            stop_flag,
        )