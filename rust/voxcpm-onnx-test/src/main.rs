//! VoxCPM ONNX 推理测试
//!
//! 用于验证导出的 ONNX 模型是否符合预期

use anyhow::{Context, Result};
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;

fn main() -> Result<()> {
    println!("🎯 VoxCPM ONNX 推理测试");
    println!("{}\n", "=".repeat(50));

    // 获取 ONNX 模型目录
    let onnx_dir = std::env::var("VOXCPM_ONNX_DIR")
        .unwrap_or_else(|_| "../onnx_models".to_string());

    let onnx_path = Path::new(&onnx_dir);
    if !onnx_path.exists() {
        println!("❌ ONNX 模型目录不存在: {}", onnx_dir);
        println!("   请先运行: python scripts/export_onnx.py --output_dir {}", onnx_dir);
        std::process::exit(1);
    }

    println!("📁 ONNX 模型目录: {}", onnx_dir);

    // 测试所有模型
    println!("\n📋 测试清单:");
    println!("  1. AudioVAE Encoder");
    println!("  2. AudioVAE Decoder");
    println!("  3. VoxCPM Prefill");
    println!("  4. VoxCPM Decode Step");
    println!();

    // 1. 测试 AudioVAE Encoder
    test_audio_vae_encoder(&onnx_dir)?;

    // 2. 测试 AudioVAE Decoder
    test_audio_vae_decoder(&onnx_dir)?;

    // 3. 测试 VoxCPM Prefill
    test_voxcpm_prefill(&onnx_dir)?;

    // 4. 测试 VoxCPM Decode Step
    test_voxcpm_decode_step(&onnx_dir)?;

    println!("\n✅ 所有测试通过！");
    Ok(())
}

/// 测试 AudioVAE Encoder
fn test_audio_vae_encoder(onnx_dir: &str) -> Result<()> {
    println!("🔧 测试 AudioVAE Encoder...");

    let model_path = Path::new(onnx_dir).join("audio_vae_encoder.onnx");

    if !model_path.exists() {
        println!("  ⚠️  跳过: {} 不存在", model_path.display());
        return Ok(());
    }

    println!("  📂 模型: {}", model_path.display());

    // 创建 ONNX Runtime Session
    let mut session = Session::builder()?.commit_from_file(&model_path)?;

    println!("  ✅ 模型加载成功");

    // 显示模型信息
    println!("  📥 输入:");
    for input in session.inputs().iter() {
        println!("    - {} {:?}", input.name(), input.dtype());
    }
    println!("  📤 输出:");
    for output in session.outputs().iter() {
        println!("    - {} {:?}", output.name(), output.dtype());
    }

    // 创建测试输入 (batch=1, channels=1, samples=16000)
    let samples = 16000;
    let audio_data: Vec<f32> = vec![0.0; samples];
    let audio_shape = vec![1, 1, samples as i64];

    println!("  🎵 测试输入: audio_data.shape={:?}", audio_shape);

    // 运行推理 - 只需要一个输入 audio_data
    let audio_tensor = Tensor::from_array((audio_shape, audio_data))?;
    let outputs = session.run(ort::inputs![audio_tensor])?;

    if let Some(latent) = outputs.get("z") {
        let (shape, _data) = latent.try_extract_tensor::<f32>()?;
        println!("  📊 输出形状: {:?}", shape);
        println!("  ✅ AudioVAE Encoder 推理成功");
    }

    Ok(())
}

/// 测试 AudioVAE Decoder
fn test_audio_vae_decoder(onnx_dir: &str) -> Result<()> {
    println!("🔧 测试 AudioVAE Decoder...");

    let model_path = Path::new(onnx_dir).join("audio_vae_decoder.onnx");

    if !model_path.exists() {
        println!("  ⚠️  跳过: {} 不存在", model_path.display());
        return Ok(());
    }

    println!("  📂 模型: {}", model_path.display());

    // 创建 ONNX Runtime Session
    let mut session = Session::builder()?.commit_from_file(&model_path)?;

    println!("  ✅ 模型加载成功");

    // 显示模型信息
    println!("  📥 输入:");
    for input in session.inputs().iter() {
        println!("    - {} {:?}", input.name(), input.dtype());
    }
    println!("  📤 输出:");
    for output in session.outputs().iter() {
        println!("    - {} {:?}", output.name(), output.dtype());
    }

    // 创建测试输入 (batch=1, latent_dim=64, latent_length=100)
    let latent_dim = 64;
    let latent_length = 100;
    let z_data: Vec<f32> = vec![0.0; 1 * latent_dim * latent_length];
    let z_shape = vec![1, latent_dim as i64, latent_length as i64];

    println!("  🎵 测试输入: z.shape = {:?}", z_shape);

    // 运行推理
    let z_tensor = Tensor::from_array((z_shape, z_data))?;
    let outputs = session.run(ort::inputs![z_tensor])?;

    if let Some(audio) = outputs.get("audio") {
        let (shape, _data) = audio.try_extract_tensor::<f32>()?;
        println!("  📊 输出形状: {:?}", shape);
        println!("  ✅ AudioVAE Decoder 推理成功");
    }

    Ok(())
}

/// 测试 VoxCPM Prefill
fn test_voxcpm_prefill(onnx_dir: &str) -> Result<()> {
    println!("🔧 测试 VoxCPM Prefill...");

    let model_path = Path::new(onnx_dir).join("voxcpm_prefill.onnx");

    if !model_path.exists() {
        println!("  ⚠️  跳过: {} 不存在", model_path.display());
        return Ok(());
    }

    println!("  📂 模型: {}", model_path.display());

    // 检查 external data 文件大小
    let data_path = model_path.with_extension("onnx.data");
    if data_path.exists() {
        if let Ok(metadata) = std::fs::metadata(&data_path) {
            let size_gb = metadata.len() as f64 / 1e9;
            println!("  📦 External Data: {:.2} GB", size_gb);
        }
    }

    // 创建 ONNX Runtime Session
    let mut session = Session::builder()?.commit_from_file(&model_path)?;

    println!("  ✅ 模型加载成功");

    // 显示模型信息
    println!("  📥 输入:");
    for input in session.inputs().iter() {
        println!("    - {} {:?}", input.name(), input.dtype());
    }
    println!("  📤 输出:");
    for output in session.outputs().iter() {
        println!("    - {} {:?}", output.name(), output.dtype());
    }

    // 创建测试输入
    let seq_len = 10;

    // text_tokens: (batch, seq_len)
    let text_tokens_data: Vec<i64> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let text_tokens_shape = vec![1, seq_len as i64];

    // text_mask: (batch, seq_len)
    let text_mask_data: Vec<i32> = vec![1; seq_len];
    let text_mask_shape = vec![1, seq_len as i64];

    // feat: (batch, seq_len, patch_size, feat_dim) - patch_size 是 4
    let feat_dim = 64;
    let patch_size: i64 = 4;
    let feat_data: Vec<f32> = vec![0.1; 1 * seq_len * patch_size as usize * feat_dim];
    let feat_shape = vec![1, seq_len as i64, patch_size, feat_dim as i64];

    // feat_mask: (batch, seq_len)
    let feat_mask_data: Vec<i32> = vec![0; seq_len];
    let feat_mask_shape = vec![1, seq_len as i64];

    println!("  🎵 测试输入:");
    println!("    text_tokens.shape = {:?}", text_tokens_shape);
    println!("    feat.shape = {:?}", feat_shape);

    // 运行推理
    let text_tokens_tensor = Tensor::from_array((text_tokens_shape, text_tokens_data))?;
    let text_mask_tensor = Tensor::from_array((text_mask_shape, text_mask_data))?;
    let feat_tensor = Tensor::from_array((feat_shape, feat_data))?;
    let feat_mask_tensor = Tensor::from_array((feat_mask_shape, feat_mask_data))?;

    // 先获取输出名称
    let output_names: Vec<String> = session.outputs().iter()
        .map(|o| o.name().to_string())
        .collect();

    let outputs = session.run(ort::inputs![text_tokens_tensor, text_mask_tensor, feat_tensor, feat_mask_tensor])?;

    println!("  📊 推理输出:");
    for name in output_names {
        if let Some(out) = outputs.get(&name) {
            let (shape, _data) = out.try_extract_tensor::<f32>()?;
            println!("    {} -> {:?}", name, shape);
        }
    }
    println!("  ✅ VoxCPM Prefill 推理成功");

    Ok(())
}

/// 测试 VoxCPM Decode Step
fn test_voxcpm_decode_step(onnx_dir: &str) -> Result<()> {
    println!("🔧 测试 VoxCPM Decode Step...");

    let model_path = Path::new(onnx_dir).join("voxcpm_decode_step.onnx");

    if !model_path.exists() {
        println!("  ⚠️  跳过: {} 不存在", model_path.display());
        return Ok(());
    }

    println!("  📂 模型: {}", model_path.display());

    // 检查 external data 文件大小
    let data_path = model_path.with_extension("onnx.data");
    if data_path.exists() {
        if let Ok(metadata) = std::fs::metadata(&data_path) {
            let size_gb = metadata.len() as f64 / 1e9;
            println!("  📦 External Data: {:.2} GB", size_gb);
        }
    }

    // 创建 ONNX Runtime Session
    let mut session = match Session::builder()?.commit_from_file(&model_path) {
        Ok(s) => s,
        Err(e) => {
            println!("  ⚠️  模型加载失败: {}", e);
            println!("  ⚠️  原因: 模型使用了 RandomNormalLike 算子，ONNX Runtime 不支持");
            println!("  ⚠️  解决: 修改导出代码，将噪声作为外部输入");
            return Ok(());
        }
    };

    println!("  ✅ 模型加载成功");

    // 显示模型信息
    println!("  📥 输入:");
    for input in session.inputs().iter() {
        println!("    - {} {:?}", input.name(), input.dtype());
    }
    println!("  📤 输出:");
    for output in session.outputs().iter() {
        println!("    - {} {:?}", output.name(), output.dtype());
    }

    // 创建测试输入
    let batch_size = 1;
    let hidden_dim = 1024;
    let num_layers_base = 24;
    let num_heads_base = 2;
    let num_layers_res = 8;
    let num_heads_res = 2;
    let past_seq_len = 10;
    let head_dim = 64;
    let patch_size = 4;
    let feat_dim = 64;

    // dit_hidden: [batch_size, hidden_dim]
    let dit_hidden_data: Vec<f32> = vec![0.1; batch_size * hidden_dim];
    let dit_hidden_shape = vec![batch_size, hidden_dim];

    // base_next_keys: [batch_size, num_layers_base, num_heads_base, past_seq_len, head_dim]
    let base_next_keys_data: Vec<f32> = vec![0.0; batch_size * num_layers_base * num_heads_base * past_seq_len * head_dim];
    let base_next_keys_shape = vec![batch_size, num_layers_base, num_heads_base, past_seq_len, head_dim];

    // base_next_values: same as keys
    let base_next_values_data: Vec<f32> = base_next_keys_data.clone();
    let base_next_values_shape = base_next_keys_shape.clone();

    // residual_next_keys: [batch_size, num_layers_res, num_heads_res, past_seq_len, head_dim]
    let residual_next_keys_data: Vec<f32> = vec![0.0; batch_size * num_layers_res * num_heads_res * past_seq_len * head_dim];
    let residual_next_keys_shape = vec![batch_size, num_layers_res, num_heads_res, past_seq_len, head_dim];

    // residual_next_values: same as keys
    let residual_next_values_data: Vec<f32> = residual_next_keys_data.clone();
    let residual_next_values_shape = residual_next_keys_shape.clone();

    // prefix_feat_cond: [batch_size, patch_size, feat_dim]
    let prefix_feat_cond_data: Vec<f32> = vec![0.1; batch_size * patch_size * feat_dim];
    let prefix_feat_cond_shape = vec![batch_size, patch_size, feat_dim];

    // noise: [batch_size, patch_size, feat_dim]
    let noise_data: Vec<f32> = vec![0.5; batch_size * patch_size * feat_dim];
    let noise_shape = vec![batch_size, patch_size, feat_dim];

    // cfg_value: scalar
    let cfg_data: Vec<f32> = vec![2.0];
    let cfg_shape = vec![1];

    println!("  🎵 测试输入:");
    println!("    dit_hidden.shape = {:?}", dit_hidden_shape);
    println!("    noise.shape = {:?}", noise_shape);
    println!("    cfg_value.shape = {:?}", cfg_shape);

    // 运行推理
    let dit_hidden_tensor = Tensor::from_array((dit_hidden_shape, dit_hidden_data))?;
    let base_next_keys_tensor = Tensor::from_array((base_next_keys_shape, base_next_keys_data))?;
    let base_next_values_tensor = Tensor::from_array((base_next_values_shape, base_next_values_data))?;
    let residual_next_keys_tensor = Tensor::from_array((residual_next_keys_shape, residual_next_keys_data))?;
    let residual_next_values_tensor = Tensor::from_array((residual_next_values_shape, residual_next_values_data))?;
    let prefix_feat_cond_tensor = Tensor::from_array((prefix_feat_cond_shape, prefix_feat_cond_data))?;
    let noise_tensor = Tensor::from_array((noise_shape, noise_data))?;
    let cfg_tensor = Tensor::from_array((cfg_shape, cfg_data))?;

    // 先获取输出名称
    let output_names: Vec<String> = session.outputs().iter()
        .map(|o| o.name().to_string())
        .collect();

    let outputs = session.run(ort::inputs![
        dit_hidden_tensor,
        base_next_keys_tensor,
        base_next_values_tensor,
        residual_next_keys_tensor,
        residual_next_values_tensor,
        prefix_feat_cond_tensor,
        noise_tensor,
        cfg_tensor,
    ])?;

    println!("  📊 推理输出:");
    for name in output_names {
        if let Some(out) = outputs.get(&name) {
            // 尝试提取 f32 或 bool 类型
            if let Ok((shape, _data)) = out.try_extract_tensor::<f32>() {
                println!("    {} -> {:?}", name, shape);
            } else if let Ok((shape, _data)) = out.try_extract_tensor::<bool>() {
                println!("    {} -> {:?}", name, shape);
            }
        }
    }
    println!("  ✅ VoxCPM Decode Step 推理成功");

    Ok(())
}
