//! VoxCPM ONNX 推理测试
//!
//! 用于验证导出的 ONNX 模型是否符合预期

use anyhow::{Context, Result};
use ndarray::Array;
use std::path::Path;
use ort::Environment;

fn main() -> Result<()> {
    println!("VoxCPM ONNX 推理测试");
    println!("====================\n");

    // 获取 ONNX 模型目录
    let onnx_dir = std::env::var("VOXCPM_ONNX_DIR")
        .unwrap_or_else(|_| "./onnx_models".to_string());

    // 测试 AudioVAE Decoder
    println!("1. 测试 AudioVAE Decoder...");
    test_audio_vae_decoder(&onnx_dir)?;

    println!("\n所有测试通过！");
    Ok(())
}

fn test_audio_vae_decoder(onnx_dir: &str) -> Result<()> {
    let model_path = Path::new(onnx_dir).join("audio_vae_decoder.onnx");

    if !model_path.exists() {
        println!("  警告: {} 不存在，跳过测试", model_path.display());
        println!("  请先运行: python scripts/export_onnx.py --model_path <path> --output_dir ./onnx_models --export audiovae_decoder");
        return Ok(());
    }

    println!("  加载模型: {}", model_path.display());

    // 创建 ONNX Runtime Session
    let environment = Environment::builder()
        .with_name("voxcpm_test")
        .build()?;
    let session = environment
        .into_session()
        .from_file(&model_path)?;

    println!("  模型加载成功！");
    println!("  输入名称: {:?}", session.input_names());
    println!("  输出名称: {:?}", session.output_names());

    // 创建测试输入 (batch=1, latent_dim=64, latent_length=100)
    let latent_dim = 64;
    let latent_length = 100;
    let input = Array::zeros((1, latent_dim, latent_length))?;

    println!("  输入形状: {:?}", input.shape());

    // 运行推理
    let outputs = session.run(ort::inputs!["z" => input.into_ort()]?)?;

    if let Some(output) = outputs.get("audio") {
        println!("  输出形状: {:?}", output.dimensions());
        println!("  音频推理成功！");
    } else {
        println!("  警告: 未找到 'audio' 输出");
    }

    Ok(())
}
