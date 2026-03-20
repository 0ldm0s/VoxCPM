use anyhow::Result;
use ort::{session::Session, value::Tensor};
use std::path::Path;

fn main() -> Result<()> {
    println!("🎯 最小 ONNX API 验证测试");
    println!("{}\n", "=".repeat(50));

    // 获取 ONNX 模型目录
    let onnx_dir = std::env::var("VOXCPM_ONNX_DIR")
        .unwrap_or_else(|_| "../../onnx_models".to_string());

    let onnx_path = Path::new(&onnx_dir);
    if !onnx_path.exists() {
        println!("❌ ONNX 模型目录不存在: {}", onnx_dir);
        return Ok(());
    }

    println!("📁 ONNX 模型目录: {}", onnx_dir);

    // 测试 AudioVAE Decoder (最小的模型)
    let model_path = onnx_path.join("audio_vae_decoder.onnx");
    if !model_path.exists() {
        println!("⚠️  跳过: {} 不存在", model_path.display());
        return Ok(());
    }

    println!("📂 测试模型: {}", model_path.display());

    // 加载模型
    println!("🔧 加载模型...");
    let mut session = Session::builder()?
        .commit_from_file(&model_path)?;

    println!("✅ 模型加载成功!");

    // 显示模型信息
    println!("\n📊 模型信息:");
    println!("  输入:");
    for input in session.inputs().iter() {
        println!("    - {} ({:?})", input.name(), input.dtype());
    }
    println!("  输出:");
    for output in session.outputs().iter() {
        println!("    - {} ({:?})", output.name(), output.dtype());
    }

    // 创建简单的测试输入
    let latent_dim = 64;
    let latent_length = 10;

    // 创建输入数据: z (batch=1, latent_dim=64, latent_length=10)
    // 使用 (shape, data) 元组格式
    let z_data: Vec<f32> = vec![0.0; 1 * latent_dim * latent_length];
    let z_shape = vec![1, latent_dim as i64, latent_length as i64];

    println!("\n🎵 测试输入:");
    println!("  z.shape = {:?}", z_shape);

    // 运行推理
    println!("\n🚀 运行推理...");

    // 创建输入 tensor
    let z_tensor = Tensor::from_array((z_shape.clone(), z_data))?;

    // 运行推理
    let outputs = session.run(ort::inputs![z_tensor])?;

    // 检查输出
    if let Some(audio) = outputs.get("audio") {
        let (shape, _data) = audio.try_extract_tensor::<f32>()?;
        println!("  输出形状: {:?}", shape);
        println!("  ✅ 推理成功!");
    }

    println!("\n✅ 最小测试完成!");
    Ok(())
}
