//! GPU Training Demo - Complete AI Training Pipeline
//! 
//! Demonstrates the full GPU training pipeline with all AI models:
//! - Spiking Neural Networks
//! - Quantum-Inspired Graph Neural Networks  
//! - Causal Inference Models
//! - Flash Attention Transformers

use trenchbot_ai::ai_training::{
    AITrainingPipeline, TrainingConfig, DeviceConfig,
    training_data_generator::TrainingDataGenerator,
};
use anyhow::Result;
use std::path::Path;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    tracing::info!("🚀 TrenchBot AI - GPU Training Pipeline Demo");
    tracing::info!("============================================");
    
    // Configure GPU training
    let config = TrainingConfig {
        device: DeviceConfig::CUDA(0), // Use first CUDA GPU
        batch_size: 16,               // Smaller batch for demo
        learning_rate: 0.001,
        epochs: 50,                   // Reduced epochs for demo
        save_interval: 10,
        model_path: "models/demo/".to_string(),
    };
    
    tracing::info!("📊 Training Configuration:");
    tracing::info!("  Device: {:?}", config.device);
    tracing::info!("  Batch size: {}", config.batch_size);
    tracing::info!("  Learning rate: {}", config.learning_rate);
    tracing::info!("  Epochs: {}", config.epochs);
    
    // Generate training data
    tracing::info!("\n🎲 Generating Training Data...");
    let mut data_generator = TrainingDataGenerator::new(42);
    let training_data = data_generator.generate_balanced_dataset(1000)?; // 1K examples for demo
    
    // Initialize AI training pipeline
    tracing::info!("\n🧠 Initializing AI Training Pipeline...");
    let mut pipeline = AITrainingPipeline::new(config)?;
    pipeline.initialize_trainers().await?;
    
    // Train all models
    tracing::info!("\n🔥 Starting Multi-Model Training...");
    let results = pipeline.train_all_models(&training_data).await?;
    
    // Print results
    results.print_summary();
    
    // Save trained models
    tracing::info!("\n💾 Saving Trained Models...");
    let model_path = Path::new("models/demo");
    std::fs::create_dir_all(model_path)?;
    pipeline.save_models(model_path).await?;
    
    // Demonstrate ensemble prediction
    tracing::info!("\n🏆 Ensemble Performance Analysis:");
    let ensemble_perf = results.ensemble_performance();
    tracing::info!("  Weighted Accuracy: {:.3}", ensemble_perf.accuracy);
    tracing::info!("  Weighted F1 Score: {:.3}", ensemble_perf.f1_score);
    tracing::info!("  Total Training Time: {:.2}s", ensemble_perf.training_time.as_secs_f64());
    
    // Performance breakdown by model
    tracing::info!("\n📈 Individual Model Breakdown:");
    tracing::info!("  🧠 Spiking NN:");
    tracing::info!("    - Accuracy: {:.3}", results.spiking_performance.accuracy);
    tracing::info!("    - F1 Score: {:.3}", results.spiking_performance.f1_score);
    tracing::info!("    - Training Time: {:.2}s", results.spiking_performance.training_time.as_secs_f64());
    
    tracing::info!("  🌀 Quantum GNN:");
    tracing::info!("    - Accuracy: {:.3}", results.quantum_gnn_performance.accuracy);
    tracing::info!("    - F1 Score: {:.3}", results.quantum_gnn_performance.f1_score);
    tracing::info!("    - Training Time: {:.2}s", results.quantum_gnn_performance.training_time.as_secs_f64());
    
    tracing::info!("  🔍 Causal Inference:");
    tracing::info!("    - Accuracy: {:.3}", results.causal_performance.accuracy);
    tracing::info!("    - F1 Score: {:.3}", results.causal_performance.f1_score);
    tracing::info!("    - Training Time: {:.2}s", results.causal_performance.training_time.as_secs_f64());
    
    tracing::info!("  ⚡ Flash Attention:");
    tracing::info!("    - Accuracy: {:.3}", results.flash_attention_performance.accuracy);
    tracing::info!("    - F1 Score: {:.3}", results.flash_attention_performance.f1_score);
    tracing::info!("    - Training Time: {:.2}s", results.flash_attention_performance.training_time.as_secs_f64());
    
    tracing::info!("\n✅ GPU Training Pipeline Demo Complete!");
    tracing::info!("🎯 All models trained and saved successfully");
    tracing::info!("🚀 Ready for deployment on RunPod A100 GPUs!");
    
    Ok(())
}