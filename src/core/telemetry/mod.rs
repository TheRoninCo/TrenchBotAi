//! Telemetry and monitoring
use anyhow::Result;

pub fn init_tracing() -> Result<()> {
    let log_level = std::env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string());
    
    tracing_subscriber::fmt()
        .with_env_filter(log_level)
        .with_target(false)
        .with_thread_ids(true)
        .init();
        
    Ok(())
}

pub fn log_sandwich_opportunity(profit: f64, confidence: f64) {
    tracing::info!("💰 Sandwich opportunity: ${:.2} profit, {:.1}% confidence", profit, confidence * 100.0);
}

pub fn log_execution_result(success: bool, profit: Option<f64>) {
    if success {
        tracing::info!("✅ Trade executed successfully: ${:.2}", profit.unwrap_or(0.0));
    } else {
        tracing::warn!("❌ Trade execution failed");
    }
}
