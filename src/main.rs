//! MEV Bot Main Entry Point
//! Optimized for MacBook development and RunPod deployment

use mev_bot::{
    core::{config::Config, hardware::HardwareManager, telemetry},
    orchestrator::TradingOrchestrator,
};
use anyhow::Result;
use tracing::info;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize telemetry
    telemetry::init_tracing()?;
    
    // Load configuration
    let config = Config::from_env()?;
    
    #[cfg(feature = "local-dev")]
    info!("🖥️ Starting MEV Bot in MacBook development mode");
    
    #[cfg(feature = "gpu")]
    info!("🚀 Starting MEV Bot in RunPod GPU mode");
    
    // Initialize hardware
    let _hardware = HardwareManager::new(config.hardware.clone()).await?;
    
    // Start trading orchestrator
    let mut orchestrator = TradingOrchestrator::new(config).await?;
    orchestrator.run_trading_loop().await?;
    
    Ok(())
}
use mev_bot::observability::{
    OmniLoggerBuilder,
    init_combat_logger, CombatContext,
    ops::{recon, engage},
    severity::info
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize OmniLogger
    let logger = OmniLoggerBuilder::new()
        .with_config("logging.toml")
        .build()
        .await?;

    // Initialize combat logger with default context
    init_combat_logger(
        logger,
        CombatContext {
            operation_id: "midnight_sun".into(),
            squad: "alpha".into(),
            ..Default::default()
        }
    )?;

    // Log combat operations
    recon("drone", (34.0522, -118.2437), json!({"target": "base"}), None).await?;
    engage("artillery", 3, json!({"coordinates": [34.0522, -118.2437]}), None).await?;

    // Traditional logging
    info("system", "startup", json!({"status": "online"}), None).await?;

    Ok(())
}use sonar::{LiveFeed, Quarantine, WhaleScorer};
use solana_sdk::pubkey;

#[tokio::main]
async fn main() {
    // Load config
    let config = SonarConfig::load("configs/sonar.toml");
    
    // Initialize core systems
    let (tx, rx) = tokio::sync::mpsc::channel(1000);
    let mut feed = LiveFeed::new(&config.rpc.jito, vec![spl_token::ID], tx);
    let scorer = WhaleScorer::new(&config.ml.model_path);
    let quarantine = Quarantine::new(config.blacklist);
    
    // Launch subsystems
    tokio::spawn(async move { feed.run().await });
    
    // Process stream
    while let Some(tx) = rx.recv().await {
        let wallet = tx.get_wallet();
        let score = scorer.score(&wallet);
        
        if score > config.blacklist.auto_blacklist_threshold {
            quarantine.flag(wallet, Violation::ML(score));
        }
    }
    let trainer = WolframTrainer::new(
        "scripts/analyze_batches.wl",
        "model_outputs",
        24, // Retrain daily
    );
    tokio::spawn(async move { trainer.start().await });use observability::metrics::{register_metrics, start_metrics_server};
    use prometheus::Registry;
    use std::net::SocketAddr;
    
    #[tokio::main]
    async fn main() {
        // 1. Initialize metrics
        let registry = Registry::new();
        register_metrics(&registry).expect("Failed to register metrics");
        
        // 2. Start metrics server
        let metrics_addr: SocketAddr = "0.0.0.0:9090".parse().unwrap();
        let _metrics_handle = start_metrics_server(metrics_addr, registry.clone());
        
        // 3. Example metric usage
        metrics::LOGS_PROCESSED.with_label_values(&["combat"]).inc();
        
        // ... rest of your application
    }// In src/main.rs
#[derive(clap::Subcommand)]
enum Commands {
    ValidateConfig { path: Option<PathBuf> }
}// src/main.rs or src/lib.rs
use trenchie_core::{killfeed::{Killfeed, KillfeedBuilder, WeaponType, ChainType}, prometheus};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
  // src/main.rs or src/lib.rs
use trenchie_core::{killfeed::{Killfeed, KillfeedBuilder, WeaponType, ChainType}, prometheus};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // ---- 1. Initialize Killfeed ----
    let killfeed = Killfeed::builder()
        .enable_prometheus()  // Only if using Prometheus
        .enable_gpu()         // Only if GPU processing needed
        .build()?;

    // ---- 2. Set Up GPU Consumer (if enabled) ----
    #[cfg(feature = "gpu")]
    {
        let gpu_tx = killfeed.get_gpu_sender(); // Get the channel sender
        std::thread::spawn(move || {
            let gpu_processor = GpuProcessor::new();
            while let Ok(event) = gpu_rx.recv() {
                gpu_processor.predict(event);
            }
        });
    }

    // ---- 3. Start Metrics Server (Prometheus) ----
    let metrics_handle = prometheus::start_metrics_server("0.0.0.0:8080")?;

    // ---- 4. Your Main Application Loop ----
    let app_state = AppState { killfeed };
    axum::Server::bind(&"0.0.0.0:3000".parse()?)
        .serve(app_state.into_make_service())
        .await?;

    Ok(())
}