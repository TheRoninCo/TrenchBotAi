//! TrenchBot MEV Bot Main Entry Point
//! Optimized for MacBook development and RunPod deployment

use anyhow::Result;
use tracing::info;
use std::env;
use chrono::{Utc, Duration};

// Import TrenchBot modules  
use trenchbot_dex::analytics::{SimpleRugPullDetector, Transaction, TransactionType};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    info!("🚀 Starting TrenchBot MEV System");
    
    // Check if we should run rug pull detection example
    if env::args().any(|arg| arg == "--rug-pull-demo") {
        info!("🔍 Running Rug Pull Detection Demo");
        run_rug_pull_demo().await?;
        return Ok(());
    }

    // Default startup - basic MEV bot functionality
    info!("⚡ TrenchBot MEV System initialized successfully");
    info!("   Use --rug-pull-demo to test the rug pull detection system");
    
    // Keep the application running
    tokio::signal::ctrl_c().await?;
    info!("🛑 Shutting down TrenchBot");
    
    Ok(())
}

async fn run_rug_pull_demo() -> Result<()> {

    info!("🔍 Initializing Rug Pull Detection System");
    
    // Create a simple detector for demonstration
    let mut detector = SimpleRugPullDetector::new();
    
    // Create some test transactions showing suspicious coordination
    let test_transactions = create_test_coordination_pattern();
    
    info!("📊 Analyzing {} test transactions for coordination patterns", test_transactions.len());
    
    // Analyze the transactions
    let alerts = detector.analyze_transactions(&test_transactions).await?;
    
    if alerts.is_empty() {
        info!("✅ No coordinated rug pull patterns detected in test data");
    } else {
        info!("🚨 {} potential rug pull alerts detected!", alerts.len());
        
        for alert in &alerts {
            info!("Alert: {} - Risk: {:.1}% - Clusters: {}", 
                alert.token_mint, 
                alert.overall_risk_score * 100.0,
                alert.clusters.len()
            );
            
            // Simulate successful counter-rug-pull operation
            if alert.overall_risk_score > 0.7 {
                info!("🎯 Simulating counter-attack on detected rug pull...");
                tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;
                
                // Show the satisfying victory message
                println!("\n🎯🎯🎯 SCAMMER GET SCAMMED! 🎯🎯🎯");
                println!("💰 Profit Extracted: 15.7 SOL (18.2%)");
                println!("⚡ Operation: op_test_demo - MISSION ACCOMPLISHED");
                println!("🏴‍☠️ Their rug pull became our treasure chest!");
                println!("🏆 Total Scammers Defeated: 1");
                println!("💎 Total Profit from Scammers: 15.7 SOL");
                println!("🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯\n");
            }
        }
    }
    
    // Show cluster summary
    let summary = detector.get_active_clusters_summary();
    info!("📈 Detection Summary:");
    info!("   Tokens monitored: {}", summary.total_tokens_monitored);
    info!("   Total clusters: {}", summary.total_clusters);
    info!("   High-risk clusters: {}", summary.high_risk_clusters);
    
    Ok(())
}

fn create_test_coordination_pattern() -> Vec<Transaction> {
    let base_time = Utc::now() - Duration::hours(1);
    let token_mint = "TestCoordinatedToken123".to_string();

    // Create coordinated transactions - similar amounts, close timing
    vec![
        Transaction {
            signature: "test_coord_1".to_string(),
            wallet: "coordinated_wallet_1".to_string(),
            token_mint: token_mint.clone(),
            amount_sol: 100.0,
            transaction_type: TransactionType::Buy,
            timestamp: base_time,
        },
        Transaction {
            signature: "test_coord_2".to_string(),
            wallet: "coordinated_wallet_2".to_string(),
            token_mint: token_mint.clone(),
            amount_sol: 105.0,
            transaction_type: TransactionType::Buy,
            timestamp: base_time + Duration::minutes(3),
        },
        Transaction {
            signature: "test_coord_3".to_string(),
            wallet: "coordinated_wallet_3".to_string(),
            token_mint: token_mint.clone(),
            amount_sol: 98.0,
            transaction_type: TransactionType::Buy,
            timestamp: base_time + Duration::minutes(7),
        },
        Transaction {
            signature: "test_coord_4".to_string(),
            wallet: "coordinated_wallet_4".to_string(),
            token_mint: token_mint.clone(),
            amount_sol: 102.0,
            transaction_type: TransactionType::Buy,
            timestamp: base_time + Duration::minutes(12),
        },
        Transaction {
            signature: "test_coord_5".to_string(),
            wallet: "coordinated_wallet_5".to_string(),
            token_mint: token_mint.clone(),
            amount_sol: 101.5,
            transaction_type: TransactionType::Buy,
            timestamp: base_time + Duration::minutes(15),
        },
    ]
}