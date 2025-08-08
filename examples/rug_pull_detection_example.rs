//! Rug Pull Detection System Usage Examples
//! 
//! This example demonstrates how to use the coordinated rug pull detection system
//! to analyze early investor behavior and detect potential coordinated attacks.

use anyhow::Result;
use chrono::{DateTime, Utc, Duration};
use trenchbot_dex::analytics::{
    RugPullSystemBuilder, SimpleRugPullDetector, Transaction, TransactionType,
    RiskLevel, RugPullAlert
};

#[tokio::main]
async fn main() -> Result<()> {
    println!("🔍 TrenchBot Rug Pull Detection System Examples\n");

    // Example 1: Simple analysis of transaction batch
    simple_analysis_example().await?;

    // Example 2: Full monitoring system with alerts
    // full_monitoring_example().await?;

    // Example 3: Custom configuration
    custom_configuration_example().await?;

    Ok(())
}

/// Example 1: Simple batch analysis
async fn simple_analysis_example() -> Result<()> {
    println!("📊 Example 1: Simple Batch Analysis");
    println!("=====================================\n");

    // Create a simple detector
    let mut detector = SimpleRugPullDetector::new();

    // Create some test transactions that show coordination patterns
    let suspicious_transactions = create_suspicious_transaction_pattern();
    let normal_transactions = create_normal_transaction_pattern();

    println!("Analyzing {} suspicious transactions...", suspicious_transactions.len());
    let suspicious_alerts = detector.analyze_transactions(&suspicious_transactions).await?;
    
    println!("Analyzing {} normal transactions...", normal_transactions.len());
    let normal_alerts = detector.analyze_transactions(&normal_transactions).await?;

    // Display results
    println!("\n🚨 Suspicious Transaction Results:");
    for alert in &suspicious_alerts {
        display_alert_summary(alert);
    }

    println!("\n✅ Normal Transaction Results:");
    if normal_alerts.is_empty() {
        println!("  No coordinated patterns detected (as expected)");
    } else {
        for alert in &normal_alerts {
            display_alert_summary(alert);
        }
    }

    // Show cluster summary
    let summary = detector.get_active_clusters_summary();
    println!("\n📈 Cluster Summary:");
    println!("  Tokens monitored: {}", summary.total_tokens_monitored);
    println!("  Total clusters: {}", summary.total_clusters);
    println!("  High-risk clusters: {}", summary.high_risk_clusters);

    println!("\n" + "=".repeat(50).as_str() + "\n");
    Ok(())
}

/// Example 2: Full monitoring system (commented out to avoid long-running process)
#[allow(dead_code)]
async fn full_monitoring_example() -> Result<()> {
    println!("🚀 Example 2: Full Monitoring System");
    println!("====================================\n");

    // Build a complete monitoring system with alerts
    let system = RugPullSystemBuilder::new()
        .with_risk_threshold(0.6) // Lower threshold for demo
        .with_cooldown_minutes(5) // Short cooldown for demo
        .build()
        .await?;

    // Subscribe to events
    let mut event_receiver = system.subscribe_to_events();

    println!("✅ Monitoring system started!");
    println!("📡 Listening for rug pull events...\n");

    // In a real application, this would run continuously
    // For the example, we'll just show system health
    let health = system.get_system_health().await;
    println!("🏥 System Health:");
    println!("  Status: {:?}", health.overall_status);
    println!("  Monitor: {}", if health.monitor_healthy { "✅" } else { "❌" });
    println!("  Alerts: {}", if health.alert_system_healthy { "✅" } else { "❌" });
    println!("  Uptime: {} seconds", health.uptime_seconds);

    // Listen for a few events (in practice this would run indefinitely)
    println!("\n⏳ Waiting for events (timeout after 5 seconds)...");
    match tokio::time::timeout(std::time::Duration::from_secs(5), event_receiver.recv()).await {
        Ok(Ok(event)) => {
            println!("📨 Received event: {:?}", event.event_type);
        }
        Ok(Err(_)) => {
            println!("📡 Event channel closed");
        }
        Err(_) => {
            println!("⏰ No events received within timeout");
        }
    }

    // Shutdown gracefully
    system.shutdown().await?;
    println!("🛑 System shutdown complete");

    println!("\n" + "=".repeat(50).as_str() + "\n");
    Ok(())
}

/// Example 3: Custom configuration
async fn custom_configuration_example() -> Result<()> {
    println!("⚙️  Example 3: Custom Configuration");
    println!("==================================\n");

    // Show different configuration options
    let config_examples = vec![
        ("High Security", 0.9, 120),
        ("Balanced", 0.7, 60),
        ("Sensitive", 0.5, 30),
    ];

    for (name, threshold, cooldown) in config_examples {
        println!("🔧 {} Configuration:", name);
        println!("   Risk Threshold: {:.1}%", threshold * 100.0);
        println!("   Alert Cooldown: {} minutes", cooldown);
        
        // This would create a system with these settings:
        // let _system = RugPullSystemBuilder::new()
        //     .with_risk_threshold(threshold)
        //     .with_cooldown_minutes(cooldown)
        //     .build()
        //     .await?;
        
        println!("   ✅ Configuration validated\n");
    }

    println!("💡 Integration Options:");
    println!("   🤖 Telegram: Set TELEGRAM_CHAT_ID environment variable");
    println!("   💬 Discord: Set DISCORD_WEBHOOK_URL environment variable");
    println!("   📧 Email: Configure SMTP settings");
    println!("   📊 Metrics: Export to Prometheus/Grafana");

    println!("\n" + "=".repeat(50).as_str() + "\n");
    Ok(())
}

/// Create a pattern of transactions that should trigger rug pull detection
fn create_suspicious_transaction_pattern() -> Vec<Transaction> {
    let base_time = Utc::now() - Duration::hours(2);
    let token_mint = "SuspiciousToken123456789".to_string();

    // Create coordinated purchases - same time window, similar amounts
    vec![
        Transaction {
            signature: "coord_tx_1".to_string(),
            wallet: "wallet_coordinated_1".to_string(),
            token_mint: token_mint.clone(),
            amount_sol: 100.0,
            transaction_type: TransactionType::Buy,
            timestamp: base_time,
        },
        Transaction {
            signature: "coord_tx_2".to_string(),
            wallet: "wallet_coordinated_2".to_string(),
            token_mint: token_mint.clone(),
            amount_sol: 105.0, // Very similar amount
            transaction_type: TransactionType::Buy,
            timestamp: base_time + Duration::minutes(2), // Close timing
        },
        Transaction {
            signature: "coord_tx_3".to_string(),
            wallet: "wallet_coordinated_3".to_string(),
            token_mint: token_mint.clone(),
            amount_sol: 98.0, // Very similar amount
            transaction_type: TransactionType::Buy,
            timestamp: base_time + Duration::minutes(5), // Close timing
        },
        Transaction {
            signature: "coord_tx_4".to_string(),
            wallet: "wallet_coordinated_4".to_string(),
            token_mint: token_mint.clone(),
            amount_sol: 102.0, // Very similar amount
            transaction_type: TransactionType::Buy,
            timestamp: base_time + Duration::minutes(8), // Close timing
        },
        Transaction {
            signature: "coord_tx_5".to_string(),
            wallet: "wallet_coordinated_5".to_string(),
            token_mint: token_mint.clone(),
            amount_sol: 99.5, // Very similar amount
            transaction_type: TransactionType::Buy,
            timestamp: base_time + Duration::minutes(10), // Close timing
        },
    ]
}

/// Create a pattern of normal, non-coordinated transactions
fn create_normal_transaction_pattern() -> Vec<Transaction> {
    let base_time = Utc::now() - Duration::hours(1);
    let token_mint = "NormalToken987654321".to_string();

    // Create normal purchases - spread out timing, varied amounts
    vec![
        Transaction {
            signature: "normal_tx_1".to_string(),
            wallet: "wallet_normal_1".to_string(),
            token_mint: token_mint.clone(),
            amount_sol: 50.0,
            transaction_type: TransactionType::Buy,
            timestamp: base_time,
        },
        Transaction {
            signature: "normal_tx_2".to_string(),
            wallet: "wallet_normal_2".to_string(),
            token_mint: token_mint.clone(),
            amount_sol: 250.0, // Very different amount
            transaction_type: TransactionType::Buy,
            timestamp: base_time + Duration::minutes(45), // Spread out timing
        },
        Transaction {
            signature: "normal_tx_3".to_string(),
            wallet: "wallet_normal_3".to_string(),
            token_mint: token_mint.clone(),
            amount_sol: 15.0, // Very different amount
            transaction_type: TransactionType::Buy,
            timestamp: base_time + Duration::hours(2), // Much later
        },
    ]
}

/// Display a summary of a rug pull alert
fn display_alert_summary(alert: &RugPullAlert) {
    let risk_emoji = match alert.overall_risk_score {
        score if score >= 0.9 => "🚨",
        score if score >= 0.7 => "⚠️",
        score if score >= 0.5 => "⚡",
        _ => "ℹ️",
    };

    println!("  {} Alert ID: {}", risk_emoji, alert.alert_id);
    println!("    Token: {}", alert.token_mint);
    println!("    Risk Score: {:.1}%", alert.overall_risk_score * 100.0);
    println!("    Confidence: {:.1}%", alert.confidence * 100.0);
    println!("    Clusters: {}", alert.clusters.len());
    
    let total_wallets: usize = alert.clusters.iter().map(|c| c.wallets.len()).sum();
    let total_investment: f64 = alert.clusters.iter().map(|c| c.total_investment).sum();
    
    println!("    Total Wallets: {}", total_wallets);
    println!("    Total Investment: {:.2} SOL", total_investment);
    
    // Show risk level
    let max_risk = alert.clusters.iter()
        .map(|c| &c.risk_level)
        .max()
        .unwrap_or(&RiskLevel::Low);
    println!("    Max Risk Level: {:?}", max_risk);
    
    // Show top recommendations
    if !alert.recommended_actions.is_empty() {
        println!("    Top Recommendation: {}", alert.recommended_actions[0]);
    }
    println!();
}