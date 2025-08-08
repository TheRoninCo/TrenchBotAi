//! Quick test to verify basic compilation without heavy dependencies

use std::env;
use anyhow::Result;

// Test basic types from our analytics module
use trenchbot_dex::analytics::{Transaction, TransactionType};
use trenchbot_dex::{FireMode, TrenchConfig};

fn main() -> Result<()> {
    println!("🚀 TrenchBot Quick Test");

    // Test basic config loading
    let mut config = TrenchConfig::default();
    config.fire_control = FireMode::Cold;
    
    println!("✅ Config created: {:?}", config.fire_control);

    // Test basic transaction creation
    let test_tx = Transaction {
        signature: "test123".to_string(),
        wallet: "test_wallet".to_string(),
        token_mint: "test_token".to_string(),
        amount_sol: 10.0,
        transaction_type: TransactionType::Buy,
        timestamp: chrono::Utc::now(),
    };
    
    println!("✅ Transaction created: {} SOL", test_tx.amount_sol);

    // Test environment variable access
    if let Ok(key) = env::var("HELIUS_API_KEY") {
        println!("✅ Helius API key found: {}...{}", &key[..8], &key[key.len()-8..]);
    } else {
        println!("⚠️  No Helius API key found in environment");
    }

    println!("🎯 All basic tests passed!");
    Ok(())
}