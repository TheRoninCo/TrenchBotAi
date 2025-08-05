use serde::{Deserialize, Serialize};
use solana_sdk::pubkey::Pubkey;
use chrono::{DateTime, Utc};
use std::collections::HashMap;

// Transaction-related types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingTransaction {
    pub signature: String,
    pub program_id: Pubkey,
    pub token_mint: Pubkey,
    pub amount: f64,
    pub slippage: f64,
    pub gas_price: u64,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeResult {
    pub success: bool,
    pub profit: Option<f64>,
    pub gas_used: u64,
    pub execution_time_ms: u64,
}

// Market data types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketSnapshot {
    pub slot: u64,
    pub timestamp: DateTime<Utc>,
    pub tokens: HashMap<Pubkey, TokenData>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenData {
    pub price: f64,
    pub liquidity: f64,
    pub price_impact: f64,
    pub volume_24h: f64,
    pub social_metrics: SocialMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SocialMetrics {
    pub twitter_sentiment: f64,
    pub reddit_sentiment: f64,
    pub telegram_activity: f64,
}

// Opportunity types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandwichOpportunity {
    pub id: String,
    pub target_tx: String,
    pub frontrun_amount: f64,
    pub backrun_amount: f64,
    pub expected_profit: f64,
    pub confidence: f64,
    pub token_mint: Pubkey,
    pub created_at: DateTime<Utc>,
    pub status: OpportunityStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OpportunityStatus {
    Pending,
    Executing,
    Completed,
    Failed,
}

// AI/ML types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingData {
    pub features: Vec<f64>,
    pub label: f64,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlphaWalletSignal {
    pub wallet_address: Pubkey,
    pub signal_strength: f64,
    pub historical_accuracy: f64,
    pub current_positions: Vec<Pubkey>,
    pub estimated_next_move: Option<String>,
}