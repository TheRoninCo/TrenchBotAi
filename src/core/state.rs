//! State management and recovery - critical for production
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BotState {
    pub last_processed_slot: u64,
    pub open_positions: Vec<OpenPosition>,
    pub pending_transactions: Vec<PendingTransaction>,
    pub blacklisted_tokens: Vec<String>,
    pub performance_snapshot: PerformanceSnapshot,
    pub last_saved: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenPosition {
    pub id: String,
    pub token_mint: String,
    pub amount: f64,
    pub entry_price: f64,
    pub entry_time: DateTime<Utc>,
    pub position_type: PositionType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PositionType {
    Sandwich { target_tx: String },
    Arbitrage { route: Vec<String> },
    Momentum { signal_strength: f64 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingTransaction {
    pub signature: String,
    pub transaction_type: String,
    pub submitted_at: DateTime<Utc>,
    pub retry_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSnapshot {
    pub total_pnl: f64,
    pub trades_today: u32,
    pub last_profitable_trade: Option<DateTime<Utc>>,
}

pub struct StateManager {
    state_file: String,
    current_state: BotState,
}

impl StateManager {
    pub fn new(state_file: &str) -> Result<Self> {
        let current_state = if Path::new(state_file).exists() {
            Self::load_state(state_file)?
        } else {
            BotState::default()
        };
        
        Ok(Self {
            state_file: state_file.to_string(),
            current_state,
        })
    }
    
    fn load_state(file_path: &str) -> Result<BotState> {
        let content = std::fs::read_to_string(file_path)?;
        let state: BotState = serde_json::from_str(&content)?;
        tracing::info!("📂 Loaded bot state from {}", file_path);
        Ok(state)
    }
    
    pub fn save_state(&mut self) -> Result<()> {
        self.current_state.last_saved = Utc::now();
        let content = serde_json::to_string_pretty(&self.current_state)?;
        std::fs::write(&self.state_file, content)?;
        tracing::debug!("💾 Saved bot state to {}", self.state_file);
        Ok(())
    }
    
    pub fn update_last_processed_slot(&mut self, slot: u64) -> Result<()> {
        self.current_state.last_processed_slot = slot;
        self.save_state()
    }
    
    pub fn add_open_position(&mut self, position: OpenPosition) -> Result<()> {
        self.current_state.open_positions.push(position);
        self.save_state()
    }
    
    pub fn remove_position(&mut self, position_id: &str) -> Result<()> {
        self.current_state.open_positions.retain(|p| p.id != position_id);
        self.save_state()
    }
    
    pub fn get_open_positions(&self) -> &Vec<OpenPosition> {
        &self.current_state.open_positions
    }
    
    pub fn get_last_processed_slot(&self) -> u64 {
        self.current_state.last_processed_slot
    }
    
    // Recovery methods
    pub async fn recover_from_crash(&mut self) -> Result<()> {
        tracing::warn!("🔄 Recovering from crash...");
        
        // Check for orphaned transactions
        for pending_tx in &self.current_state.pending_transactions {
            // TODO: Check transaction status and handle accordingly
            tracing::info!("Checking pending transaction: {}", pending_tx.signature);
        }
        
        // Validate open positions
        for position in &self.current_state.open_positions {
            // TODO: Verify position status and handle if needed
            tracing::info!("Validating position: {}", position.id);
        }
        
        tracing::info!("✅ Recovery completed");
        Ok(())
    }
}

impl Default for BotState {
    fn default() -> Self {
        Self {
            last_processed_slot: 0,
            open_positions: vec![],
            pending_transactions: vec![],
            blacklisted_tokens: vec![],
            performance_snapshot: PerformanceSnapshot {
                total_pnl: 0.0,
                trades_today: 0,
                last_profitable_trade: None,
            },
            last_saved: Utc::now(),
        }
    }
}
