use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use anyhow::Result;
use solana_sdk::pubkey::Pubkey;

#[derive(Debug)]
pub struct RiskManager {
    circuit_breaker: Arc<AtomicBool>,
    max_position_size: f64,
    max_daily_loss: f64,
    blacklisted_tokens: Vec<Pubkey>,
}

impl RiskManager {
    pub fn new(max_position_size: f64, max_daily_loss: f64, blacklisted_tokens: Vec<Pubkey>) -> Self {
        Self {
            circuit_breaker: Arc::new(AtomicBool::new(false)),
            max_position_size,
            max_daily_loss,
            blacklisted_tokens,
        }
    }

    pub fn validate_trade(
        &self,
        opportunity: &SandwichOpportunity,
        current_exposure: f64,
    ) -> Result<()> {
        if self.circuit_breaker.load(Ordering::Relaxed) {
            anyhow::bail!("Trading halted by circuit breaker");
        }

        if opportunity.expected_profit <= 0.0 {
            anyhow::bail!("Negative profit opportunity");
        }

        if self.blacklisted_tokens.contains(&opportunity.token_mint) {
            anyhow::bail!("Token is blacklisted");
        }

        if opportunity.frontrun_amount > self.max_position_size {
            anyhow::bail!("Position size exceeds limit");
        }

        if current_exposure + opportunity.frontrun_amount > self.max_position_size * 3.0 {
            anyhow::bail!("Would exceed total exposure limit");
        }

        Ok(())
    }

    pub fn trigger_circuit_breaker(&self) {
        self.circuit_breaker.store(true, Ordering::Relaxed);
    }

    pub fn reset_circuit_breaker(&self) {
        self.circuit_breaker.store(false, Ordering::Relaxed);
    }
}use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use solana_sdk::pubkey::Pubkey;

#[derive(Debug)]
pub struct RiskManager {
    circuit_breaker: Arc<AtomicBool>,
    max_position_size: f64,
    blacklisted_tokens: Vec<Pubkey>,
}

impl RiskManager {
    pub fn new(max_position_size: f64, blacklisted_tokens: Vec<Pubkey>) -> Self {
        Self {
            circuit_breaker: Arc::new(AtomicBool::new(false)),
            max_position_size,
            blacklisted_tokens,
        }
    }

    pub fn validate_trade(&self, amount: f64, token: &Pubkey) -> bool {
        !self.circuit_breaker.load(Ordering::Relaxed) &&
        !self.blacklisted_tokens.contains(token) &&
        amount <= self.max_position_size
    }

    pub fn trigger_circuit_breaker(&self) {
        self.circuit_breaker.store(true, Ordering::Relaxed);
    }
}