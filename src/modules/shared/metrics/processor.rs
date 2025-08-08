use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MevMetrics {
    pub total_transactions: u64,
    pub successful_extractions: u64,
    pub total_profit_lamports: u64,
    pub average_response_time_ms: f64,
    pub by_strategy: HashMap<String, StrategyMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyMetrics {
    pub attempts: u64,
    pub successes: u64,
    pub profit_lamports: u64,
}

impl Default for MevMetrics {
    fn default() -> Self {
        Self {
            total_transactions: 0,
            successful_extractions: 0,
            total_profit_lamports: 0,
            average_response_time_ms: 0.0,
            by_strategy: HashMap::new(),
        }
    }
}

pub struct MevProcessor;

impl MevProcessor {
    pub async fn should_submit(
        &self,
        bundle: &Bundle,
        whale: Option<&Whale>,
        token: &str,  // Primary token mint in bundle
    ) -> anyhow::Result<bool> {
        let imbalance = self.pool_analyzer.get_imbalance(token).await?;
        let personality = whale.map(|w| w.personality.as_score()).unwrap_or(0.0);
        let velocity = whale.map(|w| w.velocity).unwrap_or(0.0);

        let profit = self.token_predictor.predict(
            token,
            personality,
            velocity,
            imbalance,
            bundle.tip_lamports as f32,
            bundle.cu_limit as f32,
        )?;

        Ok(profit > self.config.min_profit)
    }
}