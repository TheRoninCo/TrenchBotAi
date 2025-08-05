//! Market data validation and cross-verification
use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketDataPoint {
    pub price: f64,
    pub volume: f64,
    pub timestamp: DateTime<Utc>,
    pub source: DataSource,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataSource {
    Jupiter,
    Raydium,
    Orca,
    Serum,
    Helius,
}

pub struct DataValidator {
    price_sources: HashMap<String, Vec<MarketDataPoint>>,
    anomaly_threshold: f64,
    min_sources_required: usize,
}

impl DataValidator {
    pub fn new() -> Self {
        Self {
            price_sources: HashMap::new(),
            anomaly_threshold: 0.05, // 5% price deviation threshold
            min_sources_required: 2,
        }
    }
    
    pub fn add_price_data(&mut self, token_mint: &str, data: MarketDataPoint) {
        self.price_sources
            .entry(token_mint.to_string())
            .or_insert_with(Vec::new)
            .push(data);
        
        // Keep only recent data (last 5 minutes)
        let cutoff = Utc::now() - chrono::Duration::minutes(5);
        if let Some(prices) = self.price_sources.get_mut(token_mint) {
            prices.retain(|p| p.timestamp > cutoff);
        }
    }
    
    pub fn validate_price(&self, token_mint: &str, claimed_price: f64) -> Result<ValidationResult> {
        let prices = self.price_sources.get(token_mint)
            .ok_or_else(|| anyhow::anyhow!("No price data for token {}", token_mint))?;
        
        if prices.len() < self.min_sources_required {
            return Ok(ValidationResult {
                is_valid: false,
                confidence: 0.0,
                reason: "Insufficient price sources".to_string(),
                consensus_price: None,
            });
        }
        
        // Calculate consensus price
        let recent_prices: Vec<f64> = prices.iter()
            .filter(|p| (Utc::now() - p.timestamp).num_seconds() < 30) // Last 30 seconds
            .map(|p| p.price)
            .collect();
        
        if recent_prices.is_empty() {
            return Ok(ValidationResult {
                is_valid: false,
                confidence: 0.0,
                reason: "No recent price data".to_string(),
                consensus_price: None,
            });
        }
        
        let consensus_price = self.calculate_weighted_average(&recent_prices);
        let price_deviation = (claimed_price - consensus_price).abs() / consensus_price;
        
        let is_valid = price_deviation <= self.anomaly_threshold;
        let confidence = if is_valid {
            1.0 - (price_deviation / self.anomaly_threshold)
        } else {
            0.0
        };
        
        Ok(ValidationResult {
            is_valid,
            confidence,
            reason: if is_valid {
                "Price within acceptable range".to_string()
            } else {
                format!("Price deviation {:.2}% exceeds threshold", price_deviation * 100.0)
            },
            consensus_price: Some(consensus_price),
        })
    }
    
    fn calculate_weighted_average(&self, prices: &[f64]) -> f64 {
        if prices.is_empty() {
            return 0.0;
        }
        
        // Simple average for now, could weight by source reliability
        prices.iter().sum::<f64>() / prices.len() as f64
    }
    
    pub fn detect_market_manipulation(&self, token_mint: &str) -> Result<ManipulationAlert> {
        let prices = self.price_sources.get(token_mint)
            .ok_or_else(|| anyhow::anyhow!("No price data for token {}", token_mint))?;
        
        // Check for sudden price spikes
        let recent_prices: Vec<f64> = prices.iter()
            .filter(|p| (Utc::now() - p.timestamp).num_minutes() < 10)
            .map(|p| p.price)
            .collect();
        
        if recent_prices.len() < 3 {
            return Ok(ManipulationAlert {
                risk_level: RiskLevel::Low,
                message: "Insufficient data for manipulation detection".to_string(),
            });
        }
        
        let min_price = recent_prices.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_price = recent_prices.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let price_range_ratio = (max_price - min_price) / min_price;
        
        let (risk_level, message) = if price_range_ratio > 0.50 {
            (RiskLevel::High, "Extreme price volatility detected - possible manipulation".to_string())
        } else if price_range_ratio > 0.20 {
            (RiskLevel::Medium, "High price volatility detected".to_string())
        } else {
            (RiskLevel::Low, "Normal price behavior".to_string())
        };
        
        Ok(ManipulationAlert { risk_level, message })
    }
}

#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub confidence: f64,
    pub reason: String,
    pub consensus_price: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct ManipulationAlert {
    pub risk_level: RiskLevel,
    pub message: String,
}

#[derive(Debug, Clone)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
}
