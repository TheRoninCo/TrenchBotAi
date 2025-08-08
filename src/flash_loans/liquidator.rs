//! Liquidation Engine for Flash Loan MEV
//! Finds and executes liquidations across protocols

use super::mango::LiquidatablePosition;
use anyhow::{Result, anyhow};
use solana_sdk::pubkey::Pubkey;
use std::collections::HashMap;
use tracing::{info, warn};

/// Liquidation opportunity found across protocols
#[derive(Debug, Clone)]
pub struct LiquidationOpportunity {
    pub position: Pubkey,
    pub protocol: String,
    pub collateral_token: Pubkey,
    pub debt_token: Pubkey,
    pub collateral_amount: u64,
    pub debt_amount: u64,
    pub health_factor: f32,
    pub expected_profit: u64,
    pub liquidation_bonus: f32,
    pub gas_cost_estimate: u64,
    pub time_sensitivity: TimeSensitivity,
}

/// How urgently a liquidation needs to be executed
#[derive(Debug, Clone)]
pub enum TimeSensitivity {
    Critical,  // Health < 0.9, liquidate immediately
    High,      // Health 0.9-0.95, liquidate within minutes
    Medium,    // Health 0.95-0.98, liquidate within hour
    Low,       // Health 0.98-1.0, monitor for changes
}

/// Multi-protocol liquidation engine
pub struct LiquidationEngine {
    protocols: Vec<String>,
    health_thresholds: HashMap<String, f32>,
    min_profit_threshold: u64,
    max_gas_cost: u64,
}

impl LiquidationEngine {
    pub async fn new() -> Result<Self> {
        let mut health_thresholds = HashMap::new();
        health_thresholds.insert("solend".to_string(), 1.0);   // 100% health = liquidatable
        health_thresholds.insert("mango".to_string(), 1.0);    // 100% health = liquidatable  
        health_thresholds.insert("marginfi".to_string(), 1.05); // 105% health = liquidatable
        
        Ok(Self {
            protocols: vec![
                "solend".to_string(),
                "mango".to_string(),
                "marginfi".to_string(),
            ],
            health_thresholds,
            min_profit_threshold: 50_000, // 0.05 SOL minimum profit
            max_gas_cost: 20_000, // 0.02 SOL max gas
        })
    }
    
    /// Scan all protocols for liquidation opportunities
    pub async fn scan_all_liquidations(&self) -> Result<Vec<LiquidationOpportunity>> {
        let mut all_opportunities = Vec::new();
        
        info!("🔍 Scanning {} protocols for liquidations...", self.protocols.len());
        
        for protocol in &self.protocols {
            match self.scan_protocol_liquidations(protocol).await {
                Ok(mut opportunities) => {
                    info!("Found {} liquidations in {}", opportunities.len(), protocol);
                    all_opportunities.append(&mut opportunities);
                },
                Err(e) => {
                    warn!("Failed to scan {} liquidations: {}", protocol, e);
                },
            }
        }
        
        // Sort by profitability and time sensitivity
        all_opportunities.sort_by(|a, b| {
            // First by time sensitivity (critical first)
            let time_priority_a = match a.time_sensitivity {
                TimeSensitivity::Critical => 4,
                TimeSensitivity::High => 3,
                TimeSensitivity::Medium => 2,
                TimeSensitivity::Low => 1,
            };
            let time_priority_b = match b.time_sensitivity {
                TimeSensitivity::Critical => 4,
                TimeSensitivity::High => 3,
                TimeSensitivity::Medium => 2,
                TimeSensitivity::Low => 1,
            };
            
            // Then by profit (higher first)
            time_priority_b.cmp(&time_priority_a)
                .then_with(|| b.expected_profit.cmp(&a.expected_profit))
        });
        
        info!("🎯 Total liquidation opportunities: {}", all_opportunities.len());
        Ok(all_opportunities)
    }
    
    /// Scan specific protocol for liquidations
    async fn scan_protocol_liquidations(&self, protocol: &str) -> Result<Vec<LiquidationOpportunity>> {
        match protocol {
            "solend" => self.scan_solend_liquidations().await,
            "mango" => self.scan_mango_liquidations().await,
            "marginfi" => self.scan_marginfi_liquidations().await,
            _ => Err(anyhow!("Unsupported protocol: {}", protocol)),
        }
    }
    
    /// Scan Solend for liquidatable positions
    pub async fn scan_solend_liquidations(&self) -> Result<Vec<LiquidationOpportunity>> {
        info!("🔍 Scanning Solend liquidations...");
        
        // In a real implementation, this would:
        // 1. Query all Solend obligations
        // 2. Calculate health factors
        // 3. Find positions with health < threshold
        // 4. Calculate liquidation rewards
        
        // Mock data for demonstration
        Ok(vec![
            LiquidationOpportunity {
                position: Pubkey::new_unique(),
                protocol: "solend".to_string(),
                collateral_token: solana_sdk::native_token::NATIVE_MINT, // SOL
                debt_token: Pubkey::new_unique(), // USDC
                collateral_amount: 10_000_000_000, // 10 SOL
                debt_amount: 950_000_000, // 950 USDC (assuming SOL = $100)
                health_factor: 0.92, // Underwater
                expected_profit: 500_000_000, // 0.5 SOL profit (5% bonus)
                liquidation_bonus: 0.05, // 5% liquidation bonus
                gas_cost_estimate: 15_000, // 0.015 SOL gas
                time_sensitivity: TimeSensitivity::High,
            }
        ])
    }
    
    /// Scan Mango for liquidatable positions
    pub async fn scan_mango_liquidations(&self) -> Result<Vec<LiquidationOpportunity>> {
        info!("🔍 Scanning Mango liquidations...");
        
        // Mock high-profit Mango liquidation
        Ok(vec![
            LiquidationOpportunity {
                position: Pubkey::new_unique(),
                protocol: "mango".to_string(),
                collateral_token: Pubkey::new_unique(), // MSOL
                debt_token: solana_sdk::native_token::NATIVE_MINT, // SOL
                collateral_amount: 12_000_000_000, // 12 mSOL
                debt_amount: 11_000_000_000, // 11 SOL
                health_factor: 0.85, // Heavily underwater
                expected_profit: 800_000_000, // 0.8 SOL profit  
                liquidation_bonus: 0.08, // 8% bonus
                gas_cost_estimate: 18_000, // 0.018 SOL gas
                time_sensitivity: TimeSensitivity::Critical,
            }
        ])
    }
    
    /// Scan MarginFi for liquidatable positions
    async fn scan_marginfi_liquidations(&self) -> Result<Vec<LiquidationOpportunity>> {
        info!("🔍 Scanning MarginFi liquidations...");
        
        // Mock MarginFi liquidation
        Ok(vec![
            LiquidationOpportunity {
                position: Pubkey::new_unique(),
                protocol: "marginfi".to_string(),
                collateral_token: Pubkey::new_unique(), // BONK
                debt_token: Pubkey::new_unique(), // USDC  
                collateral_amount: 1_000_000_000_000, // 1T BONK tokens
                debt_amount: 200_000_000, // 200 USDC
                health_factor: 0.98, // Barely liquidatable
                expected_profit: 60_000_000, // 0.06 SOL profit
                liquidation_bonus: 0.03, // 3% bonus
                gas_cost_estimate: 12_000, // 0.012 SOL gas
                time_sensitivity: TimeSensitivity::Low,
            }
        ])
    }
    
    /// Filter opportunities by profitability
    pub fn filter_profitable(&self, opportunities: Vec<LiquidationOpportunity>) -> Vec<LiquidationOpportunity> {
        opportunities
            .into_iter()
            .filter(|opp| {
                let net_profit = opp.expected_profit.saturating_sub(opp.gas_cost_estimate);
                net_profit >= self.min_profit_threshold && opp.gas_cost_estimate <= self.max_gas_cost
            })
            .collect()
    }
    
    /// Get the most profitable liquidation opportunity
    pub async fn get_best_opportunity(&self) -> Result<Option<LiquidationOpportunity>> {
        let all_opportunities = self.scan_all_liquidations().await?;
        let profitable = self.filter_profitable(all_opportunities);
        
        Ok(profitable.into_iter().max_by_key(|opp| {
            // Score by net profit and urgency
            let net_profit = opp.expected_profit.saturating_sub(opp.gas_cost_estimate);
            let urgency_multiplier = match opp.time_sensitivity {
                TimeSensitivity::Critical => 5,
                TimeSensitivity::High => 3,
                TimeSensitivity::Medium => 2, 
                TimeSensitivity::Low => 1,
            };
            net_profit * urgency_multiplier
        }))
    }
    
    /// Calculate expected profit after all costs
    pub fn calculate_net_profit(&self, opportunity: &LiquidationOpportunity, flash_loan_fee: u64) -> i64 {
        let gross_profit = opportunity.expected_profit as i64;
        let total_costs = (opportunity.gas_cost_estimate + flash_loan_fee) as i64;
        gross_profit - total_costs
    }
    
    /// Check if position is still liquidatable (health factor may have changed)
    pub async fn verify_liquidation_opportunity(&self, opportunity: &LiquidationOpportunity) -> Result<bool> {
        // In reality, this would re-check the position's health factor
        // For now, simulate some positions becoming unliquidatable over time
        let still_liquidatable = opportunity.health_factor < 0.95;
        
        if !still_liquidatable {
            warn!("⚠️ Position {} no longer liquidatable (health improved)", opportunity.position);
        }
        
        Ok(still_liquidatable)
    }
}