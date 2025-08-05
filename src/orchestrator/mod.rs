//! Trading orchestration and coordination
use crate::types::{SandwichOpportunity, TradeResult};
use crate::engines::mev::sandwich::SandwichAttackEngine;
use crate::core::config::Config;
use anyhow::Result;

pub struct TradingOrchestrator {
    config: Config,
    sandwich_engine: SandwichAttackEngine,
}

impl TradingOrchestrator {
    pub async fn new(config: Config) -> Result<Self> {
        let sandwich_engine = SandwichAttackEngine::new(config.sandwich.clone())?;
        
        Ok(Self {
            config,
            sandwich_engine,
        })
    }
    
    pub async fn run_trading_loop(&mut self) -> Result<()> {
        println!("🎯 Starting trading orchestrator...");
        
        loop {
            // Scan for sandwich opportunities
            let opportunities = self.sandwich_engine.scan_opportunities().await?;
            
            if !opportunities.is_empty() {
                println!("🔍 Found {} sandwich opportunities", opportunities.len());
                
                for opportunity in opportunities {
                    if self.should_execute(&opportunity).await? {
                        match self.sandwich_engine.execute_sandwich(&opportunity).await {
                            Ok(result) => {
                                println!("✅ Sandwich executed: {:?}", result);
                            }
                            Err(e) => {
                                println!("❌ Sandwich failed: {}", e);
                            }
                        }
                    }
                }
            }
            
            // Small delay between iterations
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }
    }
    
    async fn should_execute(&self, opportunity: &SandwichOpportunity) -> Result<bool> {
        // Risk assessment and execution decision
        Ok(opportunity.pattern.estimated_profit > self.config.sandwich.min_profit_threshold)
    }
}
