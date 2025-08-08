//! Counter Rug Pull Strategy
//! 
//! This module implements sophisticated trading strategies to profit from detected
//! coordinated rug pull attempts. We identify the patterns early and execute
//! tactical extraction operations before the main dump occurs.
//!
//! WARNING: This is high-risk, high-reward warfare. Use appropriate position sizing.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc, Duration};
use std::collections::{HashMap, VecDeque};
use tokio::sync::{broadcast, mpsc};

use crate::analytics::{RugPullAlert, EarlyInvestorCluster, RiskLevel, RugPullEvent};
use crate::types::{PendingTransaction, TradeResult};
use crate::modules::shared::training::combat_logger::{self, CombatContext};
use crate::war::runescape_rankings::{RuneScapeRankingSystem, MonsterKill};
use crate::war::position_management::{PositionManager, RiskConfig};
use crate::war::circuit_breakers::{EmergencyManager, CircuitConfig};
use crate::war::ultra_low_latency::{UltraLowLatencyEngine, Trade as UltraLowLatencyTrade};

/// Counter-rug-pull strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterRugConfig {
    /// Maximum position size per token (in SOL)
    pub max_position_size: f64,
    /// Minimum risk score to trigger engagement
    pub min_risk_threshold: f64,
    /// How long to hold position before forced exit (minutes)
    pub max_hold_duration_minutes: i64,
    /// Profit target (percentage)
    pub profit_target: f64,
    /// Stop loss (percentage)
    pub stop_loss: f64,
    /// Maximum concurrent positions
    pub max_concurrent_positions: usize,
    /// Reserve capital percentage (keep as safety margin)
    pub reserve_capital_pct: f64,
}

impl Default for CounterRugConfig {
    fn default() -> Self {
        Self {
            max_position_size: 50.0,     // 50 SOL max per token
            min_risk_threshold: 0.75,    // 75% confidence required
            max_hold_duration_minutes: 30, // 30 minute max hold
            profit_target: 0.15,         // 15% profit target
            stop_loss: 0.08,             // 8% stop loss
            max_concurrent_positions: 3, // Max 3 simultaneous wars
            reserve_capital_pct: 0.3,    // Keep 30% in reserve
        }
    }
}

/// Trading phases in the counter-rug-pull operation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TradingPhase {
    Reconnaissance,  // Monitoring for patterns
    Infiltration,    // Early entry during accumulation
    Extraction,      // Exit before the dump
    Retreat,         // Emergency exit
    PostMortem,      // Analysis after operation
}

/// Individual counter-rug-pull operation
#[derive(Debug, Clone, Serialize)]
pub struct CounterRugOperation {
    pub operation_id: String,
    pub token_mint: String,
    pub phase: TradingPhase,
    pub entry_time: DateTime<Utc>,
    pub entry_price: f64,
    pub position_size: f64,
    pub current_price: f64,
    pub unrealized_pnl: f64,
    pub target_exit_time: DateTime<Utc>,
    pub risk_score: f64,
    pub cluster_info: ClusterIntel,
    pub exit_triggers: Vec<ExitTrigger>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ClusterIntel {
    pub cluster_count: usize,
    pub total_wallets: usize,
    pub coordination_score: f64,
    pub estimated_total_supply: f64,
    pub cluster_accumulation_rate: f64,
    pub time_to_estimated_dump: Option<Duration>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub enum ExitTrigger {
    ProfitTarget,
    StopLoss,
    TimeLimit,
    ClusterBehaviorChange,
    RugPullExecuted,
    EmergencyExit,
}

/// Market timing intelligence for optimal entry/exit
#[derive(Debug, Clone)]
pub struct MarketTiming {
    pub liquidity_depth: f64,
    pub volume_spike_detected: bool,
    pub whale_movement_detected: bool,
    pub estimated_dump_window: Option<DateTime<Utc>>,
    pub market_impact_score: f64,
}

pub struct CounterRugPullStrategy {
    config: CounterRugConfig,
    active_operations: HashMap<String, CounterRugOperation>,
    operation_history: VecDeque<CounterRugOperation>,
    available_capital: f64,
    reserved_capital: f64,
    context: CombatContext,
    alert_receiver: broadcast::Receiver<RugPullEvent>,
    trade_executor: mpsc::UnboundedSender<TradeCommand>,
    scammer_scammed_count: u64, // Track our victories!
    total_scammer_profit_extracted: f64, // Total profit from scammers
    runescape_ranking: RuneScapeRankingSystem, // RuneScape-style progression
    position_manager: PositionManager, // Advanced position management
    emergency_manager: EmergencyManager, // Circuit breakers and safety systems
    ultra_low_latency_engine: UltraLowLatencyEngine, // Sub-microsecond trading engine
}

#[derive(Debug, Clone)]
pub struct TradeCommand {
    pub command_type: TradeCommandType,
    pub token_mint: String,
    pub amount: f64,
    pub max_slippage: f64,
    pub operation_id: String,
}

#[derive(Debug, Clone)]
pub enum TradeCommandType {
    Buy,
    Sell,
    EmergencyExit,
}

impl CounterRugPullStrategy {
    pub fn new(
        config: CounterRugConfig,
        initial_capital: f64,
        alert_receiver: broadcast::Receiver<RugPullEvent>,
        trade_executor: mpsc::UnboundedSender<TradeCommand>,
    ) -> Self {
        let reserved_capital = initial_capital * config.reserve_capital_pct;
        let available_capital = initial_capital - reserved_capital;

        // Initialize safety systems
        let risk_config = RiskConfig::default();
        let circuit_config = CircuitConfig::default();
        let (emergency_manager, _emergency_receiver) = EmergencyManager::new(circuit_config);

        Self {
            config,
            active_operations: HashMap::new(),
            operation_history: VecDeque::with_capacity(100),
            available_capital,
            reserved_capital,
            context: CombatContext {
                operation_id: "counter_rug_warfare".to_string(),
                squad: "extraction_unit".to_string(),
                trace_id: uuid::Uuid::new_v4().to_string(),
            },
            alert_receiver,
            trade_executor,
            scammer_scammed_count: 0,
            total_scammer_profit_extracted: 0.0,
            runescape_ranking: RuneScapeRankingSystem::new("TrenchBot Warrior".to_string()),
            position_manager: PositionManager::new(risk_config, initial_capital),
            emergency_manager,
            ultra_low_latency_engine: UltraLowLatencyEngine::new(),
        }
    }

    /// Start the counter-rug-pull warfare engine
    pub async fn engage_warfare(&mut self) -> Result<()> {
        combat_logger::severity::info(
            "counter_rug_warfare",
            "combat_systems_online",
            serde_json::json!({
                "available_capital": self.available_capital,
                "max_concurrent_ops": self.config.max_concurrent_positions,
                "profit_target": self.config.profit_target * 100.0,
                "status": "weapons_hot"
            }),
            Some(self.context.clone())
        ).await?;

        // Main warfare loop
        loop {
            tokio::select! {
                // Listen for rug pull alerts
                alert_result = self.alert_receiver.recv() => {
                    match alert_result {
                        Ok(event) => {
                            if let Err(e) = self.process_intelligence(event).await {
                                combat_logger::severity::error(
                                    "counter_rug_warfare",
                                    "intelligence_processing_failure",
                                    serde_json::json!({ "error": e.to_string() }),
                                    Some(self.context.clone())
                                ).await?;
                            }
                        }
                        Err(_) => break, // Channel closed
                    }
                },
                
                // Monitor active operations
                _ = tokio::time::sleep(tokio::time::Duration::from_secs(10)) => {
                    if let Err(e) = self.monitor_active_operations().await {
                        combat_logger::severity::error(
                            "counter_rug_warfare",
                            "operation_monitoring_failure",
                            serde_json::json!({ "error": e.to_string() }),
                            Some(self.context.clone())
                        ).await?;
                    }
                }
            }
        }

        Ok(())
    }

    /// Process incoming rug pull intelligence
    async fn process_intelligence(&mut self, event: RugPullEvent) -> Result<()> {
        match event.event_type {
            crate::analytics::rug_pull_monitor::EventType::NewAlert | 
            crate::analytics::rug_pull_monitor::EventType::HighRiskDetected => {
                if let Some(alert) = event.alert {
                    self.evaluate_engagement_opportunity(alert).await?;
                }
            }
            crate::analytics::rug_pull_monitor::EventType::ClusterUpdate => {
                if let Some(cluster) = event.cluster_update {
                    self.update_operation_intelligence(&event.token_mint, cluster).await?;
                }
            }
            _ => {} // Other events don't require action
        }
        Ok(())
    }

    /// Evaluate whether to engage in counter-rug-pull operation
    async fn evaluate_engagement_opportunity(&mut self, alert: RugPullAlert) -> Result<()> {
        // First check emergency systems
        if !self.emergency_manager.can_proceed().await? {
            combat_logger::severity::warn(
                "counter_rug_warfare",
                "engagement_blocked_emergency",
                serde_json::json!({
                    "token": alert.token_mint,
                    "reason": "emergency_systems_active"
                }),
                Some(self.context.clone())
            ).await?;
            return Ok(());
        }

        // Check if we meet engagement criteria
        if !self.should_engage(&alert) {
            combat_logger::severity::debug(
                "counter_rug_warfare",
                "engagement_declined",
                serde_json::json!({
                    "token": alert.token_mint,
                    "risk_score": alert.overall_risk_score,
                    "reason": "criteria_not_met"
                }),
                Some(self.context.clone())
            ).await?;
            return Ok(());
        }

        // Calculate optimal position size using position manager
        let confidence = alert.confidence;
        let expected_return = 0.15; // Default 15% expected return
        let risk_score = alert.overall_risk_score;
        
        let optimal_position_size = self.position_manager.calculate_position_size(
            &alert.token_mint,
            confidence,
            expected_return,
            risk_score,
        ).await?;
        
        // Fallback to legacy calculation if position manager returns 0
        let position_size = if optimal_position_size > 0.0 {
            optimal_position_size
        } else {
            self.calculate_position_size(&alert)?
        };
        
        let market_timing = self.analyze_market_timing(&alert).await?;
        
        // Validate position before execution
        if position_size > 0.0 {
            // Create temporary operation for validation
            let temp_operation = CounterRugOperation {
                operation_id: format!("temp_validation_{}", Utc::now().timestamp()),
                token_mint: alert.token_mint.clone(),
                phase: TradingPhase::Reconnaissance,
                entry_time: Utc::now(),
                entry_price: 1.0, // Temporary
                position_size,
                current_price: 1.0,
                unrealized_pnl: 0.0,
                target_exit_time: Utc::now() + Duration::minutes(self.config.max_hold_duration_minutes),
                risk_score: alert.overall_risk_score,
                cluster_info: ClusterIntel {
                    cluster_count: alert.clusters.len(),
                    total_wallets: alert.clusters.iter().map(|c| c.wallets.len()).sum(),
                    coordination_score: alert.clusters.iter().map(|c| c.coordination_score).fold(0.0, f64::max),
                    estimated_total_supply: 1_000_000.0,
                    cluster_accumulation_rate: 0.0,
                    time_to_estimated_dump: market_timing.estimated_dump_window.map(|t| t - Utc::now()),
                },
                exit_triggers: vec![ExitTrigger::ProfitTarget],
            };

            let validation = self.position_manager.validate_position(&temp_operation, position_size).await?;
            
            if !validation.is_valid {
                combat_logger::severity::warn(
                    "counter_rug_warfare",
                    "position_validation_failed",
                    serde_json::json!({
                        "token": alert.token_mint,
                        "position_size": position_size,
                        "violations": validation.violations.len(),
                        "reason": "position_validation_failed"
                    }),
                    Some(self.context.clone())
                ).await?;
                return Ok(());
            }

            // Execute infiltration
            self.execute_infiltration(&alert, position_size, market_timing).await?;
        }
        
        Ok(())
    }

    fn should_engage(&self, alert: &RugPullAlert) -> bool {
        // Check risk threshold
        if alert.overall_risk_score < self.config.min_risk_threshold {
            return false;
        }

        // Check if we already have a position in this token
        if self.active_operations.contains_key(&alert.token_mint) {
            return false;
        }

        // Check maximum concurrent positions
        if self.active_operations.len() >= self.config.max_concurrent_positions {
            return false;
        }

        // Check available capital
        let min_position = self.config.max_position_size * 0.1; // Minimum 10% of max position
        if self.available_capital < min_position {
            return false;
        }

        // Check cluster characteristics for profitability
        let total_wallets: usize = alert.clusters.iter().map(|c| c.wallets.len()).sum();
        let total_investment: f64 = alert.clusters.iter().map(|c| c.total_investment).sum();
        
        // Need sufficient size for profitable extraction
        total_wallets >= 5 && total_investment >= 500.0 // At least 500 SOL coordinated
    }

    fn calculate_position_size(&self, alert: &RugPullAlert) -> Result<f64> {
        let base_size = self.config.max_position_size;
        
        // Adjust based on risk score (higher risk = smaller position)
        let risk_adjustment = 2.0 - alert.overall_risk_score; // 0.9 risk -> 1.1x, 0.7 risk -> 1.3x
        
        // Adjust based on cluster size (larger clusters = larger opportunity)
        let total_investment: f64 = alert.clusters.iter().map(|c| c.total_investment).sum();
        let size_multiplier = (total_investment / 1000.0).min(2.0); // Cap at 2x for 1000+ SOL
        
        let calculated_size = base_size * risk_adjustment * size_multiplier;
        
        // Ensure we don't exceed available capital
        Ok(calculated_size.min(self.available_capital * 0.8)) // Max 80% of available capital per trade
    }

    async fn analyze_market_timing(&self, alert: &RugPullAlert) -> Result<MarketTiming> {
        // Analyze cluster behavior to estimate timing
        let mut earliest_entry = Utc::now();
        let mut latest_entry = Utc::now();
        
        for cluster in &alert.clusters {
            if cluster.first_purchase_time < earliest_entry {
                earliest_entry = cluster.first_purchase_time;
            }
        }
        
        // Estimate accumulation phase duration
        let accumulation_duration = Utc::now() - earliest_entry;
        
        // Typical rug pull timeline: 
        // - Accumulation: 1-4 hours
        // - Price pump: 30 minutes - 2 hours  
        // - Dump: 5-15 minutes
        let estimated_dump_window = if accumulation_duration > Duration::hours(2) {
            Some(Utc::now() + Duration::minutes(30)) // Dump likely soon
        } else {
            Some(Utc::now() + Duration::hours(1)) // More time for accumulation
        };

        Ok(MarketTiming {
            liquidity_depth: 1000.0, // TODO: Get real liquidity data
            volume_spike_detected: false, // TODO: Implement volume analysis
            whale_movement_detected: true, // We detected the coordinated cluster
            estimated_dump_window,
            market_impact_score: 0.7, // TODO: Calculate based on our position vs liquidity
        })
    }

    async fn execute_infiltration(
        &mut self,
        alert: &RugPullAlert,
        position_size: f64,
        market_timing: MarketTiming,
    ) -> Result<()> {
        let operation_id = format!("op_{}_{}", 
            alert.token_mint[..8].to_string(), 
            Utc::now().timestamp()
        );

        // Create the operation
        let operation = CounterRugOperation {
            operation_id: operation_id.clone(),
            token_mint: alert.token_mint.clone(),
            phase: TradingPhase::Infiltration,
            entry_time: Utc::now(),
            entry_price: 0.0, // Will be updated when trade executes
            position_size,
            current_price: 0.0,
            unrealized_pnl: 0.0,
            target_exit_time: Utc::now() + Duration::minutes(self.config.max_hold_duration_minutes),
            risk_score: alert.overall_risk_score,
            cluster_info: ClusterIntel {
                cluster_count: alert.clusters.len(),
                total_wallets: alert.clusters.iter().map(|c| c.wallets.len()).sum(),
                coordination_score: alert.clusters.iter().map(|c| c.coordination_score).fold(0.0, f64::max),
                estimated_total_supply: 1_000_000.0, // TODO: Get real token supply
                cluster_accumulation_rate: 0.0, // TODO: Calculate accumulation rate
                time_to_estimated_dump: market_timing.estimated_dump_window.map(|t| t - Utc::now()),
            },
            exit_triggers: vec![
                ExitTrigger::ProfitTarget,
                ExitTrigger::StopLoss,
                ExitTrigger::TimeLimit,
            ],
        };

        // Log the engagement
        combat_logger::severity::info(
            "counter_rug_warfare",
            "infiltration_commenced",
            serde_json::json!({
                "operation_id": operation_id,
                "token": alert.token_mint,
                "position_size": position_size,
                "target_profit": self.config.profit_target * 100.0,
                "max_hold_minutes": self.config.max_hold_duration_minutes,
                "cluster_intel": operation.cluster_info
            }),
            Some(self.context.clone())
        ).await?;

        // Execute ultra-fast buy order using low-latency engine
        match self.execute_ultra_fast_buy(&alert.token_mint, position_size, 0.03).await {
            Ok(trades) => {
                if !trades.is_empty() {
                    // Update operation with actual execution data
                    if let Some(operation) = self.active_operations.get_mut(&alert.token_mint) {
                        operation.entry_price = trades[0].price;
                        operation.current_price = trades[0].price;
                    }
                    
                    combat_logger::severity::info(
                        "counter_rug_warfare",
                        "ultra_fast_execution_success",
                        serde_json::json!({
                            "operation_id": operation_id,
                            "trades_executed": trades.len(),
                            "entry_price": trades[0].price,
                            "total_size": trades.iter().map(|t| t.size).sum::<f64>(),
                            "execution_latency_ns": "sub_microsecond"
                        }),
                        Some(self.context.clone())
                    ).await?;
                } else {
                    // Fallback to traditional execution
                    let trade_command = TradeCommand {
                        command_type: TradeCommandType::Buy,
                        token_mint: alert.token_mint.clone(),
                        amount: position_size,
                        max_slippage: 0.03,
                        operation_id: operation_id.clone(),
                    };
                    self.trade_executor.send(trade_command)?;
                }
            },
            Err(e) => {
                // Fallback to traditional execution on error
                combat_logger::severity::warn(
                    "counter_rug_warfare",
                    "ultra_fast_execution_fallback",
                    serde_json::json!({
                        "operation_id": operation_id,
                        "error": e.to_string(),
                        "fallback": "traditional_execution"
                    }),
                    Some(self.context.clone())
                ).await?;
                
                let trade_command = TradeCommand {
                    command_type: TradeCommandType::Buy,
                    token_mint: alert.token_mint.clone(),
                    amount: position_size,
                    max_slippage: 0.03,
                    operation_id: operation_id.clone(),
                };
                self.trade_executor.send(trade_command)?;
            }
        }
        
        // Reserve the capital
        self.available_capital -= position_size;
        
        // Register position with position manager
        self.position_manager.open_position(&operation).await?;
        
        // Track the operation
        self.active_operations.insert(alert.token_mint.clone(), operation);

        Ok(())
    }

    /// Monitor active operations for exit signals
    async fn monitor_active_operations(&mut self) -> Result<()> {
        let mut operations_to_exit = Vec::new();

        for (token_mint, operation) in &mut self.active_operations {
            // TODO: Update current price from real market data
            // operation.current_price = get_current_price(token_mint).await?;
            
            // Calculate unrealized P&L
            if operation.entry_price > 0.0 {
                let price_change = (operation.current_price - operation.entry_price) / operation.entry_price;
                operation.unrealized_pnl = operation.position_size * price_change;
                
                // Check exit conditions
                if self.should_exit_operation(operation) {
                    operations_to_exit.push(token_mint.clone());
                }
            }
        }

        // Execute exits
        for token_mint in operations_to_exit {
            self.execute_extraction(&token_mint).await?;
        }

        Ok(())
    }

    fn should_exit_operation(&self, operation: &CounterRugOperation) -> bool {
        if operation.entry_price <= 0.0 {
            return false; // Trade hasn't executed yet
        }
        
        let price_change = (operation.current_price - operation.entry_price) / operation.entry_price;
        
        // Profit target hit
        if price_change >= self.config.profit_target {
            return true;
        }
        
        // Stop loss hit 
        if price_change <= -self.config.stop_loss {
            return true;
        }
        
        // Time limit exceeded
        if Utc::now() >= operation.target_exit_time {
            return true;
        }
        
        // TODO: Add more sophisticated exit signals:
        // - Cluster behavior changes (coordinated selling starts)
        // - Volume spike indicating dump beginning
        // - Liquidity dropping significantly
        
        false
    }

    async fn execute_extraction(&mut self, token_mint: &str) -> Result<()> {
        if let Some(mut operation) = self.active_operations.remove(token_mint) {
            operation.phase = TradingPhase::Extraction;
            
            let exit_reason = self.determine_exit_reason(&operation);
            let is_profitable = operation.unrealized_pnl > 0.0;
            
            // Special victory message for profitable extractions
            if is_profitable && matches!(exit_reason, ExitTrigger::ProfitTarget) {
                // Record the kill in our RuneScape ranking system
                let rs_notifications = self.runescape_ranking.record_kill(&operation, operation.unrealized_pnl).await?;
                
                // 🎯 SCAMMER GET SCAMMED! 🎯
                combat_logger::severity::info(
                    "counter_rug_warfare",
                    "VICTORY_SCAMMER_GET_SCAMMED",
                    serde_json::json!({
                        "message": "🎯 SCAMMER GET SCAMMED! 🎯",
                        "operation_id": operation.operation_id,
                        "token": token_mint,
                        "profit_extracted": operation.unrealized_pnl,
                        "profit_percentage": if operation.entry_price > 0.0 { 
                            ((operation.current_price - operation.entry_price) / operation.entry_price * 100.0) 
                        } else { 0.0 },
                        "hold_duration_minutes": (Utc::now() - operation.entry_time).num_minutes(),
                        "victory_status": "MISSION_ACCOMPLISHED",
                        "taunt": "We turned their own scam into our profit! 💰"
                    }),
                    Some(self.context.clone())
                ).await?;
                
                // Record trade result in emergency manager
                self.emergency_manager.record_trade_result(operation.unrealized_pnl, &operation.operation_id).await?;
                
                // Update victory statistics
                self.scammer_scammed_count += 1;
                self.total_scammer_profit_extracted += operation.unrealized_pnl;
                
                // Also print to console for immediate satisfaction with RuneScape flavor
                println!("\n🎯🎯🎯 SCAMMER GET SCAMMED! 🎯🎯🎯");
                println!("💰 Profit Extracted: {:.2} SOL ({:.1}%)", 
                    operation.unrealized_pnl,
                    if operation.entry_price > 0.0 { 
                        (operation.current_price - operation.entry_price) / operation.entry_price * 100.0 
                    } else { 0.0 }
                );
                println!("⚡ Operation: {} - MISSION ACCOMPLISHED", operation.operation_id);
                println!("🏴‍☠️ Their rug pull became our treasure chest!");
                println!("🏆 Total Scammers Defeated: {}", self.scammer_scammed_count);
                println!("💎 Total Profit from Scammers: {:.2} SOL", self.total_scammer_profit_extracted);
                
                // Display RuneScape notifications
                for notification in rs_notifications {
                    println!("🗡️ {}", notification);
                }
                
                println!("🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯\n");
            } else {
                // Regular extraction log
                combat_logger::severity::info(
                    "counter_rug_warfare",
                    "extraction_initiated",
                    serde_json::json!({
                        "operation_id": operation.operation_id,
                        "token": token_mint,
                        "unrealized_pnl": operation.unrealized_pnl,
                        "exit_reason": format!("{:?}", exit_reason),
                        "hold_duration_minutes": (Utc::now() - operation.entry_time).num_minutes()
                    }),
                    Some(self.context.clone())
                ).await?;
            }

            // Execute ultra-fast sell order
            match self.execute_ultra_fast_sell(token_mint, operation.position_size, 0.05).await {
                Ok(trades) => {
                    if !trades.is_empty() {
                        let exit_price = trades[0].price;
                        let total_proceeds: f64 = trades.iter().map(|t| t.size * t.price).sum();
                        
                        combat_logger::severity::info(
                            "counter_rug_warfare",
                            "ultra_fast_exit_success",
                            serde_json::json!({
                                "operation_id": operation.operation_id,
                                "exit_price": exit_price,
                                "total_proceeds": total_proceeds,
                                "execution_latency_ns": "sub_microsecond"
                            }),
                            Some(self.context.clone())
                        ).await?;
                    } else {
                        // Fallback to traditional execution
                        let trade_command = TradeCommand {
                            command_type: TradeCommandType::Sell,
                            token_mint: token_mint.to_string(),
                            amount: operation.position_size,
                            max_slippage: 0.05,
                            operation_id: operation.operation_id.clone(),
                        };
                        self.trade_executor.send(trade_command)?;
                    }
                },
                Err(e) => {
                    // Fallback to traditional execution
                    combat_logger::severity::warn(
                        "counter_rug_warfare",
                        "ultra_fast_exit_fallback",
                        serde_json::json!({
                            "operation_id": operation.operation_id,
                            "error": e.to_string(),
                            "fallback": "traditional_execution"
                        }),
                        Some(self.context.clone())
                    ).await?;
                    
                    let trade_command = TradeCommand {
                        command_type: TradeCommandType::Sell,
                        token_mint: token_mint.to_string(),
                        amount: operation.position_size,
                        max_slippage: 0.05,
                        operation_id: operation.operation_id.clone(),
                    };
                    self.trade_executor.send(trade_command)?;
                }
            }
            
            // Close position in position manager
            self.position_manager.close_position(&operation.operation_id, operation.unrealized_pnl).await?;
            
            // Return capital to available pool (will be updated with actual proceeds)
            self.available_capital += operation.position_size;
            
            // Archive the operation
            operation.phase = TradingPhase::PostMortem;
            self.operation_history.push_back(operation);
            
            // Keep history size manageable
            if self.operation_history.len() > 100 {
                self.operation_history.pop_front();
            }
        }

        Ok(())
    }

    fn determine_exit_reason(&self, operation: &CounterRugOperation) -> ExitTrigger {
        if operation.entry_price <= 0.0 {
            return ExitTrigger::EmergencyExit;
        }
        
        let price_change = (operation.current_price - operation.entry_price) / operation.entry_price;
        
        if price_change >= self.config.profit_target {
            ExitTrigger::ProfitTarget
        } else if price_change <= -self.config.stop_loss {
            ExitTrigger::StopLoss
        } else if Utc::now() >= operation.target_exit_time {
            ExitTrigger::TimeLimit
        } else {
            ExitTrigger::ClusterBehaviorChange
        }
    }

    async fn update_operation_intelligence(&mut self, token_mint: &str, cluster: EarlyInvestorCluster) -> Result<()> {
        if let Some(operation) = self.active_operations.get_mut(token_mint) {
            // Update cluster intelligence
            operation.cluster_info.coordination_score = cluster.coordination_score;
            
            // TODO: Implement more sophisticated intelligence updates
            // - Track if cluster is starting to sell
            // - Update time estimates for dump
            // - Adjust exit triggers based on new data
            
            combat_logger::severity::debug(
                "counter_rug_warfare",
                "intelligence_updated",
                serde_json::json!({
                    "operation_id": operation.operation_id,
                    "token": token_mint,
                    "new_coordination_score": cluster.coordination_score,
                    "cluster_risk_level": cluster.risk_level
                }),
                Some(self.context.clone())
            ).await?;
        }
        
        Ok(())
    }

    /// Get RuneScape-style player stats  
    pub fn get_player_stats(&self) -> String {
        self.runescape_ranking.get_stats_display()
    }

    /// Get recent kill log
    pub fn get_kill_log(&self, limit: usize) -> Vec<String> {
        self.runescape_ranking.get_kill_log(limit)
    }

    /// Get comprehensive risk report
    pub async fn get_risk_report(&self) -> crate::war::position_management::RiskReport {
        self.position_manager.get_risk_report().await
    }

    /// Get emergency system status
    pub async fn get_emergency_status(&self) -> crate::war::circuit_breakers::SystemStatus {
        self.emergency_manager.get_system_status().await
    }

    /// Manual emergency shutdown (use with extreme caution)
    pub async fn emergency_shutdown(&self, reason: String) -> Result<()> {
        self.emergency_manager.emergency_shutdown(reason).await
    }

    /// Check if system can proceed with trading
    pub async fn can_trade(&self) -> Result<bool> {
        self.emergency_manager.can_proceed().await
    }

    /// Execute ultra-fast buy order using the low-latency engine
    async fn execute_ultra_fast_buy(&mut self, token_mint: &str, size: f64, max_slippage: f64) -> Result<Vec<UltraLowLatencyTrade>> {
        // Calculate maximum price based on current market and slippage tolerance
        let base_price = 1.0; // TODO: Get real current price
        let max_price = base_price * (1.0 + max_slippage);
        
        self.ultra_low_latency_engine.execute_ultra_fast_trade(token_mint, size, max_price).await
    }
    
    /// Execute ultra-fast sell order using the low-latency engine
    async fn execute_ultra_fast_sell(&mut self, token_mint: &str, size: f64, max_slippage: f64) -> Result<Vec<UltraLowLatencyTrade>> {
        // For sell orders, we use minimum acceptable price
        let base_price = 1.0; // TODO: Get real current price
        let min_price = base_price * (1.0 - max_slippage);
        
        // For selling, we treat min_price as max_price in the engine (it's the limit)
        self.ultra_low_latency_engine.execute_ultra_fast_trade(token_mint, size, min_price).await
    }
    
    /// Get ultra-low latency performance report
    pub fn get_ultra_latency_performance(&self) -> crate::war::ultra_low_latency::PerformanceReport {
        self.ultra_low_latency_engine.get_performance_report()
    }

    /// Get performance statistics
    pub fn get_warfare_stats(&self) -> WarfareStats {
        let completed_ops: Vec<_> = self.operation_history.iter()
            .filter(|op| op.phase == TradingPhase::PostMortem)
            .collect();

        let total_trades = completed_ops.len();
        let winning_trades = completed_ops.iter().filter(|op| op.unrealized_pnl > 0.0).count();
        let total_pnl: f64 = completed_ops.iter().map(|op| op.unrealized_pnl).sum();
        
        let win_rate = if total_trades > 0 {
            winning_trades as f64 / total_trades as f64
        } else {
            0.0
        };

        let average_profit_per_scammer = if self.scammer_scammed_count > 0 {
            self.total_scammer_profit_extracted / self.scammer_scammed_count as f64
        } else {
            0.0
        };

        WarfareStats {
            active_operations: self.active_operations.len(),
            total_completed_operations: total_trades,
            win_rate: win_rate * 100.0,
            total_pnl,
            available_capital: self.available_capital,
            capital_utilization: (self.reserved_capital + self.available_capital - self.available_capital) / (self.reserved_capital + self.available_capital) * 100.0,
            scammers_defeated: self.scammer_scammed_count,
            total_scammer_profit_extracted: self.total_scammer_profit_extracted,
            average_profit_per_scammer,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct WarfareStats {
    pub active_operations: usize,
    pub total_completed_operations: usize,
    pub win_rate: f64,
    pub total_pnl: f64,
    pub available_capital: f64,
    pub capital_utilization: f64,
    pub scammers_defeated: u64,
    pub total_scammer_profit_extracted: f64,
    pub average_profit_per_scammer: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::{broadcast, mpsc};

    #[tokio::test]
    async fn test_engagement_criteria() {
        let config = CounterRugConfig::default();
        let (_, alert_receiver) = broadcast::channel(10);
        let (trade_sender, _) = mpsc::unbounded_channel();
        
        let strategy = CounterRugPullStrategy::new(
            config,
            1000.0, // 1000 SOL initial capital
            alert_receiver,
            trade_sender,
        );

        // Create a test alert
        let alert = RugPullAlert {
            alert_id: "test_123".to_string(),
            token_mint: "test_token".to_string(),
            clusters: vec![EarlyInvestorCluster {
                cluster_id: "cluster_1".to_string(),
                wallets: vec!["w1".to_string(), "w2".to_string(), "w3".to_string(), "w4".to_string(), "w5".to_string()],
                token_mint: "test_token".to_string(),
                first_purchase_time: Utc::now() - Duration::hours(1),
                total_investment: 600.0,
                coordination_score: 0.8,
                behavioral_flags: vec![],
                risk_level: RiskLevel::High,
            }],
            overall_risk_score: 0.8,
            confidence: 0.85,
            timestamp: Utc::now(),
            recommended_actions: vec![],
            evidence: vec![],
        };

        assert!(strategy.should_engage(&alert));
    }

    #[test]
    fn test_position_sizing() {
        let config = CounterRugConfig::default();
        let (_, alert_receiver) = broadcast::channel(10);
        let (trade_sender, _) = mpsc::unbounded_channel();
        
        let strategy = CounterRugPullStrategy::new(config, 1000.0, alert_receiver, trade_sender);

        let alert = RugPullAlert {
            alert_id: "test_123".to_string(),
            token_mint: "test_token".to_string(),
            clusters: vec![EarlyInvestorCluster {
                cluster_id: "cluster_1".to_string(),
                wallets: vec!["wallet1".to_string()],
                token_mint: "test_token".to_string(),
                first_purchase_time: Utc::now(),
                total_investment: 1000.0,
                coordination_score: 0.9,
                behavioral_flags: vec![],
                risk_level: RiskLevel::Critical,
            }],
            overall_risk_score: 0.9,
            confidence: 0.95,
            timestamp: Utc::now(),
            recommended_actions: vec![],
            evidence: vec![],
        };

        let position_size = strategy.calculate_position_size(&alert).unwrap();
        assert!(position_size > 0.0);
        assert!(position_size <= strategy.available_capital * 0.8);
    }
}