//! Enhanced Position Management & Risk Safeguards
//! 
//! Proactive risk management to prevent losses before they happen.
//! Dynamic position sizing, correlation limits, liquidity checks, and slippage protection.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc, Duration};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{warn, error, info, debug};

use crate::analytics::Transaction;
use crate::strategies::counter_rug_pull::{CounterRugOperation, TradingPhase};

/// Position sizing strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PositionSizing {
    Fixed(f64),              // Fixed SOL amount
    PercentCapital(f64),     // Percentage of total capital
    KellyOptimal(f64),       // Kelly Criterion with safety factor
    VolatilityAdjusted,      // Based on recent volatility
    RiskParity,              // Equal risk contribution
    Adaptive,                // ML-driven dynamic sizing
}

/// Risk management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskConfig {
    // Position limits
    pub max_position_size_sol: f64,        // Absolute max position size
    pub max_position_pct: f64,             // Max as % of capital
    pub max_concurrent_positions: usize,    // Max simultaneous positions
    pub max_exposure_per_token: f64,       // Max exposure to single token
    
    // Correlation and concentration limits
    pub max_correlation: f64,              // Max correlation between positions
    pub max_sector_concentration: f64,     // Max in similar tokens/sectors
    pub diversification_requirement: f64,  // Min diversification score
    
    // Liquidity and slippage limits
    pub min_liquidity_sol: f64,           // Minimum liquidity required
    pub max_slippage_pct: f64,            // Maximum acceptable slippage
    pub liquidity_buffer_pct: f64,        // Buffer above min liquidity
    
    // Volatility and timing
    pub volatility_lookback_hours: u64,    // Hours to look back for volatility
    pub max_volatility: f64,               // Max volatility to allow trades
    pub volatility_adjustment_factor: f64, // How much to adjust for volatility
    
    // Dynamic adjustment parameters
    pub sizing_strategy: PositionSizing,   // Default sizing strategy
    pub risk_adjustment_speed: f64,        // How fast to adjust to new conditions
    pub confidence_threshold: f64,         // Min confidence for full size
}

impl Default for RiskConfig {
    fn default() -> Self {
        Self {
            // Conservative position limits
            max_position_size_sol: 100.0,      // 100 SOL max per position
            max_position_pct: 0.10,            // 10% of capital max
            max_concurrent_positions: 3,        // Max 3 positions
            max_exposure_per_token: 150.0,     // Max 150 SOL in same token
            
            // Diversification requirements
            max_correlation: 0.7,              // 70% max correlation
            max_sector_concentration: 0.4,     // 40% max in similar tokens  
            diversification_requirement: 0.6,   // 60% min diversification
            
            // Liquidity protection
            min_liquidity_sol: 1000.0,         // 1000 SOL min liquidity
            max_slippage_pct: 0.03,            // 3% max slippage
            liquidity_buffer_pct: 0.2,         // 20% buffer above min
            
            // Volatility management
            volatility_lookback_hours: 24,      // 24 hour lookback
            max_volatility: 0.5,               // 50% max volatility
            volatility_adjustment_factor: 2.0, // 2x volatility adjustment
            
            // Strategy defaults
            sizing_strategy: PositionSizing::VolatilityAdjusted,
            risk_adjustment_speed: 0.1,        // 10% adjustment rate
            confidence_threshold: 0.75,        // 75% confidence required
        }
    }
}

/// Individual position information
#[derive(Debug, Clone, Serialize)]
pub struct PositionInfo {
    pub operation_id: String,
    pub token_mint: String,
    pub entry_time: DateTime<Utc>,
    pub position_size: f64,
    pub entry_price: f64,
    pub current_price: f64,
    pub unrealized_pnl: f64,
    pub risk_score: f64,
    pub volatility: f64,
    pub liquidity_score: f64,
    pub correlation_scores: HashMap<String, f64>, // Correlation with other positions
}

/// Market conditions snapshot
#[derive(Debug, Clone, Serialize)]
pub struct MarketConditions {
    pub timestamp: DateTime<Utc>,
    pub overall_volatility: f64,
    pub liquidity_conditions: LiquidityConditions,
    pub correlation_regime: CorrelationRegime,
    pub market_stress_score: f64,
    pub volume_profile: VolumeProfile,
}

#[derive(Debug, Clone, Serialize)]
pub enum LiquidityConditions {
    Abundant,    // High liquidity
    Normal,      // Standard conditions
    Tight,       // Reduced liquidity
    Stressed,    // Very low liquidity
}

#[derive(Debug, Clone, Serialize)]
pub enum CorrelationRegime {
    Diversified,   // Low correlations
    Clustered,     // Some correlation clusters
    Synchronized,  // High correlations across market
}

#[derive(Debug, Clone, Serialize)]
pub struct VolumeProfile {
    pub current_volume: f64,
    pub average_volume: f64,
    pub volume_trend: f64,
    pub time_of_day_factor: f64,
}

/// Main position management system
pub struct PositionManager {
    pub config: RiskConfig,
    pub active_positions: Arc<RwLock<HashMap<String, PositionInfo>>>,
    pub position_history: Arc<RwLock<VecDeque<PositionInfo>>>,
    pub market_conditions: Arc<RwLock<MarketConditions>>,
    pub capital_base: Arc<RwLock<f64>>,
    pub available_capital: Arc<RwLock<f64>>,
    
    // Risk tracking
    pub correlation_matrix: Arc<RwLock<HashMap<String, HashMap<String, f64>>>>,
    pub volatility_tracker: Arc<RwLock<HashMap<String, VecDeque<f64>>>>,
    pub liquidity_tracker: Arc<RwLock<HashMap<String, VecDeque<f64>>>>,
    
    // Performance metrics
    pub risk_adjusted_returns: Arc<RwLock<VecDeque<f64>>>,
    pub sharpe_ratio: Arc<RwLock<f64>>,
    pub max_drawdown: Arc<RwLock<f64>>,
}

impl PositionManager {
    pub fn new(config: RiskConfig, initial_capital: f64) -> Self {
        Self {
            config,
            active_positions: Arc::new(RwLock::new(HashMap::new())),
            position_history: Arc::new(RwLock::new(VecDeque::with_capacity(1000))),
            market_conditions: Arc::new(RwLock::new(MarketConditions::default())),
            capital_base: Arc::new(RwLock::new(initial_capital)),
            available_capital: Arc::new(RwLock::new(initial_capital)),
            correlation_matrix: Arc::new(RwLock::new(HashMap::new())),
            volatility_tracker: Arc::new(RwLock::new(HashMap::new())),
            liquidity_tracker: Arc::new(RwLock::new(HashMap::new())),
            risk_adjusted_returns: Arc::new(RwLock::new(VecDeque::with_capacity(1000))),
            sharpe_ratio: Arc::new(RwLock::new(0.0)),
            max_drawdown: Arc::new(RwLock::new(0.0)),
        }
    }

    /// Calculate optimal position size for a new opportunity
    pub async fn calculate_position_size(
        &self,
        token_mint: &str,
        confidence: f64,
        expected_return: f64,
        risk_score: f64,
    ) -> Result<f64> {
        let available = *self.available_capital.read().await;
        let capital_base = *self.capital_base.read().await;
        
        // Check if we can even take a position
        if !self.can_open_position(token_mint).await? {
            return Ok(0.0);
        }

        // Get current market conditions
        let market_conditions = self.market_conditions.read().await;
        let positions = self.active_positions.read().await;
        
        // Base position size calculation
        let base_size = match &self.config.sizing_strategy {
            PositionSizing::Fixed(amount) => *amount,
            PositionSizing::PercentCapital(pct) => capital_base * pct,
            PositionSizing::KellyOptimal(safety_factor) => {
                self.kelly_position_size(expected_return, risk_score, *safety_factor).await?
            },
            PositionSizing::VolatilityAdjusted => {
                self.volatility_adjusted_size(token_mint, capital_base).await?
            },
            PositionSizing::RiskParity => {
                self.risk_parity_size(token_mint, capital_base).await?
            },
            PositionSizing::Adaptive => {
                self.adaptive_position_size(token_mint, confidence, expected_return, risk_score).await?
            },
        };

        // Apply risk adjustments
        let mut adjusted_size = base_size;
        
        // Confidence adjustment
        if confidence < self.config.confidence_threshold {
            let confidence_factor = confidence / self.config.confidence_threshold;
            adjusted_size *= confidence_factor;
            debug!("Position size reduced for low confidence: {:.2} -> {:.2}", base_size, adjusted_size);
        }

        // Market stress adjustment  
        if market_conditions.market_stress_score > 0.7 {
            let stress_factor = 1.0 - (market_conditions.market_stress_score - 0.7) * 2.0;
            adjusted_size *= stress_factor.max(0.2); // Never reduce below 20%
            debug!("Position size reduced for market stress: {:.2}", stress_factor);
        }

        // Liquidity adjustment
        if matches!(market_conditions.liquidity_conditions, LiquidityConditions::Tight | LiquidityConditions::Stressed) {
            adjusted_size *= 0.5; // Reduce size in tight liquidity
            debug!("Position size reduced for tight liquidity");
        }

        // Correlation adjustment
        let correlation_penalty = self.calculate_correlation_penalty(token_mint).await?;
        adjusted_size *= (1.0 - correlation_penalty);

        // Apply absolute limits
        adjusted_size = adjusted_size
            .min(self.config.max_position_size_sol)
            .min(capital_base * self.config.max_position_pct)
            .min(available * 0.8); // Never use more than 80% of available capital

        // Check exposure limits
        let current_exposure = self.get_token_exposure(token_mint).await?;
        let max_additional = self.config.max_exposure_per_token - current_exposure;
        adjusted_size = adjusted_size.min(max_additional.max(0.0));

        info!("Position size calculated for {}: {:.2} SOL (base: {:.2}, confidence: {:.2})", 
              token_mint, adjusted_size, base_size, confidence);

        Ok(adjusted_size)
    }

    /// Check if we can open a new position
    pub async fn can_open_position(&self, token_mint: &str) -> Result<bool> {
        let positions = self.active_positions.read().await;
        
        // Check position count limit
        if positions.len() >= self.config.max_concurrent_positions {
            debug!("Cannot open position: max concurrent positions reached");
            return Ok(false);
        }

        // Check if we already have exposure to this token
        let current_exposure = self.get_token_exposure(token_mint).await?;
        if current_exposure >= self.config.max_exposure_per_token {
            debug!("Cannot open position: max token exposure reached for {}", token_mint);
            return Ok(false);
        }

        // Check available capital
        let available = *self.available_capital.read().await;
        let min_position = self.config.max_position_size_sol * 0.01; // 1% of max position
        if available < min_position {
            debug!("Cannot open position: insufficient capital");
            return Ok(false);
        }

        // Check market conditions
        let market_conditions = self.market_conditions.read().await;
        if matches!(market_conditions.liquidity_conditions, LiquidityConditions::Stressed) {
            debug!("Cannot open position: stressed liquidity conditions");
            return Ok(false);
        }

        if market_conditions.overall_volatility > self.config.max_volatility {
            debug!("Cannot open position: excessive volatility ({:.2})", market_conditions.overall_volatility);
            return Ok(false);
        }

        Ok(true)
    }

    /// Validate a position before opening
    pub async fn validate_position(
        &self,
        operation: &CounterRugOperation,
        proposed_size: f64,
    ) -> Result<PositionValidation> {
        let mut validation = PositionValidation::new();

        // Size validation
        if proposed_size > self.config.max_position_size_sol {
            validation.add_violation(
                PositionViolationType::ExcessiveSize,
                format!("Position size {:.2} exceeds max {:.2}", proposed_size, self.config.max_position_size_sol),
                RiskLevel::High,
            );
        }

        // Capital validation
        let available = *self.available_capital.read().await;
        if proposed_size > available {
            validation.add_violation(
                PositionViolationType::InsufficientCapital,
                format!("Position size {:.2} exceeds available capital {:.2}", proposed_size, available),
                RiskLevel::Critical,
            );
        }

        // Exposure validation
        let current_exposure = self.get_token_exposure(&operation.token_mint).await?;
        let total_exposure = current_exposure + proposed_size;
        if total_exposure > self.config.max_exposure_per_token {
            validation.add_violation(
                PositionViolationType::ExcessiveExposure,
                format!("Total token exposure {:.2} would exceed limit {:.2}", 
                       total_exposure, self.config.max_exposure_per_token),
                RiskLevel::High,
            );
        }

        // Correlation validation
        let correlation_risk = self.assess_correlation_risk(&operation.token_mint, proposed_size).await?;
        if correlation_risk > self.config.max_correlation {
            validation.add_violation(
                PositionViolationType::HighCorrelation,
                format!("Position would create correlation risk of {:.2} (max: {:.2})", 
                       correlation_risk, self.config.max_correlation),
                RiskLevel::Medium,
            );
        }

        // Liquidity validation
        let liquidity_check = self.check_liquidity_requirements(&operation.token_mint, proposed_size).await?;
        if !liquidity_check.sufficient {
            validation.add_violation(
                PositionViolationType::InsufficientLiquidity,
                format!("Insufficient liquidity: required {:.2}, available {:.2}", 
                       liquidity_check.required, liquidity_check.available),
                RiskLevel::High,
            );
        }

        // Slippage validation
        let expected_slippage = self.estimate_slippage(&operation.token_mint, proposed_size).await?;
        if expected_slippage > self.config.max_slippage_pct {
            validation.add_violation(
                PositionViolationType::ExcessiveSlippage,
                format!("Expected slippage {:.2}% exceeds max {:.2}%", 
                       expected_slippage * 100.0, self.config.max_slippage_pct * 100.0),
                RiskLevel::Medium,
            );
        }

        // Risk score validation
        if operation.risk_score > 0.9 {
            validation.add_warning(
                "Very high risk score detected".to_string(),
                "Consider reducing position size or avoiding trade".to_string(),
            );
        }

        Ok(validation)
    }

    /// Open a new position
    pub async fn open_position(&self, operation: &CounterRugOperation) -> Result<()> {
        let position = PositionInfo {
            operation_id: operation.operation_id.clone(),
            token_mint: operation.token_mint.clone(),
            entry_time: operation.entry_time,
            position_size: operation.position_size,
            entry_price: operation.entry_price,
            current_price: operation.current_price,
            unrealized_pnl: operation.unrealized_pnl,
            risk_score: operation.risk_score,
            volatility: self.get_token_volatility(&operation.token_mint).await.unwrap_or(0.0),
            liquidity_score: self.get_liquidity_score(&operation.token_mint).await.unwrap_or(0.0),
            correlation_scores: HashMap::new(),
        };

        // Update position tracking
        {
            let mut positions = self.active_positions.write().await;
            positions.insert(operation.operation_id.clone(), position);
        }

        // Update available capital
        {
            let mut available = self.available_capital.write().await;
            *available -= operation.position_size;
        }

        // Update correlation matrix
        self.update_correlations(&operation.token_mint).await?;

        info!("Position opened: {} for {:.2} SOL in {}", 
              operation.operation_id, operation.position_size, operation.token_mint);

        Ok(())
    }

    /// Close a position
    pub async fn close_position(&self, operation_id: &str, final_pnl: f64) -> Result<()> {
        let position = {
            let mut positions = self.active_positions.write().await;
            positions.remove(operation_id)
        };

        if let Some(mut position) = position {
            position.unrealized_pnl = final_pnl;
            
            // Return capital
            {
                let mut available = self.available_capital.write().await;
                *available += position.position_size + final_pnl;
            }

            // Archive position
            {
                let mut history = self.position_history.write().await;
                history.push_back(position.clone());
                if history.len() > 1000 {
                    history.pop_front();
                }
            }

            // Update performance metrics
            self.update_performance_metrics(final_pnl).await?;

            info!("Position closed: {} with P&L {:.2} SOL", operation_id, final_pnl);
        }

        Ok(())
    }

    /// Update position with current market data
    pub async fn update_position(&self, operation_id: &str, current_price: f64, current_pnl: f64) -> Result<()> {
        let mut positions = self.active_positions.write().await;
        if let Some(position) = positions.get_mut(operation_id) {
            position.current_price = current_price;
            position.unrealized_pnl = current_pnl;
        }
        Ok(())
    }

    // Helper methods for position sizing strategies

    async fn kelly_position_size(&self, expected_return: f64, risk: f64, safety_factor: f64) -> Result<f64> {
        let capital = *self.capital_base.read().await;
        
        // Kelly formula: f = (bp - q) / b
        // where f = fraction of capital to bet
        // b = odds received (return ratio)  
        // p = probability of winning
        // q = probability of losing (1-p)
        
        let win_prob = 0.5 + (expected_return / 2.0); // Convert expected return to probability
        let loss_prob = 1.0 - win_prob;
        let odds = expected_return.abs();
        
        if odds == 0.0 || win_prob <= loss_prob {
            return Ok(0.0);
        }
        
        let kelly_fraction = ((odds * win_prob) - loss_prob) / odds;
        let safe_kelly = kelly_fraction * safety_factor;
        
        Ok((capital * safe_kelly.max(0.0)).min(capital * 0.25)) // Cap at 25% of capital
    }

    async fn volatility_adjusted_size(&self, token_mint: &str, capital: f64) -> Result<f64> {
        let volatility = self.get_token_volatility(token_mint).await?;
        let base_volatility = 0.2; // 20% base volatility
        
        // Inverse relationship: higher volatility = smaller position
        let volatility_factor = (base_volatility / volatility.max(0.01)).min(2.0);
        let base_size = capital * self.config.max_position_pct;
        
        Ok(base_size * volatility_factor)
    }

    async fn risk_parity_size(&self, token_mint: &str, capital: f64) -> Result<f64> {
        let positions = self.active_positions.read().await;
        let num_positions = positions.len();
        
        if num_positions == 0 {
            return Ok(capital * self.config.max_position_pct);
        }
        
        // Equal risk contribution across all positions
        let target_positions = (self.config.max_concurrent_positions as f64).min(8.0);
        let equal_risk_allocation = 1.0 / target_positions;
        
        Ok(capital * equal_risk_allocation)
    }

    async fn adaptive_position_size(
        &self,
        token_mint: &str,
        confidence: f64,
        expected_return: f64,
        risk_score: f64,
    ) -> Result<f64> {
        let capital = *self.capital_base.read().await;
        
        // Combine multiple factors
        let volatility = self.get_token_volatility(token_mint).await?;
        let sharpe = *self.sharpe_ratio.read().await;
        
        // Base size from percentage
        let mut size = capital * self.config.max_position_pct;
        
        // Confidence adjustment
        size *= confidence;
        
        // Expected return adjustment  
        size *= (1.0 + expected_return).max(0.5).min(2.0);
        
        // Risk adjustment (inverse)
        size *= (2.0 - risk_score).max(0.1);
        
        // Volatility adjustment
        size *= (0.5 / volatility.max(0.1)).min(2.0);
        
        // Historical performance adjustment
        if sharpe > 1.0 {
            size *= 1.2; // Increase size if good Sharpe ratio
        } else if sharpe < 0.5 {
            size *= 0.8; // Decrease size if poor Sharpe ratio
        }
        
        Ok(size)
    }

    // Risk assessment helper methods

    async fn calculate_correlation_penalty(&self, token_mint: &str) -> Result<f64> {
        let positions = self.active_positions.read().await;
        let correlation_matrix = self.correlation_matrix.read().await;
        
        let mut max_correlation = 0.0;
        for position in positions.values() {
            if let Some(token_correlations) = correlation_matrix.get(&position.token_mint) {
                if let Some(correlation) = token_correlations.get(token_mint) {
                    max_correlation = max_correlation.max(correlation.abs());
                }
            }
        }
        
        // Penalty increases exponentially with correlation
        let penalty = if max_correlation > self.config.max_correlation {
            (max_correlation - self.config.max_correlation) * 2.0
        } else {
            0.0
        };
        
        Ok(penalty.min(0.8)) // Cap penalty at 80%
    }

    async fn assess_correlation_risk(&self, token_mint: &str, position_size: f64) -> Result<f64> {
        // Simplified correlation risk - in production would use real correlation data
        let positions = self.active_positions.read().await;
        
        if positions.is_empty() {
            return Ok(0.0);
        }
        
        // Estimate correlation risk based on position concentration
        let total_capital = *self.capital_base.read().await;
        let total_exposure: f64 = positions.values().map(|p| p.position_size).sum();
        let new_total_exposure = total_exposure + position_size;
        
        let concentration_risk = new_total_exposure / total_capital;
        Ok(concentration_risk.min(1.0))
    }

    async fn get_token_exposure(&self, token_mint: &str) -> Result<f64> {
        let positions = self.active_positions.read().await;
        let exposure = positions.values()
            .filter(|p| p.token_mint == token_mint)
            .map(|p| p.position_size)
            .sum();
        Ok(exposure)
    }

    async fn get_token_volatility(&self, token_mint: &str) -> Result<f64> {
        let volatility_tracker = self.volatility_tracker.read().await;
        if let Some(vol_history) = volatility_tracker.get(token_mint) {
            if let Some(recent_vol) = vol_history.back() {
                return Ok(*recent_vol);
            }
        }
        Ok(0.3) // Default 30% volatility
    }

    async fn get_liquidity_score(&self, token_mint: &str) -> Result<f64> {
        // Simplified liquidity scoring - in production would use real DEX data
        Ok(0.8) // Default good liquidity score
    }

    async fn check_liquidity_requirements(&self, token_mint: &str, position_size: f64) -> Result<LiquidityCheck> {
        let required_liquidity = position_size * (1.0 + self.config.liquidity_buffer_pct);
        let available_liquidity = self.config.min_liquidity_sol; // Simplified
        
        Ok(LiquidityCheck {
            required: required_liquidity,
            available: available_liquidity,
            sufficient: available_liquidity >= required_liquidity,
        })
    }

    async fn estimate_slippage(&self, token_mint: &str, position_size: f64) -> Result<f64> {
        // Simplified slippage estimation - in production would use orderbook data
        let liquidity = self.config.min_liquidity_sol;
        let slippage = (position_size / liquidity).min(0.1); // Cap at 10%
        Ok(slippage)
    }

    async fn update_correlations(&self, token_mint: &str) -> Result<()> {
        // Simplified correlation update - in production would calculate real correlations
        // This is a placeholder for the correlation matrix update logic
        Ok(())
    }

    async fn update_performance_metrics(&self, pnl: f64) -> Result<()> {
        // Update risk-adjusted returns
        {
            let mut returns = self.risk_adjusted_returns.write().await;
            returns.push_back(pnl);
            if returns.len() > 1000 {
                returns.pop_front();
            }
        }

        // Calculate and update Sharpe ratio
        {
            let returns = self.risk_adjusted_returns.read().await;
            if returns.len() >= 10 {
                let mean_return: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
                let variance: f64 = returns.iter()
                    .map(|r| (r - mean_return).powi(2))
                    .sum::<f64>() / returns.len() as f64;
                let std_dev = variance.sqrt();
                
                let new_sharpe = if std_dev > 0.0 { mean_return / std_dev } else { 0.0 };
                *self.sharpe_ratio.write().await = new_sharpe;
            }
        }

        Ok(())
    }

    /// Get comprehensive risk report
    pub async fn get_risk_report(&self) -> RiskReport {
        let positions = self.active_positions.read().await;
        let capital_base = *self.capital_base.read().await;
        let available = *self.available_capital.read().await;
        let market_conditions = self.market_conditions.read().await;
        
        let total_exposure: f64 = positions.values().map(|p| p.position_size).sum();
        let total_unrealized_pnl: f64 = positions.values().map(|p| p.unrealized_pnl).sum();
        
        let mut token_exposures = HashMap::new();
        for position in positions.values() {
            *token_exposures.entry(position.token_mint.clone()).or_insert(0.0) += position.position_size;
        }
        
        RiskReport {
            timestamp: Utc::now(),
            total_capital: capital_base,
            available_capital: available,
            total_exposure,
            capital_utilization: total_exposure / capital_base,
            total_unrealized_pnl,
            active_positions: positions.len(),
            max_concurrent_positions: self.config.max_concurrent_positions,
            token_exposures,
            market_conditions: market_conditions.clone(),
            sharpe_ratio: *self.sharpe_ratio.read().await,
            max_drawdown: *self.max_drawdown.read().await,
            risk_score: self.calculate_portfolio_risk_score().await,
        }
    }

    async fn calculate_portfolio_risk_score(&self) -> f64 {
        let positions = self.active_positions.read().await;
        if positions.is_empty() {
            return 0.0;
        }
        
        let avg_risk: f64 = positions.values().map(|p| p.risk_score).sum::<f64>() / positions.len() as f64;
        let capital_base = *self.capital_base.read().await;
        let total_exposure: f64 = positions.values().map(|p| p.position_size).sum();
        let utilization = total_exposure / capital_base;
        
        // Combine individual position risk with portfolio utilization
        (avg_risk * 0.7 + utilization * 0.3).min(1.0)
    }
}

// Supporting structures

impl Default for MarketConditions {
    fn default() -> Self {
        Self {
            timestamp: Utc::now(),
            overall_volatility: 0.2,
            liquidity_conditions: LiquidityConditions::Normal,
            correlation_regime: CorrelationRegime::Diversified,
            market_stress_score: 0.3,
            volume_profile: VolumeProfile {
                current_volume: 1000.0,
                average_volume: 1000.0,
                volume_trend: 1.0,
                time_of_day_factor: 1.0,
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct LiquidityCheck {
    pub required: f64,
    pub available: f64,
    pub sufficient: bool,
}

#[derive(Debug, Clone)]
pub struct PositionValidation {
    pub violations: Vec<PositionViolation>,
    pub warnings: Vec<String>,
    pub is_valid: bool,
}

#[derive(Debug, Clone)]
pub struct PositionViolation {
    pub violation_type: PositionViolationType,
    pub message: String,
    pub risk_level: RiskLevel,
}

#[derive(Debug, Clone)]
pub enum PositionViolationType {
    ExcessiveSize,
    InsufficientCapital,
    ExcessiveExposure,
    HighCorrelation,
    InsufficientLiquidity,
    ExcessiveSlippage,
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum RiskLevel {
    Low,
    Medium, 
    High,
    Critical,
}

impl PositionValidation {
    pub fn new() -> Self {
        Self {
            violations: Vec::new(),
            warnings: Vec::new(),
            is_valid: true,
        }
    }
    
    pub fn add_violation(&mut self, violation_type: PositionViolationType, message: String, risk_level: RiskLevel) {
        self.violations.push(PositionViolation {
            violation_type,
            message,
            risk_level,
        });
        self.is_valid = false;
    }
    
    pub fn add_warning(&mut self, warning: String, _suggestion: String) {
        self.warnings.push(warning);
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct RiskReport {
    pub timestamp: DateTime<Utc>,
    pub total_capital: f64,
    pub available_capital: f64,
    pub total_exposure: f64,
    pub capital_utilization: f64,
    pub total_unrealized_pnl: f64,
    pub active_positions: usize,
    pub max_concurrent_positions: usize,
    pub token_exposures: HashMap<String, f64>,
    pub market_conditions: MarketConditions,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub risk_score: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategies::counter_rug_pull::ClusterIntel;

    #[tokio::test]
    async fn test_position_sizing() {
        let config = RiskConfig::default();
        let manager = PositionManager::new(config, 1000.0);
        
        let size = manager.calculate_position_size(
            "test_token",
            0.8,  // 80% confidence
            0.15, // 15% expected return
            0.3,  // 30% risk score
        ).await.unwrap();
        
        assert!(size > 0.0);
        assert!(size <= 100.0); // Should respect max position size
        println!("Calculated position size: {:.2} SOL", size);
    }

    #[tokio::test]
    async fn test_position_validation() {
        let config = RiskConfig {
            max_position_size_sol: 50.0,
            max_exposure_per_token: 75.0,
            ..Default::default()
        };
        let manager = PositionManager::new(config, 1000.0);
        
        let operation = CounterRugOperation {
            operation_id: "test_op".to_string(),
            token_mint: "test_token".to_string(),
            phase: TradingPhase::Infiltration,
            entry_time: Utc::now(),
            entry_price: 1.0,
            position_size: 60.0, // Exceeds max
            current_price: 1.0,
            unrealized_pnl: 0.0,
            target_exit_time: Utc::now() + Duration::hours(1),
            risk_score: 0.8,
            cluster_info: ClusterIntel {
                cluster_count: 1,
                total_wallets: 5,
                coordination_score: 0.8,
                estimated_total_supply: 1_000_000.0,
                cluster_accumulation_rate: 0.0,
                time_to_estimated_dump: None,
            },
            exit_triggers: vec![],
        };
        
        let validation = manager.validate_position(&operation, 60.0).await.unwrap();
        assert!(!validation.is_valid); // Should be invalid due to excessive size
        assert!(!validation.violations.is_empty());
        
        println!("Validation violations: {:#?}", validation.violations);
    }

    #[tokio::test]
    async fn test_risk_report() {
        let config = RiskConfig::default();
        let manager = PositionManager::new(config, 1000.0);
        
        // Add a test position
        let operation = CounterRugOperation {
            operation_id: "test_op".to_string(),
            token_mint: "test_token".to_string(),
            phase: TradingPhase::Infiltration,
            entry_time: Utc::now(),
            entry_price: 1.0,
            position_size: 50.0,
            current_price: 1.05,
            unrealized_pnl: 2.5,
            target_exit_time: Utc::now() + Duration::hours(1),
            risk_score: 0.6,
            cluster_info: ClusterIntel {
                cluster_count: 1,
                total_wallets: 5,
                coordination_score: 0.8,
                estimated_total_supply: 1_000_000.0,
                cluster_accumulation_rate: 0.0,
                time_to_estimated_dump: None,
            },
            exit_triggers: vec![],
        };
        
        manager.open_position(&operation).await.unwrap();
        
        let report = manager.get_risk_report().await;
        assert_eq!(report.active_positions, 1);
        assert_eq!(report.total_exposure, 50.0);
        assert_eq!(report.capital_utilization, 0.05); // 5% utilization
        
        println!("Risk report: {:#?}", report);
    }
}