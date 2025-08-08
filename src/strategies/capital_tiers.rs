use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use anyhow::Result;
use tracing::{info, warn};

/// **CAPITAL TIER CLASSIFICATION SYSTEM**
/// Sophisticated strategy allocation based on available capital
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum CapitalTier {
    /// **NANO** - $100 - $1K: High frequency, low risk micro-strategies
    Nano,
    /// **MICRO** - $1K - $10K: Focused arbitrage and sniper strategies  
    Micro,
    /// **SMALL** - $10K - $100K: Multi-strategy diversification
    Small,
    /// **MEDIUM** - $100K - $1M: Advanced MEV and market making
    Medium,
    /// **LARGE** - $1M - $10M: Institutional-grade strategies
    Large,
    /// **WHALE** - $10M+: Market manipulation and ecosystem strategies
    Whale,
    /// **TITAN** - $100M+: Cross-chain arbitrage and liquidity provision
    Titan,
}

impl CapitalTier {
    pub fn from_balance(balance_usd: f64) -> Self {
        match balance_usd {
            b if b >= 100_000_000.0 => CapitalTier::Titan,
            b if b >= 10_000_000.0 => CapitalTier::Whale,
            b if b >= 1_000_000.0 => CapitalTier::Large,
            b if b >= 100_000.0 => CapitalTier::Medium,
            b if b >= 10_000.0 => CapitalTier::Small,
            b if b >= 1_000.0 => CapitalTier::Micro,
            _ => CapitalTier::Nano,
        }
    }

    pub fn min_balance(&self) -> f64 {
        match self {
            CapitalTier::Nano => 100.0,
            CapitalTier::Micro => 1_000.0,
            CapitalTier::Small => 10_000.0,
            CapitalTier::Medium => 100_000.0,
            CapitalTier::Large => 1_000_000.0,
            CapitalTier::Whale => 10_000_000.0,
            CapitalTier::Titan => 100_000_000.0,
        }
    }

    pub fn max_balance(&self) -> f64 {
        match self {
            CapitalTier::Nano => 999.0,
            CapitalTier::Micro => 9_999.0,
            CapitalTier::Small => 99_999.0,
            CapitalTier::Medium => 999_999.0,
            CapitalTier::Large => 9_999_999.0,
            CapitalTier::Whale => 99_999_999.0,
            CapitalTier::Titan => f64::INFINITY,
        }
    }

    pub fn description(&self) -> &str {
        match self {
            CapitalTier::Nano => "Precision micro-trading with minimal capital",
            CapitalTier::Micro => "Focused arbitrage and opportunity sniping",
            CapitalTier::Small => "Multi-strategy diversified approach", 
            CapitalTier::Medium => "Advanced MEV and market making strategies",
            CapitalTier::Large => "Institutional-grade multi-market operations",
            CapitalTier::Whale => "Market influence and ecosystem strategies",
            CapitalTier::Titan => "Cross-chain dominance and liquidity provision",
        }
    }

    pub fn risk_tolerance(&self) -> RiskTolerance {
        match self {
            CapitalTier::Nano => RiskTolerance::Conservative,
            CapitalTier::Micro => RiskTolerance::Moderate,
            CapitalTier::Small => RiskTolerance::Balanced,
            CapitalTier::Medium => RiskTolerance::Aggressive,
            CapitalTier::Large => RiskTolerance::VeryAggressive,
            CapitalTier::Whale => RiskTolerance::Extreme,
            CapitalTier::Titan => RiskTolerance::Unlimited,
        }
    }

    pub fn max_position_size_percentage(&self) -> f64 {
        match self {
            CapitalTier::Nano => 0.10,    // 10% max per position
            CapitalTier::Micro => 0.15,   // 15% max per position
            CapitalTier::Small => 0.20,   // 20% max per position
            CapitalTier::Medium => 0.25,  // 25% max per position
            CapitalTier::Large => 0.30,   // 30% max per position
            CapitalTier::Whale => 0.40,   // 40% max per position
            CapitalTier::Titan => 0.50,   // 50% max per position
        }
    }

    pub fn concurrent_positions(&self) -> usize {
        match self {
            CapitalTier::Nano => 3,
            CapitalTier::Micro => 5,
            CapitalTier::Small => 8,
            CapitalTier::Medium => 12,
            CapitalTier::Large => 20,
            CapitalTier::Whale => 35,
            CapitalTier::Titan => 50,
        }
    }

    pub fn preferred_strategies(&self) -> Vec<StrategyType> {
        match self {
            CapitalTier::Nano => vec![
                StrategyType::MicroArbitrage,
                StrategyType::GasOptimizedSniping,
                StrategyType::LowRiskCopyTrading,
            ],
            CapitalTier::Micro => vec![
                StrategyType::FastArbitrage,
                StrategyType::TokenSniping,
                StrategyType::LiquidityFrontrunning,
                StrategyType::SmallMEV,
            ],
            CapitalTier::Small => vec![
                StrategyType::DiversifiedArbitrage,
                StrategyType::MediumMEV,
                StrategyType::PairTrading,
                StrategyType::TrendFollowing,
                StrategyType::LiquidityProvision,
            ],
            CapitalTier::Medium => vec![
                StrategyType::AdvancedMEV,
                StrategyType::MarketMaking,
                StrategyType::CrossDEXArbitrage,
                StrategyType::LeveragedStrategies,
                StrategyType::VolatilityTrading,
            ],
            CapitalTier::Large => vec![
                StrategyType::InstitutionalMEV,
                StrategyType::LiquidityAggregation,
                StrategyType::DerivativesArbitrage,
                StrategyType::YieldFarming,
                StrategyType::StructuredProducts,
            ],
            CapitalTier::Whale => vec![
                StrategyType::MarketManipulation,
                StrategyType::EcosystemInfluence,
                StrategyType::MegaLiquidityProvision,
                StrategyType::TokenLaunching,
                StrategyType::GovernanceCapture,
            ],
            CapitalTier::Titan => vec![
                StrategyType::CrossChainDominance,
                StrategyType::ProtocolOwnership,
                StrategyType::EcosystemCreation,
                StrategyType::MultichainArbitrage,
                StrategyType::InfrastructureControl,
            ],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskTolerance {
    Conservative,   // Minimal risk, steady gains
    Moderate,       // Balanced risk/reward
    Balanced,       // Equal risk distribution
    Aggressive,     // High risk, high reward
    VeryAggressive, // Maximum risk tolerance
    Extreme,        // Unlimited risk for maximum gains
    Unlimited,      // No risk constraints
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum StrategyType {
    // Nano Strategies
    MicroArbitrage,
    GasOptimizedSniping,
    LowRiskCopyTrading,
    
    // Micro Strategies  
    FastArbitrage,
    TokenSniping,
    LiquidityFrontrunning,
    SmallMEV,
    
    // Small Strategies
    DiversifiedArbitrage,
    MediumMEV,
    PairTrading,
    TrendFollowing,
    LiquidityProvision,
    
    // Medium Strategies
    AdvancedMEV,
    MarketMaking,
    CrossDEXArbitrage,
    LeveragedStrategies,
    VolatilityTrading,
    
    // Large Strategies
    InstitutionalMEV,
    LiquidityAggregation,
    DerivativesArbitrage,
    YieldFarming,
    StructuredProducts,
    
    // Whale Strategies
    MarketManipulation,
    EcosystemInfluence,
    MegaLiquidityProvision,
    TokenLaunching,
    GovernanceCapture,
    
    // Titan Strategies
    CrossChainDominance,
    ProtocolOwnership,
    EcosystemCreation,
    MultichainArbitrage,
    InfrastructureControl,
}

/// **CAPITAL ALLOCATION MATRIX**
/// Sophisticated capital distribution across strategies
#[derive(Debug, Clone)]
pub struct CapitalAllocationMatrix {
    pub tier: CapitalTier,
    pub total_capital: f64,
    pub allocations: HashMap<StrategyType, StrategyAllocation>,
    pub risk_budget: RiskBudget,
    pub rebalancing_threshold: f64,
}

#[derive(Debug, Clone)]
pub struct StrategyAllocation {
    pub strategy_type: StrategyType,
    pub allocation_percentage: f64,
    pub absolute_amount: f64,
    pub max_position_size: f64,
    pub risk_weight: f64,
    pub expected_return: f64,
    pub volatility: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub correlation_with_others: f64,
}

#[derive(Debug, Clone)]
pub struct RiskBudget {
    pub max_portfolio_var: f64,           // Value at Risk
    pub max_expected_shortfall: f64,      // Expected Shortfall (CVaR)
    pub max_leverage: f64,                // Maximum leverage allowed
    pub diversification_requirement: f64,  // Minimum diversification score
    pub liquidity_requirement: f64,       // Minimum liquidity ratio
    pub correlation_limit: f64,           // Maximum strategy correlation
}

impl CapitalAllocationMatrix {
    pub fn new(total_capital: f64) -> Self {
        let tier = CapitalTier::from_balance(total_capital);
        let mut allocations = HashMap::new();
        
        // Get optimal allocations for this tier
        let optimal_allocations = Self::calculate_optimal_allocations(&tier, total_capital);
        
        for (strategy, allocation) in optimal_allocations {
            allocations.insert(strategy, allocation);
        }
        
        let risk_budget = Self::calculate_risk_budget(&tier, total_capital);
        
        info!("🏦 Created capital allocation matrix for {:?} tier", tier);
        info!("💰 Total capital: ${:.2}", total_capital);
        info!("📊 {} strategies allocated", allocations.len());
        
        Self {
            tier,
            total_capital,
            allocations,
            risk_budget,
            rebalancing_threshold: 0.05, // 5% deviation triggers rebalancing
        }
    }
    
    fn calculate_optimal_allocations(tier: &CapitalTier, capital: f64) -> HashMap<StrategyType, StrategyAllocation> {
        let mut allocations = HashMap::new();
        let strategies = tier.preferred_strategies();
        
        match tier {
            CapitalTier::Nano => {
                // Conservative allocation for minimal capital
                allocations.insert(StrategyType::MicroArbitrage, StrategyAllocation {
                    strategy_type: StrategyType::MicroArbitrage,
                    allocation_percentage: 0.50,
                    absolute_amount: capital * 0.50,
                    max_position_size: capital * 0.10,
                    risk_weight: 0.20,
                    expected_return: 0.15,
                    volatility: 0.08,
                    sharpe_ratio: 1.50,
                    max_drawdown: 0.05,
                    correlation_with_others: 0.30,
                });
                
                allocations.insert(StrategyType::GasOptimizedSniping, StrategyAllocation {
                    strategy_type: StrategyType::GasOptimizedSniping,
                    allocation_percentage: 0.30,
                    absolute_amount: capital * 0.30,
                    max_position_size: capital * 0.08,
                    risk_weight: 0.25,
                    expected_return: 0.20,
                    volatility: 0.12,
                    sharpe_ratio: 1.33,
                    max_drawdown: 0.08,
                    correlation_with_others: 0.25,
                });
                
                allocations.insert(StrategyType::LowRiskCopyTrading, StrategyAllocation {
                    strategy_type: StrategyType::LowRiskCopyTrading,
                    allocation_percentage: 0.20,
                    absolute_amount: capital * 0.20,
                    max_position_size: capital * 0.05,
                    risk_weight: 0.15,
                    expected_return: 0.12,
                    volatility: 0.06,
                    sharpe_ratio: 1.67,
                    max_drawdown: 0.03,
                    correlation_with_others: 0.20,
                });
            },
            
            CapitalTier::Whale => {
                // Aggressive allocation for whale capital
                allocations.insert(StrategyType::MarketManipulation, StrategyAllocation {
                    strategy_type: StrategyType::MarketManipulation,
                    allocation_percentage: 0.35,
                    absolute_amount: capital * 0.35,
                    max_position_size: capital * 0.15,
                    risk_weight: 0.40,
                    expected_return: 0.80,
                    volatility: 0.35,
                    sharpe_ratio: 1.85,
                    max_drawdown: 0.25,
                    correlation_with_others: 0.60,
                });
                
                allocations.insert(StrategyType::EcosystemInfluence, StrategyAllocation {
                    strategy_type: StrategyType::EcosystemInfluence,
                    allocation_percentage: 0.25,
                    absolute_amount: capital * 0.25,
                    max_position_size: capital * 0.12,
                    risk_weight: 0.35,
                    expected_return: 0.60,
                    volatility: 0.25,
                    sharpe_ratio: 1.92,
                    max_drawdown: 0.18,
                    correlation_with_others: 0.45,
                });
                
                allocations.insert(StrategyType::MegaLiquidityProvision, StrategyAllocation {
                    strategy_type: StrategyType::MegaLiquidityProvision,
                    allocation_percentage: 0.40,
                    absolute_amount: capital * 0.40,
                    max_position_size: capital * 0.20,
                    risk_weight: 0.25,
                    expected_return: 0.35,
                    volatility: 0.15,
                    sharpe_ratio: 1.75,
                    max_drawdown: 0.10,
                    correlation_with_others: 0.30,
                });
            },
            
            // Add more tier-specific allocations
            _ => {
                // Default balanced allocation
                let strategy_count = strategies.len() as f64;
                let base_allocation = 1.0 / strategy_count;
                
                for (i, strategy) in strategies.iter().enumerate() {
                    allocations.insert(strategy.clone(), StrategyAllocation {
                        strategy_type: strategy.clone(),
                        allocation_percentage: base_allocation,
                        absolute_amount: capital * base_allocation,
                        max_position_size: capital * tier.max_position_size_percentage(),
                        risk_weight: 0.30,
                        expected_return: 0.25 + (i as f64 * 0.05),
                        volatility: 0.15 + (i as f64 * 0.02),
                        sharpe_ratio: 1.40,
                        max_drawdown: 0.12,
                        correlation_with_others: 0.40,
                    });
                }
            }
        }
        
        allocations
    }
    
    fn calculate_risk_budget(tier: &CapitalTier, capital: f64) -> RiskBudget {
        match tier {
            CapitalTier::Nano => RiskBudget {
                max_portfolio_var: 0.02,    // 2% daily VaR
                max_expected_shortfall: 0.03,
                max_leverage: 1.0,          // No leverage
                diversification_requirement: 0.80,
                liquidity_requirement: 0.95,
                correlation_limit: 0.30,
            },
            CapitalTier::Micro => RiskBudget {
                max_portfolio_var: 0.03,
                max_expected_shortfall: 0.05,
                max_leverage: 1.5,
                diversification_requirement: 0.75,
                liquidity_requirement: 0.90,
                correlation_limit: 0.35,
            },
            CapitalTier::Whale => RiskBudget {
                max_portfolio_var: 0.15,    // 15% daily VaR
                max_expected_shortfall: 0.25,
                max_leverage: 10.0,         // High leverage allowed
                diversification_requirement: 0.40,
                liquidity_requirement: 0.60,
                correlation_limit: 0.70,
            },
            CapitalTier::Titan => RiskBudget {
                max_portfolio_var: 0.25,    // 25% daily VaR
                max_expected_shortfall: 0.40,
                max_leverage: 20.0,         // Extreme leverage
                diversification_requirement: 0.30,
                liquidity_requirement: 0.50,
                correlation_limit: 0.80,
            },
            _ => RiskBudget {
                max_portfolio_var: 0.08,
                max_expected_shortfall: 0.12,
                max_leverage: 3.0,
                diversification_requirement: 0.65,
                liquidity_requirement: 0.80,
                correlation_limit: 0.50,
            },
        }
    }
    
    pub fn get_strategy_allocation(&self, strategy: &StrategyType) -> Option<&StrategyAllocation> {
        self.allocations.get(strategy)
    }
    
    pub fn total_allocated_percentage(&self) -> f64 {
        self.allocations.values().map(|a| a.allocation_percentage).sum()
    }
    
    pub fn needs_rebalancing(&self, current_allocations: &HashMap<StrategyType, f64>) -> bool {
        for (strategy, target_allocation) in &self.allocations {
            if let Some(current) = current_allocations.get(strategy) {
                let deviation = (current - target_allocation.allocation_percentage).abs();
                if deviation > self.rebalancing_threshold {
                    return true;
                }
            }
        }
        false
    }
    
    pub fn calculate_portfolio_metrics(&self) -> PortfolioMetrics {
        let total_expected_return: f64 = self.allocations.values()
            .map(|a| a.allocation_percentage * a.expected_return)
            .sum();
            
        let total_volatility: f64 = self.allocations.values()
            .map(|a| a.allocation_percentage.powi(2) * a.volatility.powi(2))
            .sum::<f64>()
            .sqrt();
            
        let portfolio_sharpe = if total_volatility > 0.0 {
            total_expected_return / total_volatility
        } else {
            0.0
        };
        
        let max_drawdown: f64 = self.allocations.values()
            .map(|a| a.allocation_percentage * a.max_drawdown)
            .sum();
            
        PortfolioMetrics {
            expected_return: total_expected_return,
            volatility: total_volatility,
            sharpe_ratio: portfolio_sharpe,
            max_drawdown,
            diversification_score: self.calculate_diversification_score(),
            risk_adjusted_return: total_expected_return / max_drawdown.max(0.01),
        }
    }
    
    fn calculate_diversification_score(&self) -> f64 {
        let n = self.allocations.len() as f64;
        if n <= 1.0 { return 0.0; }
        
        let herfindahl_index: f64 = self.allocations.values()
            .map(|a| a.allocation_percentage.powi(2))
            .sum();
            
        (1.0 - herfindahl_index) / (1.0 - (1.0 / n))
    }
}

#[derive(Debug, Clone)]
pub struct PortfolioMetrics {
    pub expected_return: f64,
    pub volatility: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub diversification_score: f64,
    pub risk_adjusted_return: f64,
}

pub struct CapitalTierManager {
    pub current_tier: CapitalTier,
    pub allocation_matrix: CapitalAllocationMatrix,
    pub tier_history: Vec<(std::time::SystemTime, CapitalTier)>,
    pub rebalance_schedule: std::time::Duration,
}

impl CapitalTierManager {
    pub async fn new(initial_capital: f64) -> Result<Self> {
        let tier = CapitalTier::from_balance(initial_capital);
        let allocation_matrix = CapitalAllocationMatrix::new(initial_capital);
        
        info!("🎯 Initialized Capital Tier Manager");
        info!("  💼 Current tier: {:?}", tier);
        info!("  💰 Capital: ${:.2}", initial_capital);
        info!("  📋 Description: {}", tier.description());
        info!("  🎲 Risk tolerance: {:?}", tier.risk_tolerance());
        info!("  📊 Max position: {:.1}%", tier.max_position_size_percentage() * 100.0);
        info!("  🔄 Concurrent positions: {}", tier.concurrent_positions());
        
        Ok(Self {
            current_tier: tier,
            allocation_matrix,
            tier_history: vec![(std::time::SystemTime::now(), tier)],
            rebalance_schedule: std::time::Duration::from_hours(6),
        })
    }
    
    pub async fn update_capital(&mut self, new_capital: f64) -> Result<bool> {
        let new_tier = CapitalTier::from_balance(new_capital);
        let tier_changed = new_tier != self.current_tier;
        
        if tier_changed {
            info!("📈 CAPITAL TIER UPGRADE: {:?} → {:?}", self.current_tier, new_tier);
            info!("💰 Capital change: ${:.2} → ${:.2}", self.allocation_matrix.total_capital, new_capital);
            
            self.current_tier = new_tier.clone();
            self.allocation_matrix = CapitalAllocationMatrix::new(new_capital);
            self.tier_history.push((std::time::SystemTime::now(), new_tier));
            
            info!("🎯 New strategy focus: {}", new_tier.description());
            info!("📊 Updated allocations for {} strategies", self.allocation_matrix.allocations.len());
        } else {
            // Just update capital amount, keep same tier
            self.allocation_matrix.total_capital = new_capital;
        }
        
        Ok(tier_changed)
    }
    
    pub fn get_recommended_strategies(&self) -> Vec<StrategyType> {
        self.current_tier.preferred_strategies()
    }
    
    pub fn should_execute_strategy(&self, strategy: &StrategyType, position_size: f64) -> bool {
        // Check if strategy is allowed for this tier
        if !self.current_tier.preferred_strategies().contains(strategy) {
            return false;
        }
        
        // Check position size limits
        let max_position = self.allocation_matrix.total_capital * self.current_tier.max_position_size_percentage();
        if position_size > max_position {
            return false;
        }
        
        // Check if we have allocation for this strategy
        self.allocation_matrix.get_strategy_allocation(strategy).is_some()
    }
}