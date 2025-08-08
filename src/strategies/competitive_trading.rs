//! Competitive Trading Strategies
//! Ethical competitive features for market advantage

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use chrono::{DateTime, Utc, Duration};
use tokio::sync::RwLock;
use std::sync::Arc;
use crate::analytics::{WhalePersonality, Transaction, MarketSnapshot};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketLeader {
    pub wallet_address: String,
    pub success_rate: f64,
    pub avg_profit_per_trade: f64,
    pub trade_frequency: f64,
    pub preferred_tokens: Vec<String>,
    pub typical_trade_size: f64,
    pub confidence_score: f64,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreeStepPrediction {
    pub step1: PredictionStep,
    pub step2: PredictionStep, 
    pub step3: PredictionStep,
    pub overall_confidence: f64,
    pub predicted_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionStep {
    pub action_type: ActionType,
    pub target_token: String,
    pub expected_price_change: f64,
    pub confidence: f64,
    pub reasoning: String,
    pub expected_timing: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActionType {
    Buy,
    Sell,
    Hold,
    Accumulate,
    Distribute,
    RotateFrom(String), // Rotate from token X to current
    ArbitrageSetup,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompetitiveSignal {
    pub signal_type: SignalType,
    pub target_token: String,
    pub whale_wallet: String,
    pub urgency: f64, // 0.0-1.0
    pub expected_profit: f64,
    pub risk_level: f64,
    pub reasoning: String,
    pub valid_until: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SignalType {
    FollowLeader,
    ArbitrageOpportunity,
    CounterMove,
    DefensivePosition,
    AlphaCapture,
}

pub struct CompetitiveTradingEngine {
    market_leaders: Arc<RwLock<HashMap<String, MarketLeader>>>,
    whale_tracker: Arc<RwLock<HashMap<String, VecDeque<Transaction>>>>,
    prediction_cache: Arc<RwLock<HashMap<String, ThreeStepPrediction>>>,
    signal_history: Arc<RwLock<VecDeque<CompetitiveSignal>>>,
    min_leader_trades: usize,
    prediction_window: Duration,
}

impl CompetitiveTradingEngine {
    pub fn new() -> Self {
        Self {
            market_leaders: Arc::new(RwLock::new(HashMap::new())),
            whale_tracker: Arc::new(RwLock::new(HashMap::new())),
            prediction_cache: Arc::new(RwLock::new(HashMap::new())),
            signal_history: Arc::new(RwLock::new(VecDeque::new())),
            min_leader_trades: 20,
            prediction_window: Duration::minutes(30),
        }
    }

    /// Update market leaders based on recent performance
    pub async fn update_market_leaders(&self, transactions: &[Transaction]) -> Result<()> {
        let mut leaders = self.market_leaders.write().await;
        let mut tracker = self.whale_tracker.write().await;

        // Group transactions by wallet
        for tx in transactions {
            tracker.entry(tx.wallet.clone())
                .or_insert_with(VecDeque::new)
                .push_back(tx.clone());

            // Keep only recent transactions (last 100 per wallet)
            if let Some(wallet_txs) = tracker.get_mut(&tx.wallet) {
                while wallet_txs.len() > 100 {
                    wallet_txs.pop_front();
                }
            }
        }

        // Analyze each wallet for leadership potential
        for (wallet, wallet_txs) in tracker.iter() {
            if wallet_txs.len() >= self.min_leader_trades {
                if let Some(leader) = self.analyze_leadership_potential(wallet, wallet_txs).await? {
                    leaders.insert(wallet.clone(), leader);
                }
            }
        }

        // Remove stale leaders
        let cutoff = Utc::now() - Duration::hours(24);
        leaders.retain(|_, leader| leader.last_updated > cutoff);

        Ok(())
    }

    async fn analyze_leadership_potential(&self, wallet: &str, transactions: &VecDeque<Transaction>) -> Result<Option<MarketLeader>> {
        if transactions.len() < self.min_leader_trades {
            return Ok(None);
        }

        // Calculate success metrics (simplified - would need price data in production)
        let total_volume: f64 = transactions.iter().map(|tx| tx.amount_sol).sum();
        let avg_trade_size = total_volume / transactions.len() as f64;
        
        // Estimate success rate based on trade patterns
        let success_rate = self.estimate_success_rate(transactions).await?;
        let avg_profit = self.estimate_avg_profit(transactions).await?;
        
        let trade_frequency = self.calculate_trade_frequency(transactions).await?;
        let preferred_tokens = self.extract_preferred_tokens(transactions).await?;
        
        // Only promote to leader if metrics are strong
        if success_rate > 0.65 && avg_profit > 0.1 && trade_frequency > 0.5 {
            let confidence = (success_rate * 0.4 + (avg_profit.min(1.0)) * 0.4 + (trade_frequency.min(1.0)) * 0.2).min(1.0);
            
            Ok(Some(MarketLeader {
                wallet_address: wallet.to_string(),
                success_rate,
                avg_profit_per_trade: avg_profit,
                trade_frequency,
                preferred_tokens,
                typical_trade_size: avg_trade_size,
                confidence_score: confidence,
                last_updated: Utc::now(),
            }))
        } else {
            Ok(None)
        }
    }

    /// Generate three-steps-ahead prediction for a whale's next moves
    pub async fn predict_three_steps_ahead(&self, whale_wallet: &str, market_data: &MarketSnapshot) -> Result<Option<ThreeStepPrediction>> {
        let leaders = self.market_leaders.read().await;
        let tracker = self.whale_tracker.read().await;

        let leader = match leaders.get(whale_wallet) {
            Some(l) => l,
            None => return Ok(None), // Not a tracked leader
        };

        let recent_txs = match tracker.get(whale_wallet) {
            Some(txs) => txs,
            None => return Ok(None),
        };

        // Analyze recent pattern to predict next 3 moves
        let pattern_analysis = self.analyze_trading_pattern(recent_txs, market_data).await?;
        
        let step1 = self.predict_immediate_action(&pattern_analysis, leader, market_data).await?;
        let step2 = self.predict_follow_up_action(&step1, &pattern_analysis, leader, market_data).await?;
        let step3 = self.predict_final_action(&step1, &step2, &pattern_analysis, leader, market_data).await?;

        let overall_confidence = (step1.confidence * 0.5 + step2.confidence * 0.3 + step3.confidence * 0.2).min(1.0);

        Ok(Some(ThreeStepPrediction {
            step1,
            step2, 
            step3,
            overall_confidence,
            predicted_at: Utc::now(),
            expires_at: Utc::now() + self.prediction_window,
        }))
    }

    async fn predict_immediate_action(&self, pattern: &TradingPattern, leader: &MarketLeader, market_data: &MarketSnapshot) -> Result<PredictionStep> {
        // Predict the most likely immediate action based on:
        // 1. Recent trading pattern
        // 2. Current market conditions
        // 3. Whale's historical behavior
        
        let target_token = self.predict_target_token(pattern, leader, market_data).await?;
        
        let action_type = if pattern.recent_accumulation > 0.7 {
            ActionType::Buy
        } else if pattern.recent_distribution > 0.7 {
            ActionType::Sell
        } else if pattern.rotation_detected {
            ActionType::RotateFrom(pattern.rotation_from_token.clone().unwrap_or_default())
        } else {
            ActionType::Hold
        };

        let expected_price_impact = self.estimate_price_impact(&action_type, leader.typical_trade_size, &target_token, market_data).await?;
        
        Ok(PredictionStep {
            action_type,
            target_token,
            expected_price_change: expected_price_impact,
            confidence: 0.75, // High confidence for immediate prediction
            reasoning: format!("Based on recent {} pattern and market conditions", 
                if pattern.recent_accumulation > 0.7 { "accumulation" } 
                else if pattern.recent_distribution > 0.7 { "distribution" }
                else { "rotation" }
            ),
            expected_timing: Duration::minutes(5),
        })
    }

    async fn predict_follow_up_action(&self, step1: &PredictionStep, pattern: &TradingPattern, leader: &MarketLeader, market_data: &MarketSnapshot) -> Result<PredictionStep> {
        // Predict the follow-up action based on step1's expected outcome
        
        let action_type = match &step1.action_type {
            ActionType::Buy if step1.expected_price_change > 0.02 => {
                // If buy causes >2% pump, likely to accumulate more or hold
                if leader.typical_trade_size > 1000.0 {
                    ActionType::Accumulate
                } else {
                    ActionType::Hold
                }
            },
            ActionType::Sell if step1.expected_price_change < -0.02 => {
                // If sell causes >2% dump, might continue distributing
                ActionType::Distribute
            },
            ActionType::RotateFrom(_) => {
                // After rotation, usually accumulate the new token
                ActionType::Accumulate
            },
            _ => ActionType::Hold,
        };

        let target_token = if matches!(action_type, ActionType::Accumulate | ActionType::Hold) {
            step1.target_token.clone()
        } else {
            self.predict_next_rotation_target(leader, market_data).await?.unwrap_or(step1.target_token.clone())
        };

        Ok(PredictionStep {
            action_type,
            target_token,
            expected_price_change: step1.expected_price_change * 0.6, // Diminishing impact
            confidence: 0.55, // Lower confidence for 2nd step
            reasoning: format!("Follow-up to step1 {} on {}", 
                match step1.action_type { 
                    ActionType::Buy => "buy",
                    ActionType::Sell => "sell", 
                    _ => "action"
                }, 
                step1.target_token
            ),
            expected_timing: Duration::minutes(15),
        })
    }

    async fn predict_final_action(&self, step1: &PredictionStep, step2: &PredictionStep, pattern: &TradingPattern, leader: &MarketLeader, market_data: &MarketSnapshot) -> Result<PredictionStep> {
        // Predict the final action to complete the sequence
        
        let action_type = match (&step1.action_type, &step2.action_type) {
            (ActionType::Buy, ActionType::Accumulate) => {
                // After buy + accumulate, usually distribute at profit
                ActionType::Sell
            },
            (ActionType::Sell, ActionType::Distribute) => {
                // After sell + distribute, might rotate to new opportunity
                if let Some(next_token) = self.predict_next_rotation_target(leader, market_data).await? {
                    ActionType::RotateFrom(step2.target_token.clone())
                } else {
                    ActionType::Hold
                }
            },
            _ => ActionType::Hold,
        };

        let target_token = match &action_type {
            ActionType::RotateFrom(_) => {
                self.predict_next_rotation_target(leader, market_data).await?.unwrap_or(step2.target_token.clone())
            },
            _ => step2.target_token.clone(),
        };

        Ok(PredictionStep {
            action_type,
            target_token,
            expected_price_change: (step1.expected_price_change + step2.expected_price_change) * 0.3, // Combined but reduced impact
            confidence: 0.35, // Lowest confidence for 3rd step
            reasoning: "Predicted completion of trading sequence".to_string(),
            expected_timing: Duration::minutes(45),
        })
    }

    /// Generate competitive trading signals based on leader analysis
    pub async fn generate_competitive_signals(&self, market_data: &MarketSnapshot) -> Result<Vec<CompetitiveSignal>> {
        let leaders = self.market_leaders.read().await;
        let mut signals = Vec::new();

        for (wallet, leader) in leaders.iter() {
            // Only follow high-confidence leaders
            if leader.confidence_score < 0.7 {
                continue;
            }

            // Get prediction for this leader
            if let Some(prediction) = self.predict_three_steps_ahead(wallet, market_data).await? {
                if prediction.overall_confidence > 0.6 {
                    // Generate follow signal
                    let signal = CompetitiveSignal {
                        signal_type: SignalType::FollowLeader,
                        target_token: prediction.step1.target_token.clone(),
                        whale_wallet: wallet.clone(),
                        urgency: prediction.step1.confidence,
                        expected_profit: prediction.step1.expected_price_change.abs() * 0.7, // Capture 70% of whale's expected move
                        risk_level: 1.0 - prediction.overall_confidence,
                        reasoning: format!("Following leader {} - {}", wallet, prediction.step1.reasoning),
                        valid_until: Utc::now() + Duration::minutes(10),
                    };
                    signals.push(signal);
                }
            }

            // Check for arbitrage opportunities created by whale movements
            for token in &leader.preferred_tokens {
                if let Some(arb_signal) = self.detect_arbitrage_opportunity(token, leader, market_data).await? {
                    signals.push(arb_signal);
                }
            }
        }

        // Sort by urgency and expected profit
        signals.sort_by(|a, b| {
            let score_a = a.urgency * a.expected_profit;
            let score_b = b.urgency * b.expected_profit;
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(signals)
    }

    // Helper methods (simplified implementations)
    async fn estimate_success_rate(&self, transactions: &VecDeque<Transaction>) -> Result<f64> {
        // Simplified success rate calculation
        // In production, this would analyze actual profit/loss data
        let buy_sell_ratio = self.calculate_buy_sell_ratio(transactions).await?;
        Ok((buy_sell_ratio * 0.6 + 0.4).min(0.95)) // Base 40% + up to 55% based on ratio
    }

    async fn estimate_avg_profit(&self, transactions: &VecDeque<Transaction>) -> Result<f64> {
        // Simplified profit estimation
        // Would need price data and P&L calculation in production
        let volume_consistency = self.calculate_volume_consistency(transactions).await?;
        Ok(volume_consistency * 0.3) // 0-30% estimated profit based on consistency
    }

    async fn calculate_trade_frequency(&self, transactions: &VecDeque<Transaction>) -> Result<f64> {
        if transactions.len() < 2 {
            return Ok(0.0);
        }

        let time_span = transactions.back().unwrap().timestamp - transactions.front().unwrap().timestamp;
        let hours = time_span.num_hours() as f64;
        
        if hours > 0.0 {
            Ok(transactions.len() as f64 / hours)
        } else {
            Ok(0.0)
        }
    }

    async fn extract_preferred_tokens(&self, transactions: &VecDeque<Transaction>) -> Result<Vec<String>> {
        let mut token_counts: HashMap<String, usize> = HashMap::new();
        
        for tx in transactions {
            *token_counts.entry(tx.token_mint.clone()).or_default() += 1;
        }

        let mut tokens: Vec<_> = token_counts.into_iter().collect();
        tokens.sort_by(|a, b| b.1.cmp(&a.1));
        
        Ok(tokens.into_iter().take(5).map(|(token, _)| token).collect())
    }

    async fn calculate_buy_sell_ratio(&self, transactions: &VecDeque<Transaction>) -> Result<f64> {
        let (buys, sells) = transactions.iter().fold((0, 0), |(b, s), tx| {
            match tx.transaction_type {
                crate::analytics::TransactionType::Buy => (b + 1, s),
                crate::analytics::TransactionType::Sell => (b, s + 1),
                _ => (b, s),
            }
        });

        if buys + sells > 0 {
            Ok(buys as f64 / (buys + sells) as f64)
        } else {
            Ok(0.5)
        }
    }

    async fn calculate_volume_consistency(&self, transactions: &VecDeque<Transaction>) -> Result<f64> {
        if transactions.is_empty() {
            return Ok(0.0);
        }

        let volumes: Vec<f64> = transactions.iter().map(|tx| tx.amount_sol).collect();
        let mean = volumes.iter().sum::<f64>() / volumes.len() as f64;
        let variance = volumes.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / volumes.len() as f64;
        let std_dev = variance.sqrt();
        
        // Lower variance = higher consistency = higher score
        Ok((1.0 - (std_dev / mean.max(1.0))).max(0.0).min(1.0))
    }

    async fn analyze_trading_pattern(&self, transactions: &VecDeque<Transaction>, _market_data: &MarketSnapshot) -> Result<TradingPattern> {
        // Analyze recent trading pattern
        let recent_txs: Vec<_> = transactions.iter()
            .filter(|tx| tx.timestamp > Utc::now() - Duration::hours(6))
            .collect();

        let (recent_buys, recent_sells) = recent_txs.iter().fold((0, 0), |(b, s), tx| {
            match tx.transaction_type {
                crate::analytics::TransactionType::Buy => (b + 1, s),
                crate::analytics::TransactionType::Sell => (b, s + 1),
                _ => (b, s),
            }
        });

        let total_recent = recent_buys + recent_sells;
        let recent_accumulation = if total_recent > 0 { recent_buys as f64 / total_recent as f64 } else { 0.0 };
        let recent_distribution = if total_recent > 0 { recent_sells as f64 / total_recent as f64 } else { 0.0 };

        // Detect token rotation
        let (rotation_detected, rotation_from_token) = self.detect_rotation_pattern(&recent_txs).await?;

        Ok(TradingPattern {
            recent_accumulation,
            recent_distribution,
            rotation_detected,
            rotation_from_token,
            recent_volume: recent_txs.iter().map(|tx| tx.amount_sol).sum(),
        })
    }

    async fn detect_rotation_pattern(&self, transactions: &[&Transaction]) -> Result<(bool, Option<String>)> {
        // Simple rotation detection: sell of token A followed by buy of token B
        if transactions.len() < 2 {
            return Ok((false, None));
        }

        let mut last_sell_token = None;
        for tx in transactions.iter().rev().take(5) {
            match tx.transaction_type {
                crate::analytics::TransactionType::Sell => {
                    last_sell_token = Some(tx.token_mint.clone());
                },
                crate::analytics::TransactionType::Buy => {
                    if let Some(ref sell_token) = last_sell_token {
                        if *sell_token != tx.token_mint {
                            return Ok((true, Some(sell_token.clone())));
                        }
                    }
                },
                _ => {},
            }
        }

        Ok((false, None))
    }

    async fn predict_target_token(&self, pattern: &TradingPattern, leader: &MarketLeader, _market_data: &MarketSnapshot) -> Result<String> {
        // Predict which token the whale will target next
        if let Some(ref rotation_from) = pattern.rotation_from_token {
            // If rotating, return the preferred token that's not the rotation source
            for token in &leader.preferred_tokens {
                if token != rotation_from {
                    return Ok(token.clone());
                }
            }
        }

        // Default to most preferred token
        Ok(leader.preferred_tokens.first().cloned().unwrap_or_default())
    }

    async fn estimate_price_impact(&self, action_type: &ActionType, trade_size: f64, _token: &str, _market_data: &MarketSnapshot) -> Result<f64> {
        // Simplified price impact estimation
        let base_impact = (trade_size / 10000.0).min(0.1); // Max 10% impact
        
        match action_type {
            ActionType::Buy | ActionType::Accumulate => Ok(base_impact),
            ActionType::Sell | ActionType::Distribute => Ok(-base_impact),
            _ => Ok(0.0),
        }
    }

    async fn predict_next_rotation_target(&self, leader: &MarketLeader, _market_data: &MarketSnapshot) -> Result<Option<String>> {
        // Simple prediction: rotate to second most preferred token
        Ok(leader.preferred_tokens.get(1).cloned())
    }

    async fn detect_arbitrage_opportunity(&self, token: &str, leader: &MarketLeader, market_data: &MarketSnapshot) -> Result<Option<CompetitiveSignal>> {
        // Enhanced arbitrage detection
        let arbitrage_opportunity = self.scan_cross_dex_prices(token).await?;
        
        if let Some(arb) = arbitrage_opportunity {
            // Check if whale activity might affect the arbitrage
            let whale_impact = self.estimate_whale_arbitrage_impact(leader, token, &arb).await?;
            
            if arb.profit_potential > 0.005 && whale_impact.enhances_opportunity {
                return Ok(Some(CompetitiveSignal {
                    signal_type: SignalType::ArbitrageOpportunity,
                    target_token: token.to_string(),
                    whale_wallet: leader.wallet_address.clone(),
                    urgency: arb.urgency,
                    expected_profit: arb.profit_potential * whale_impact.profit_multiplier,
                    risk_level: arb.risk_level,
                    reasoning: format!("Arbitrage opportunity enhanced by whale {} activity", leader.wallet_address),
                    valid_until: Utc::now() + Duration::minutes(2), // Arbitrage windows are short
                }));
            }
        }
        
        Ok(None)
    }

    async fn scan_cross_dex_prices(&self, token: &str) -> Result<Option<ArbitrageOpportunity>> {
        // Simulate cross-DEX price scanning
        // In production, this would query Jupiter, Raydium, Orca, etc.
        
        let mock_prices = vec![
            DexPrice { dex: "Raydium".to_string(), price: 1.000, liquidity: 50000.0 },
            DexPrice { dex: "Orca".to_string(), price: 1.008, liquidity: 30000.0 },
            DexPrice { dex: "Jupiter".to_string(), price: 0.995, liquidity: 80000.0 },
        ];

        // Find best buy and sell prices
        let mut prices = mock_prices;
        prices.sort_by(|a, b| a.price.partial_cmp(&b.price).unwrap());
        
        let best_buy = &prices[0]; // Lowest price
        let best_sell = &prices[prices.len() - 1]; // Highest price
        
        let profit_potential = (best_sell.price - best_buy.price) / best_buy.price;
        
        if profit_potential > 0.003 { // Minimum 0.3% profit
            Ok(Some(ArbitrageOpportunity {
                token: token.to_string(),
                buy_dex: best_buy.dex.clone(),
                sell_dex: best_sell.dex.clone(),
                buy_price: best_buy.price,
                sell_price: best_sell.price,
                profit_potential,
                max_size: best_buy.liquidity.min(best_sell.liquidity),
                urgency: if profit_potential > 0.01 { 0.9 } else { 0.6 },
                risk_level: if profit_potential > 0.02 { 0.8 } else { 0.4 }, // Higher profit = higher risk
                detected_at: Utc::now(),
            }))
        } else {
            Ok(None)
        }
    }

    async fn estimate_whale_arbitrage_impact(&self, leader: &MarketLeader, token: &str, arb: &ArbitrageOpportunity) -> Result<WhaleArbitrageImpact> {
        // Determine if whale activity enhances or reduces arbitrage opportunity
        let enhances = leader.preferred_tokens.contains(&token.to_string()) && 
                      leader.typical_trade_size > 100.0;
        
        let profit_multiplier = if enhances {
            // Whale activity creates more price divergence
            1.2 + (leader.typical_trade_size / 10000.0).min(0.5)
        } else {
            1.0
        };

        Ok(WhaleArbitrageImpact {
            enhances_opportunity: enhances,
            profit_multiplier,
            additional_risk: if leader.typical_trade_size > 1000.0 { 0.2 } else { 0.0 },
        })
    }

    /// Smart order routing to avoid whale impact
    pub async fn route_order_smart(&self, token: &str, amount: f64, is_buy: bool, market_data: &MarketSnapshot) -> Result<SmartOrderRoute> {
        let leaders = self.market_leaders.read().await;
        
        // Check if any whales are likely to impact this token
        let whale_threats = self.assess_whale_threats(token, amount, is_buy, &leaders).await?;
        
        if whale_threats.high_risk_whales.is_empty() {
            // No whale threat - use normal routing
            return Ok(SmartOrderRoute {
                recommended_dex: "Jupiter".to_string(), // Default aggregator
                split_strategy: SplitStrategy::Single,
                delay_recommendation: Duration::seconds(0),
                risk_level: 0.2,
                reasoning: "No significant whale threats detected".to_string(),
            });
        }

        // Whale threats detected - implement defensive routing
        let route = if whale_threats.imminent_collision_risk > 0.7 {
            SmartOrderRoute {
                recommended_dex: "Private".to_string(), // Use private mempool
                split_strategy: SplitStrategy::TimeSplit { intervals: 3, delay: Duration::minutes(2) },
                delay_recommendation: Duration::minutes(1),
                risk_level: 0.8,
                reasoning: format!("High collision risk with {} whales - using defensive routing", 
                    whale_threats.high_risk_whales.len()),
            }
        } else if whale_threats.imminent_collision_risk > 0.4 {
            SmartOrderRoute {
                recommended_dex: "Jupiter".to_string(),
                split_strategy: SplitStrategy::SizeSplit { parts: 4 },
                delay_recommendation: Duration::seconds(30),
                risk_level: 0.5,
                reasoning: "Moderate whale collision risk - splitting order".to_string(),
            }
        } else {
            SmartOrderRoute {
                recommended_dex: "Raydium".to_string(), // Fast execution
                split_strategy: SplitStrategy::DexSplit { dexes: vec!["Raydium".to_string(), "Orca".to_string()] },
                delay_recommendation: Duration::seconds(0),
                risk_level: 0.3,
                reasoning: "Low whale risk - using multi-DEX routing".to_string(),
            }
        };

        Ok(route)
    }

    async fn assess_whale_threats(&self, token: &str, amount: f64, is_buy: bool, leaders: &HashMap<String, MarketLeader>) -> Result<WhaleThreatAssessment> {
        let mut high_risk_whales = Vec::new();
        let mut total_threat_score = 0.0;

        for (wallet, leader) in leaders.iter() {
            if leader.preferred_tokens.contains(&token.to_string()) {
                let threat_score = self.calculate_threat_score(leader, token, amount, is_buy).await?;
                
                if threat_score > 0.6 {
                    high_risk_whales.push(WhaleThreat {
                        wallet: wallet.clone(),
                        threat_score,
                        typical_size: leader.typical_trade_size,
                        likely_competing: self.predict_competition(leader, token, amount, is_buy).await?,
                    });
                }
                
                total_threat_score += threat_score;
            }
        }

        let imminent_collision_risk = (total_threat_score / leaders.len() as f64).min(1.0);

        Ok(WhaleThreatAssessment {
            high_risk_whales,
            imminent_collision_risk,
            total_threat_score,
        })
    }

    async fn calculate_threat_score(&self, leader: &MarketLeader, token: &str, amount: f64, is_buy: bool) -> Result<f64> {
        let mut threat_score = 0.0;

        // Size comparison threat
        let size_ratio = leader.typical_trade_size / amount.max(1.0);
        threat_score += if size_ratio > 10.0 { 0.8 } else { size_ratio / 10.0 };

        // Frequency threat (more active = higher threat)
        threat_score += (leader.trade_frequency / 10.0).min(0.3);

        // Token preference threat
        let token_preference = if leader.preferred_tokens[0] == token { 0.4 } else { 0.1 };
        threat_score += token_preference;

        // Success rate threat (better whales are more threatening)
        threat_score += leader.success_rate * 0.3;

        Ok(threat_score.min(1.0))
    }

    async fn predict_competition(&self, leader: &MarketLeader, token: &str, amount: f64, is_buy: bool) -> Result<bool> {
        // Predict if whale is likely to compete for the same opportunity
        let same_direction = if is_buy {
            leader.typical_trade_size > 0.0 // Assuming positive = net buying
        } else {
            leader.typical_trade_size < 0.0 // Negative = net selling  
        };

        let similar_size = (leader.typical_trade_size - amount).abs() / amount.max(1.0) < 2.0;
        let preferred_token = leader.preferred_tokens.contains(&token.to_string());

        Ok(same_direction && similar_size && preferred_token)
    }

    /// Defensive counter-trading mechanisms
    pub async fn activate_defense_mode(&self, attack_wallet: &str, token: &str) -> Result<DefenseStrategy> {
        let leaders = self.market_leaders.read().await;
        
        if let Some(attacker) = leaders.get(attack_wallet) {
            let defense = self.design_counter_strategy(attacker, token).await?;
            
            // Log the defensive action
            let mut signal_history = self.signal_history.write().await;
            signal_history.push_back(CompetitiveSignal {
                signal_type: SignalType::DefensivePosition,
                target_token: token.to_string(),
                whale_wallet: attack_wallet.to_string(),
                urgency: defense.urgency,
                expected_profit: 0.0, // Defense focuses on protection, not profit
                risk_level: defense.risk_level,
                reasoning: defense.reasoning.clone(),
                valid_until: Utc::now() + Duration::minutes(30),
            });

            Ok(defense)
        } else {
            Ok(DefenseStrategy {
                strategy_type: DefenseType::Monitor,
                urgency: 0.3,
                risk_level: 0.2,
                reasoning: "Unknown attacker - monitoring only".to_string(),
                actions: vec![DefenseAction::IncreaseMonitoring { target: attack_wallet.to_string() }],
            })
        }
    }

    async fn design_counter_strategy(&self, attacker: &MarketLeader, token: &str) -> Result<DefenseStrategy> {
        let mut actions = Vec::new();
        let mut urgency = 0.5;
        let mut risk_level = 0.4;

        // Analyze attacker's typical patterns
        if attacker.typical_trade_size > 1000.0 {
            // Large whale - use evasive tactics
            actions.push(DefenseAction::SplitOrders { max_size: attacker.typical_trade_size / 4.0 });
            actions.push(DefenseAction::UsePrivateMempool);
            urgency = 0.8;
            risk_level = 0.3;
        } else if attacker.trade_frequency > 5.0 {
            // High frequency attacker - use timing disruption
            actions.push(DefenseAction::RandomizeTimings { min_delay: Duration::seconds(10), max_delay: Duration::minutes(2) });
            actions.push(DefenseAction::UseDifferentDEXes);
            urgency = 0.7;
            risk_level = 0.4;
        } else {
            // Standard defensive measures
            actions.push(DefenseAction::MonitorClosely);
            actions.push(DefenseAction::AdjustSlippage { new_tolerance: 0.5 });
            urgency = 0.5;
            risk_level = 0.3;
        }

        // Always add monitoring
        actions.push(DefenseAction::IncreaseMonitoring { target: attacker.wallet_address.clone() });

        let strategy_type = if urgency > 0.7 {
            DefenseType::ActiveCountering
        } else if urgency > 0.5 {
            DefenseType::PassiveAvoidance  
        } else {
            DefenseType::Monitor
        };

        Ok(DefenseStrategy {
            strategy_type,
            urgency,
            risk_level,
            reasoning: format!("Defensive strategy against {} (size: {}, freq: {})", 
                attacker.wallet_address, attacker.typical_trade_size, attacker.trade_frequency),
            actions,
        })
    }

    /// Alpha capture for legitimate competitive advantage
    pub async fn capture_alpha_opportunity(&self, opportunity_type: AlphaType, market_data: &MarketSnapshot) -> Result<Option<CompetitiveSignal>> {
        match opportunity_type {
            AlphaType::NewTokenLaunch { token_mint } => {
                self.analyze_new_token_launch(&token_mint, market_data).await
            },
            AlphaType::NewsEvent { token_affected, impact_score } => {
                self.analyze_news_impact(&token_affected, impact_score, market_data).await
            },
            AlphaType::LiquidityEvent { token_mint, liquidity_change } => {
                self.analyze_liquidity_event(&token_mint, liquidity_change, market_data).await
            },
        }
    }

    async fn analyze_new_token_launch(&self, token_mint: &str, _market_data: &MarketSnapshot) -> Result<Option<CompetitiveSignal>> {
        // Analyze new token launches for early entry opportunities
        let launch_analysis = self.evaluate_launch_potential(token_mint).await?;
        
        if launch_analysis.potential_score > 0.7 {
            Ok(Some(CompetitiveSignal {
                signal_type: SignalType::AlphaCapture,
                target_token: token_mint.to_string(),
                whale_wallet: "system".to_string(),
                urgency: 0.9, // New launches require fast action
                expected_profit: launch_analysis.expected_return,
                risk_level: 0.8, // High risk/reward
                reasoning: format!("New token launch detected with high potential ({})", launch_analysis.potential_score),
                valid_until: Utc::now() + Duration::minutes(5), // Very short window
            }))
        } else {
            Ok(None)
        }
    }

    async fn analyze_news_impact(&self, token: &str, impact_score: f64, _market_data: &MarketSnapshot) -> Result<Option<CompetitiveSignal>> {
        if impact_score.abs() > 0.5 {
            Ok(Some(CompetitiveSignal {
                signal_type: SignalType::AlphaCapture,
                target_token: token.to_string(),
                whale_wallet: "news".to_string(),
                urgency: impact_score.abs(),
                expected_profit: impact_score.abs() * 0.6,
                risk_level: 0.6,
                reasoning: format!("News impact detected: {}", impact_score),
                valid_until: Utc::now() + Duration::minutes(15),
            }))
        } else {
            Ok(None)
        }
    }

    async fn analyze_liquidity_event(&self, token: &str, liquidity_change: f64, _market_data: &MarketSnapshot) -> Result<Option<CompetitiveSignal>> {
        if liquidity_change.abs() > 0.3 {
            Ok(Some(CompetitiveSignal {
                signal_type: SignalType::AlphaCapture,
                target_token: token.to_string(),
                whale_wallet: "liquidity".to_string(),
                urgency: 0.6,
                expected_profit: liquidity_change.abs() * 0.4,
                risk_level: 0.5,
                reasoning: format!("Significant liquidity change: {:.1}%", liquidity_change * 100.0),
                valid_until: Utc::now() + Duration::minutes(20),
            }))
        } else {
            Ok(None)
        }
    }

    async fn evaluate_launch_potential(&self, _token_mint: &str) -> Result<LaunchAnalysis> {
        // Simplified launch analysis
        // In production, would analyze tokenomics, team, marketing, etc.
        Ok(LaunchAnalysis {
            potential_score: 0.6, // Mock score
            expected_return: 0.15, // 15% expected return
            risk_factors: vec!["New project".to_string(), "Limited liquidity".to_string()],
        })
    }
}

#[derive(Debug)]
pub struct TradingPattern {
    recent_accumulation: f64,
    recent_distribution: f64,
    rotation_detected: bool,
    rotation_from_token: Option<String>,
    recent_volume: f64,
}

#[derive(Debug, Clone)]
struct ArbitrageOpportunity {
    token: String,
    buy_dex: String,
    sell_dex: String,
    buy_price: f64,
    sell_price: f64,
    profit_potential: f64,
    max_size: f64,
    urgency: f64,
    risk_level: f64,
    detected_at: DateTime<Utc>,
}

#[derive(Debug)]
struct DexPrice {
    dex: String,
    price: f64,
    liquidity: f64,
}

#[derive(Debug)]
struct WhaleArbitrageImpact {
    enhances_opportunity: bool,
    profit_multiplier: f64,
    additional_risk: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmartOrderRoute {
    pub recommended_dex: String,
    pub split_strategy: SplitStrategy,
    pub delay_recommendation: Duration,
    pub risk_level: f64,
    pub reasoning: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SplitStrategy {
    Single,
    SizeSplit { parts: usize },
    TimeSplit { intervals: usize, delay: Duration },
    DexSplit { dexes: Vec<String> },
}

#[derive(Debug)]
struct WhaleThreatAssessment {
    high_risk_whales: Vec<WhaleThreat>,
    imminent_collision_risk: f64,
    total_threat_score: f64,
}

#[derive(Debug)]
struct WhaleThreat {
    wallet: String,
    threat_score: f64,
    typical_size: f64,
    likely_competing: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DefenseStrategy {
    pub strategy_type: DefenseType,
    pub urgency: f64,
    pub risk_level: f64,
    pub reasoning: String,
    pub actions: Vec<DefenseAction>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DefenseType {
    Monitor,
    PassiveAvoidance,
    ActiveCountering,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DefenseAction {
    IncreaseMonitoring { target: String },
    SplitOrders { max_size: f64 },
    UsePrivateMempool,
    RandomizeTimings { min_delay: Duration, max_delay: Duration },
    UseDifferentDEXes,
    MonitorClosely,
    AdjustSlippage { new_tolerance: f64 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlphaType {
    NewTokenLaunch { token_mint: String },
    NewsEvent { token_affected: String, impact_score: f64 },
    LiquidityEvent { token_mint: String, liquidity_change: f64 },
}

#[derive(Debug)]
struct LaunchAnalysis {
    potential_score: f64,
    expected_return: f64,
    risk_factors: Vec<String>,
}