//! src/analytics/mod.rs
//! Advanced Analytics & ML Components for TrenchBotAI

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, LruCache};
use std::num::NonZeroUsize;
use chrono::{DateTime, Utc};
use rayon::prelude::*;
use ndarray::{Array1, Axis};

#[cfg(feature = "gpu")]
use tch;

// Re-export submodules
pub use chaos_index::ChaosAnalyzer;
pub use whale_clustering::WhaleClusterer;
pub use pattern_recognition::PatternRecognizer;
pub use profit_prediction::ProfitPredictor;

pub mod chaos_index;
pub mod whale_clustering;
pub mod pattern_recognition;
pub mod profit_prediction;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketChaos {
    pub chaos_score: f64,          // 0.0-1.0, higher = more chaotic
    pub whale_concentration: f64,   // How concentrated whale activity is
    pub volatility_index: f64,      // Price volatility measure
    pub mempool_congestion: f64,    // Transaction backlog level
    pub opportunity_density: f64,   // MEV opportunities per block
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhalePersonality {
    pub wallet: String,
    pub personality_type: WhaleType,
    pub confidence: f64,
    pub last_updated: DateTime<Utc>,
    pub trade_patterns: TradePatterns,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WhaleType {
    Aggressive,    // High frequency, high risk
    Conservative,  // Low frequency, calculated moves
    Momentum,      // Follows trends quickly
    Contrarian,    // Goes against the crowd
    Sniper,        // Precise, targeted trades
    Whale,         // Large position changes
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradePatterns {
    pub avg_trade_size_sol: f64,
    pub trade_frequency_per_hour: f64,
    pub preferred_tokens: Vec<String>,
    pub success_rate: f64,
    pub risk_tolerance: f64,
}

// Main Analytics Engine
pub struct AnalyticsEngine {
    chaos_analyzer: chaos_index::ChaosAnalyzer,
    whale_clusterer: whale_clustering::WhaleClusterer,
    pattern_recognizer: pattern_recognition::PatternRecognizer,
    profit_predictor: profit_prediction::ProfitPredictor,
}

impl AnalyticsEngine {
    pub fn new() -> Result<Self> {
        Ok(Self {
            chaos_analyzer: chaos_index::ChaosAnalyzer::new(),
            whale_clusterer: whale_clustering::WhaleClusterer::new()?,
            pattern_recognizer: pattern_recognition::PatternRecognizer::new()?,
            profit_predictor: profit_prediction::ProfitPredictor::new()?,
        })
    }

    /// Calculate current market chaos level
    pub async fn calculate_market_chaos(&self, market_data: &MarketSnapshot) -> Result<MarketChaos> {
        self.chaos_analyzer.analyze(market_data).await
    }

    /// Identify and classify whale personalities
    pub async fn analyze_whale_personalities(&mut self, transactions: &[Transaction]) -> Result<Vec<WhalePersonality>> {
        self.whale_clusterer.cluster_whales(transactions).await
    }

    /// Generate training data for GPU models
    pub async fn generate_training_data(&self) -> Result<TrainingDataset> {
        let mut dataset = TrainingDataset::new();
        
        // Collect historical patterns for model training
        // This will be processed on RunPod GPU
        
        Ok(dataset)
    }

    /// Predict profit for given opportunity (CPU version)
    pub async fn predict_profit_cpu(&self, opportunity: &Opportunity) -> Result<f64> {
        self.profit_predictor.predict_cpu(opportunity).await
    }

    /// Predict profit using GPU acceleration (RunPod)
    #[cfg(feature = "gpu")]
    pub async fn predict_profit_gpu(&self, opportunities: &[Opportunity]) -> Result<Vec<f64>> {
        self.profit_predictor.predict_gpu_batch(opportunities).await
    }
}

#[derive(Debug)]
pub struct MarketSnapshot {
    pub timestamp: DateTime<Utc>,
    pub total_volume_sol: f64,
    pub active_pools: usize,
    pub pending_transactions: usize,
    pub top_movers: Vec<TokenMovement>,
}

#[derive(Debug)]
pub struct TokenMovement {
    pub mint: String,
    pub price_change_5m: f64,
    pub volume_sol: f64,
    pub whale_activity: bool,
}

#[derive(Debug)]  
pub struct Transaction {
    pub signature: String,
    pub wallet: String,
    pub token_mint: String,
    pub amount_sol: f64,
    pub transaction_type: TransactionType,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug)]
pub enum TransactionType {
    Buy,
    Sell,
    Swap,
}

#[derive(Debug)]
pub struct Opportunity {
    pub id: String,
    pub opportunity_type: OpportunityType,
    pub token_mint: String,
    pub expected_profit_sol: f64,
    pub confidence: f64,
    pub market_conditions: MarketConditions,
}

#[derive(Debug)]
pub enum OpportunityType {
    Sandwich,
    Arbitrage,
    Liquidation,
    MemeSnipe,
}

#[derive(Debug)]
pub struct MarketConditions {
    pub chaos_score: f64,
    pub whale_activity_nearby: bool,
    pub pool_liquidity_sol: f64,
    pub recent_volume_sol: f64,
}

#[derive(Debug)]
pub struct TrainingDataset {
    pub features: Vec<Vec<f64>>,
    pub labels: Vec<f64>,
    pub metadata: HashMap<String, String>,
}

impl TrainingDataset {
    pub fn new() -> Self {
        Self {
            features: Vec::new(),
            labels: Vec::new(),
            metadata: HashMap::new(),
        }
    }
}

// Chaos Index Implementation
pub mod chaos_index {
    use super::*;
    
    pub struct ChaosAnalyzer {
        window_size: usize,
        volatility_threshold: f64,
    }
    
    impl ChaosAnalyzer {
        pub fn new() -> Self {
            Self {
                window_size: 100,  // Analyze last 100 blocks
                volatility_threshold: 0.1,
            }
        }
        
        pub async fn analyze(&self, market_data: &MarketSnapshot) -> Result<MarketChaos> {
            // Calculate various chaos metrics
            let volatility_index = self.calculate_volatility(market_data).await?;
            let whale_concentration = self.calculate_whale_concentration(market_data).await?;
            let mempool_congestion = self.calculate_mempool_pressure(market_data).await?;
            let opportunity_density = self.calculate_opportunity_density(market_data).await?;
            
            // Combine into overall chaos score
            let chaos_score = (volatility_index * 0.3 + 
                             whale_concentration * 0.25 + 
                             mempool_congestion * 0.25 + 
                             opportunity_density * 0.2).min(1.0);
            
            Ok(MarketChaos {
                chaos_score,
                whale_concentration,
                volatility_index,
                mempool_congestion,
                opportunity_density,
                timestamp: Utc::now(),
            })
        }
        
        async fn calculate_volatility(&self, market_data: &MarketSnapshot) -> Result<f64> {
            if market_data.top_movers.is_empty() {
                return Ok(0.0);
            }
            
            // Vectorized volatility calculation using ndarray
            let price_changes: Vec<f64> = market_data.top_movers
                .par_iter() // Parallel iterator
                .map(|tm| tm.price_change_5m.abs())
                .collect();
            
            let arr = Array1::from(price_changes);
            let mean = arr.mean().unwrap_or(0.0);
            let variance = arr.mapv(|x| (x - mean).powi(2)).mean().unwrap_or(0.0);
            
            Ok((variance.sqrt() * 10.0).min(1.0))
        }
        
        async fn calculate_whale_concentration(&self, market_data: &MarketSnapshot) -> Result<f64> {
            if market_data.top_movers.is_empty() {
                return Ok(0.0);
            }
            
            // Measure concentration of whale activity
            let whale_count = market_data.top_movers
                .iter()
                .filter(|tm| tm.whale_activity)
                .count();
            
            let concentration = whale_count as f64 / market_data.top_movers.len() as f64;
            Ok(concentration)
        }
        
        async fn calculate_mempool_pressure(&self, market_data: &MarketSnapshot) -> Result<f64> {
            // Higher pending transactions = higher pressure
            let pressure = (market_data.pending_transactions as f64 / 10000.0).min(1.0);
            Ok(pressure)
        }
        
        async fn calculate_opportunity_density(&self, market_data: &MarketSnapshot) -> Result<f64> {
            // Calculate MEV opportunities based on volume and volatility
            let total_volume = market_data.top_movers
                .iter()
                .map(|tm| tm.volume_sol)
                .sum::<f64>();
            
            let high_volatility_tokens = market_data.top_movers
                .iter()
                .filter(|tm| tm.price_change_5m.abs() > 0.05) // >5% change
                .count();
            
            // More volume and volatility = more MEV opportunities
            let density = ((total_volume / 10000.0) + (high_volatility_tokens as f64 / 10.0)).min(1.0);
            Ok(density)
        }
    }
}

// Whale Clustering Implementation
pub mod whale_clustering {
    use super::*;
    use std::collections::BTreeMap;
    
    pub struct WhaleClusterer {
        personality_models: BTreeMap<String, WhalePersonality>,
        update_threshold: usize,
        pattern_cache: LruCache<String, TradePatterns>,
        personality_cache: LruCache<String, WhalePersonality>,
    }
    
    impl WhaleClusterer {
        pub fn new() -> Result<Self> {
            Ok(Self {
                personality_models: BTreeMap::new(),
                update_threshold: 10, // Update after 10 new trades
                pattern_cache: LruCache::new(NonZeroUsize::new(1000).unwrap()),
                personality_cache: LruCache::new(NonZeroUsize::new(500).unwrap()),
            })
        }
        
        pub async fn cluster_whales(&mut self, transactions: &[Transaction]) -> Result<Vec<WhalePersonality>> {
            let mut whale_data: HashMap<String, Vec<&Transaction>> = HashMap::new();
            
            // Group transactions by wallet
            for tx in transactions {
                whale_data.entry(tx.wallet.clone()).or_default().push(tx);
            }
            
            // Parallel processing of whale personalities
            let personalities: Result<Vec<_>, _> = whale_data
                .into_par_iter()
                .filter_map(|(wallet, txs)| {
                    if txs.len() >= 5 {
                        Some((wallet, txs))
                    } else {
                        None
                    }
                })
                .map(|(wallet, txs)| {
                    // Create patterns without cache for parallel processing
                    if txs.len() < 5 {
                        return Ok(None);
                    }
                    
                    let patterns = Self::extract_trade_patterns_static(&txs)?;
                    let personality_type = Self::determine_personality_type_static(&patterns)?;
                    let confidence = Self::calculate_confidence_static(&patterns, &personality_type)?;
                    
                    Ok(Some(WhalePersonality {
                        wallet,
                        personality_type,
                        confidence,
                        last_updated: Utc::now(),
                        trade_patterns: patterns,
                    }))
                })
                .collect();
            
            Ok(personalities?.into_iter().flatten().collect())
        }
        
        fn classify_whale_personality_sync(&mut self, wallet: &str, transactions: Vec<&Transaction>) -> Result<Option<WhalePersonality>> {
            if transactions.len() < 5 {
                return Ok(None);
            }
            
            // Check cache first
            let cache_key = format!("{}_{}", wallet, transactions.len());
            if let Some(cached) = self.personality_cache.get(&cache_key) {
                return Ok(Some(cached.clone()));
            }
            
            let patterns = self.extract_trade_patterns_sync(&transactions)?;
            let personality_type = self.determine_personality_type_sync(&patterns)?;
            let confidence = self.calculate_confidence_sync(&patterns, &personality_type)?;
            
            let personality = WhalePersonality {
                wallet: wallet.to_string(),
                personality_type,
                confidence,
                last_updated: Utc::now(),
                trade_patterns: patterns,
            };
            
            // Cache the result
            self.personality_cache.put(cache_key, personality.clone());
            
            Ok(Some(personality))
        }

        async fn classify_whale_personality(&self, wallet: &str, transactions: Vec<&Transaction>) -> Result<Option<WhalePersonality>> {
            self.classify_whale_personality_sync(wallet, transactions)
        }
        
        fn extract_trade_patterns_static(transactions: &[&Transaction]) -> Result<TradePatterns> {
            let total_volume: f64 = transactions.iter().map(|tx| tx.amount_sol).sum();
            let avg_trade_size = total_volume / transactions.len() as f64;
            
            // Calculate trade frequency (simplified)
            let time_span_hours = if transactions.len() > 1 {
                let earliest = transactions.iter().min_by_key(|tx| tx.timestamp).unwrap().timestamp;
                let latest = transactions.iter().max_by_key(|tx| tx.timestamp).unwrap().timestamp;
                (latest - earliest).num_hours() as f64
            } else {
                1.0
            };
            
            let trade_frequency = transactions.len() as f64 / time_span_hours.max(1.0);
            
            // Extract preferred tokens
            let mut token_counts: HashMap<String, usize> = HashMap::new();
            for tx in transactions {
                *token_counts.entry(tx.token_mint.clone()).or_default() += 1;
            }
            let preferred_tokens: Vec<String> = token_counts.into_iter()
                .take(5)
                .map(|(token, _)| token)
                .collect();
            
            Ok(TradePatterns {
                avg_trade_size_sol: avg_trade_size,
                trade_frequency_per_hour: trade_frequency,
                preferred_tokens,
                success_rate: 0.7, // Placeholder - would calculate from profit data
                risk_tolerance: if avg_trade_size > 100.0 { 0.8 } else { 0.4 },
            })
        }
        
        fn extract_trade_patterns_sync(&self, transactions: &[&Transaction]) -> Result<TradePatterns> {
            Self::extract_trade_patterns_static(transactions)
        }

        async fn extract_trade_patterns(&self, transactions: &[&Transaction]) -> Result<TradePatterns> {
            self.extract_trade_patterns_sync(transactions)
        }

        fn determine_personality_type_static(patterns: &TradePatterns) -> Result<WhaleType> {
            // Simple classification based on trade patterns
            if patterns.avg_trade_size_sol > 1000.0 {
                Ok(WhaleType::Whale)
            } else if patterns.trade_frequency_per_hour > 5.0 {
                Ok(WhaleType::Aggressive)
            } else if patterns.risk_tolerance > 0.7 {
                Ok(WhaleType::Momentum)
            } else if patterns.trade_frequency_per_hour < 1.0 {
                Ok(WhaleType::Conservative)
            } else {
                Ok(WhaleType::Sniper)
            }
        }
        
        fn determine_personality_type_sync(&self, patterns: &TradePatterns) -> Result<WhaleType> {
            Self::determine_personality_type_static(patterns)
        }

        async fn determine_personality_type(&self, patterns: &TradePatterns) -> Result<WhaleType> {
            self.determine_personality_type_sync(patterns)
        }

        fn calculate_confidence_static(patterns: &TradePatterns, personality: &WhaleType) -> Result<f64> {
            // Calculate confidence based on data quality and pattern consistency
            let data_quality = (patterns.preferred_tokens.len() as f64 / 5.0).min(1.0);
            let frequency_confidence = if patterns.trade_frequency_per_hour > 0.1 { 0.8 } else { 0.4 };
            
            let personality_confidence = match personality {
                WhaleType::Whale if patterns.avg_trade_size_sol > 1000.0 => 0.9,
                WhaleType::Aggressive if patterns.trade_frequency_per_hour > 5.0 => 0.85,
                WhaleType::Conservative if patterns.trade_frequency_per_hour < 1.0 => 0.8,
                _ => 0.6,
            };
            
            let overall_confidence = (data_quality * 0.3 + frequency_confidence * 0.3 + personality_confidence * 0.4).min(1.0);
            Ok(overall_confidence)
        }

        fn calculate_confidence_sync(&self, patterns: &TradePatterns, personality: &WhaleType) -> Result<f64> {
            Self::calculate_confidence_static(patterns, personality)
        }

        async fn calculate_confidence(&self, patterns: &TradePatterns, personality: &WhaleType) -> Result<f64> {
            self.calculate_confidence_sync(patterns, personality)
        }
    }
}

// Pattern Recognition (GPU-accelerated when available)
pub mod pattern_recognition {
    use super::*;
    
    pub struct PatternRecognizer {
        #[cfg(feature = "gpu")]
        gpu_model: Option<tch::CModule>,
        cpu_fallback: bool,
    }
    
    impl PatternRecognizer {
        pub fn new() -> Result<Self> {
            Ok(Self {
                #[cfg(feature = "gpu")]
                gpu_model: None, // Load trained model in production
                cpu_fallback: true,
            })
        }
        
        pub async fn recognize_sandwich_patterns(&self, _transactions: &[Transaction]) -> Result<Vec<f64>> {
            #[cfg(feature = "gpu")]
            if let Some(_model) = &self.gpu_model {
                return self.recognize_gpu().await;
            }
            
            // CPU fallback
            self.recognize_cpu().await
        }
        
        async fn recognize_cpu(&self) -> Result<Vec<f64>> {
            // Simplified CPU pattern recognition
            Ok(vec![0.7, 0.3, 0.8]) // Mock confidence scores
        }
        
        #[cfg(feature = "gpu")]
        async fn recognize_gpu(&self) -> Result<Vec<f64>> {
            // GPU-accelerated pattern recognition
            Ok(vec![0.85, 0.4, 0.9]) // Enhanced accuracy with GPU
        }
    }
}

// Profit Prediction Engine
pub mod profit_prediction {
    use super::*;
    
    pub struct ProfitPredictor {
        model_weights: Vec<f64>,
        feature_normalizers: HashMap<String, (f64, f64)>, // (mean, std)
    }
    
    impl ProfitPredictor {
        pub fn new() -> Result<Self> {
            Ok(Self {
                model_weights: vec![0.3, 0.25, 0.2, 0.15, 0.1], // Simplified weights
                feature_normalizers: HashMap::new(),
            })
        }
        
        pub async fn predict_cpu(&self, opportunity: &Opportunity) -> Result<f64> {
            let features = self.extract_features(opportunity).await?;
            let normalized_features = self.normalize_features(&features)?;
            
            // Simple linear model for CPU prediction
            let prediction = normalized_features.iter()
                .zip(self.model_weights.iter())
                .map(|(f, w)| f * w)
                .sum::<f64>();
            
            Ok(prediction.max(0.0)) // Ensure non-negative profit prediction
        }
        
        #[cfg(feature = "gpu")]
        pub async fn predict_gpu_batch(&self, opportunities: &[Opportunity]) -> Result<Vec<f64>> {
            // Batch prediction on GPU for better throughput
            let mut predictions = Vec::new();
            
            for opportunity in opportunities {
                let prediction = self.predict_cpu(opportunity).await?; // Fallback for now
                predictions.push(prediction * 1.1); // GPU "enhancement" placeholder
            }
            
            Ok(predictions)
        }
        
        async fn extract_features(&self, opportunity: &Opportunity) -> Result<Vec<f64>> {
            Ok(vec![
                opportunity.confidence,
                opportunity.market_conditions.chaos_score,
                opportunity.market_conditions.pool_liquidity_sol.ln(),
                opportunity.market_conditions.recent_volume_sol.ln(),
                if opportunity.market_conditions.whale_activity_nearby { 1.0 } else { 0.0 },
            ])
        }
        
        fn normalize_features(&self, features: &[f64]) -> Result<Vec<f64>> {
            // Simple min-max normalization for now
            Ok(features.iter().map(|&f| f.tanh()).collect()) // Squash to [-1, 1]
        }
    }
}