//! Causal Inference Engine for Predictive Supremacy
//! 
//! This module implements advanced causal discovery and inference techniques
//! to predict rug pull events before they happen by understanding causal
//! relationships in transaction patterns and market dynamics.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, BTreeMap, VecDeque};
use chrono::{DateTime, Utc, Duration};
use ndarray::{Array1, Array2, Array3, Axis};
use rayon::prelude::*;
use std::sync::{Arc, RwLock};

/// Causal variable in the rug pull prediction model
#[derive(Debug, Clone, Serialize)]
pub struct CausalVariable {
    pub name: String,
    pub variable_type: VariableType,
    pub current_value: f64,
    pub historical_values: VecDeque<TimeSeriesPoint>,
    pub causal_parents: Vec<String>,      // Variables that cause this one
    pub causal_children: Vec<String>,     // Variables this one causes
    pub intervention_effects: HashMap<String, f64>, // Effect sizes on other variables
    pub confounders: Vec<String>,         // Variables that confound relationships
}

#[derive(Debug, Clone, Serialize)]
pub enum VariableType {
    Continuous,    // Price, volume, etc.
    Binary,        // Rug pull occurred (0/1)
    Categorical,   // Transaction type
    Count,         // Number of transactions
    Latent,        // Hidden confounders
}

#[derive(Debug, Clone, Serialize)]  
pub struct TimeSeriesPoint {
    pub timestamp: DateTime<Utc>,
    pub value: f64,
    pub confidence: f64,
}

/// Causal graph edge representing causal relationship
#[derive(Debug, Clone)]
pub struct CausalEdge {
    pub from_variable: String,
    pub to_variable: String,
    pub causal_strength: f64,        // How strong the causal effect is
    pub time_lag_seconds: i64,       // How long the causal effect takes
    pub confidence: f64,             // How confident we are in this relationship
    pub mechanism: CausalMechanism,  // Type of causal mechanism
}

#[derive(Debug, Clone)]
pub enum CausalMechanism {
    DirectCause,        // A directly causes B
    MediatesCause,      // A causes B through C
    ConfounderEffect,   // A and B are both caused by C
    ColliderEffect,     // A and B both cause C
    InstrumentalVariable, // A affects B only through C
}

/// Structural Causal Model for rug pull prediction
pub struct StructuralCausalModel {
    pub variables: HashMap<String, CausalVariable>,
    pub causal_graph: Vec<CausalEdge>,
    pub structural_equations: HashMap<String, StructuralEquation>,
    pub intervention_history: Vec<InterventionExperiment>,
    pub counterfactual_cache: HashMap<String, CounterfactualResult>,
}

#[derive(Debug, Clone)]
pub struct StructuralEquation {
    pub target_variable: String,
    pub equation_type: EquationType,
    pub coefficients: Vec<f64>,
    pub parent_variables: Vec<String>,
    pub noise_variance: f64,
    pub nonlinear_terms: Vec<NonlinearTerm>,
}

#[derive(Debug, Clone)]
pub enum EquationType {
    Linear,
    Polynomial,
    Exponential,
    Logistic,
    NeuralNetwork,
}

#[derive(Debug, Clone)]
pub struct NonlinearTerm {
    pub term_type: NonlinearType,
    pub variables: Vec<String>,
    pub coefficient: f64,
    pub power: f64,
}

#[derive(Debug, Clone)]
pub enum NonlinearType {
    Interaction,  // X1 * X2
    Polynomial,   // X^n
    Exponential,  // exp(X)
    Logarithmic,  // log(X)
    Trigonometric, // sin(X), cos(X)
}

/// Causal Inference Engine
pub struct CausalInferenceEngine {
    pub causal_model: StructuralCausalModel,
    pub causal_discovery: CausalDiscoveryAlgorithm,
    pub prediction_horizon_hours: i64,
    pub intervention_simulator: InterventionSimulator,
    pub counterfactual_engine: CounterfactualEngine,
    pub learning_rate: f64,
    pub regularization_strength: f64,
}

impl CausalInferenceEngine {
    pub fn new(prediction_horizon_hours: i64) -> Result<Self> {
        let causal_model = StructuralCausalModel {
            variables: Self::initialize_causal_variables(),
            causal_graph: Vec::new(),
            structural_equations: HashMap::new(),
            intervention_history: Vec::new(),
            counterfactual_cache: HashMap::new(),
        };

        Ok(Self {
            causal_model,
            causal_discovery: CausalDiscoveryAlgorithm::new()?,
            prediction_horizon_hours,
            intervention_simulator: InterventionSimulator::new(),
            counterfactual_engine: CounterfactualEngine::new(),
            learning_rate: 0.01,
            regularization_strength: 0.001,
        })
    }

    /// Predict rug pull probability using causal inference
    pub async fn predict_rug_pull_probability(
        &mut self,
        transactions: &[crate::analytics::Transaction],
        market_data: Option<&crate::analytics::MarketSnapshot>,
    ) -> Result<CausalPredictionResult> {
        // Step 1: Update causal variables with new data
        self.update_causal_variables(transactions, market_data).await?;
        
        // Step 2: Discover/update causal relationships
        self.discover_causal_relationships().await?;
        
        // Step 3: Fit structural equations
        self.fit_structural_equations().await?;
        
        // Step 4: Make causal predictions
        let prediction = self.make_causal_prediction().await?;
        
        // Step 5: Generate counterfactual scenarios
        let counterfactuals = self.generate_counterfactual_scenarios().await?;
        
        // Step 6: Simulate interventions
        let intervention_effects = self.simulate_interventions().await?;
        
        Ok(CausalPredictionResult {
            rug_pull_probability: prediction.rug_pull_probability,
            confidence: prediction.confidence,
            prediction_horizon_hours: self.prediction_horizon_hours,
            causal_factors: prediction.causal_factors,
            counterfactual_scenarios: counterfactuals,
            recommended_interventions: intervention_effects,
            causal_graph_summary: self.summarize_causal_graph(),
            processing_time_ms: std::time::Instant::now().elapsed().as_millis() as u64,
            timestamp: Utc::now(),
        })
    }

    fn initialize_causal_variables() -> HashMap<String, CausalVariable> {
        let mut variables = HashMap::new();
        
        // Target variable
        variables.insert("rug_pull_event".to_string(), CausalVariable {
            name: "rug_pull_event".to_string(),
            variable_type: VariableType::Binary,
            current_value: 0.0,
            historical_values: VecDeque::new(),
            causal_parents: vec!["coordination_score".to_string(), "whale_concentration".to_string()],
            causal_children: vec!["token_price".to_string(), "liquidity".to_string()],
            intervention_effects: HashMap::new(),
            confounders: vec!["market_sentiment".to_string()],
        });

        // Coordination variables
        variables.insert("coordination_score".to_string(), CausalVariable {
            name: "coordination_score".to_string(),
            variable_type: VariableType::Continuous,
            current_value: 0.0,
            historical_values: VecDeque::new(),
            causal_parents: vec!["wallet_similarity".to_string(), "timing_synchrony".to_string()],
            causal_children: vec!["rug_pull_event".to_string()],
            intervention_effects: HashMap::new(),
            confounders: vec![],
        });

        variables.insert("wallet_similarity".to_string(), CausalVariable {
            name: "wallet_similarity".to_string(),
            variable_type: VariableType::Continuous,
            current_value: 0.0,
            historical_values: VecDeque::new(),
            causal_parents: vec!["whale_creation_time".to_string()],
            causal_children: vec!["coordination_score".to_string()],
            intervention_effects: HashMap::new(),
            confounders: vec![],
        });

        variables.insert("timing_synchrony".to_string(), CausalVariable {
            name: "timing_synchrony".to_string(),
            variable_type: VariableType::Continuous,
            current_value: 0.0,
            historical_values: VecDeque::new(),
            causal_parents: vec!["bot_coordination".to_string()],
            causal_children: vec!["coordination_score".to_string()],
            intervention_effects: HashMap::new(),
            confounders: vec![],
        });

        // Market variables
        variables.insert("token_price".to_string(), CausalVariable {
            name: "token_price".to_string(),
            variable_type: VariableType::Continuous,
            current_value: 0.0,
            historical_values: VecDeque::new(),
            causal_parents: vec!["buy_pressure".to_string(), "sell_pressure".to_string()],
            causal_children: vec!["fomo_buying".to_string()],
            intervention_effects: HashMap::new(),
            confounders: vec!["market_sentiment".to_string()],
        });

        variables.insert("liquidity".to_string(), CausalVariable {
            name: "liquidity".to_string(),
            variable_type: VariableType::Continuous,
            current_value: 0.0,
            historical_values: VecDeque::new(),
            causal_parents: vec!["initial_liquidity".to_string(), "liquidity_removal".to_string()],
            causal_children: vec!["price_volatility".to_string()],
            intervention_effects: HashMap::new(),
            confounders: vec![],
        });

        variables.insert("whale_concentration".to_string(), CausalVariable {
            name: "whale_concentration".to_string(),
            variable_type: VariableType::Continuous,
            current_value: 0.0,
            historical_values: VecDeque::new(),
            causal_parents: vec!["large_transactions".to_string()],
            causal_children: vec!["rug_pull_event".to_string(), "price_volatility".to_string()],
            intervention_effects: HashMap::new(),
            confounders: vec![],
        });

        // Hidden/Latent variables
        variables.insert("market_sentiment".to_string(), CausalVariable {
            name: "market_sentiment".to_string(),
            variable_type: VariableType::Latent,
            current_value: 0.0,
            historical_values: VecDeque::new(),
            causal_parents: vec!["social_media_buzz".to_string()],
            causal_children: vec!["token_price".to_string(), "rug_pull_event".to_string()],
            intervention_effects: HashMap::new(),
            confounders: vec![],
        });

        variables.insert("bot_coordination".to_string(), CausalVariable {
            name: "bot_coordination".to_string(),
            variable_type: VariableType::Binary,
            current_value: 0.0,
            historical_values: VecDeque::new(),
            causal_parents: vec![],
            causal_children: vec!["timing_synchrony".to_string(), "wallet_similarity".to_string()],
            intervention_effects: HashMap::new(),
            confounders: vec![],
        });

        variables
    }

    async fn update_causal_variables(
        &mut self,
        transactions: &[crate::analytics::Transaction],
        market_data: Option<&crate::analytics::MarketSnapshot>,
    ) -> Result<()> {
        let timestamp = Utc::now();

        // Calculate coordination score
        let coordination_score = self.calculate_coordination_score(transactions)?;
        self.update_variable("coordination_score", coordination_score, timestamp)?;

        // Calculate wallet similarity
        let wallet_similarity = self.calculate_wallet_similarity(transactions)?;
        self.update_variable("wallet_similarity", wallet_similarity, timestamp)?;

        // Calculate timing synchrony
        let timing_synchrony = self.calculate_timing_synchrony(transactions)?;
        self.update_variable("timing_synchrony", timing_synchrony, timestamp)?;

        // Calculate whale concentration
        let whale_concentration = self.calculate_whale_concentration(transactions)?;
        self.update_variable("whale_concentration", whale_concentration, timestamp)?;

        // Update market variables if available
        if let Some(market) = market_data {
            self.update_market_variables(market, timestamp)?;
        }

        // Infer latent variables
        self.infer_latent_variables(transactions)?;

        Ok(())
    }

    fn update_variable(&mut self, name: &str, value: f64, timestamp: DateTime<Utc>) -> Result<()> {
        if let Some(variable) = self.causal_model.variables.get_mut(name) {
            variable.current_value = value;
            variable.historical_values.push_back(TimeSeriesPoint {
                timestamp,
                value,
                confidence: 0.9, // Default confidence
            });

            // Keep only recent history (24 hours)
            let cutoff = timestamp - Duration::hours(24);
            while let Some(front) = variable.historical_values.front() {
                if front.timestamp < cutoff {
                    variable.historical_values.pop_front();
                } else {
                    break;
                }
            }
        }
        Ok(())
    }

    fn calculate_coordination_score(&self, transactions: &[crate::analytics::Transaction]) -> Result<f64> {
        if transactions.len() < 2 {
            return Ok(0.0);
        }

        // Group by wallet
        let mut wallet_groups: HashMap<String, Vec<&crate::analytics::Transaction>> = HashMap::new();
        for tx in transactions {
            wallet_groups.entry(tx.wallet.clone()).or_default().push(tx);
        }

        // Calculate coordination indicators
        let mut coordination_signals = Vec::new();

        // 1. Similar amounts (suspicious)
        let amounts: Vec<f64> = transactions.iter().map(|tx| tx.amount_sol).collect();
        let amount_variance = self.calculate_variance(&amounts);
        let amount_coordination = if amount_variance < 0.01 { 1.0 } else { (-amount_variance * 100.0).exp() };
        coordination_signals.push(amount_coordination);

        // 2. Temporal clustering
        let timestamps: Vec<i64> = transactions.iter().map(|tx| tx.timestamp.timestamp()).collect();
        let time_variance = self.calculate_variance(&timestamps.iter().map(|&t| t as f64).collect::<Vec<f64>>());
        let temporal_coordination = (-time_variance / 3600.0).exp(); // Normalize by hour
        coordination_signals.push(temporal_coordination);

        // 3. Wallet pattern similarity
        let unique_wallets = wallet_groups.len();
        let wallet_coordination = if unique_wallets < transactions.len() / 2 {
            1.0 - (unique_wallets as f64 / transactions.len() as f64)
        } else {
            0.0
        };
        coordination_signals.push(wallet_coordination);

        // Combine signals
        let coordination_score = coordination_signals.iter().sum::<f64>() / coordination_signals.len() as f64;
        Ok(coordination_score.clamp(0.0, 1.0))
    }

    fn calculate_variance(&self, values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / values.len() as f64;
        
        variance
    }

    fn calculate_wallet_similarity(&self, transactions: &[crate::analytics::Transaction]) -> Result<f64> {
        if transactions.len() < 2 {
            return Ok(0.0);
        }

        let wallets: Vec<&String> = transactions.iter().map(|tx| &tx.wallet).collect();
        let mut similarity_scores = Vec::new();

        for i in 0..wallets.len() {
            for j in (i + 1)..wallets.len() {
                let similarity = self.calculate_string_similarity(wallets[i], wallets[j]);
                similarity_scores.push(similarity);
            }
        }

        let avg_similarity = similarity_scores.iter().sum::<f64>() / similarity_scores.len() as f64;
        Ok(avg_similarity)
    }

    fn calculate_string_similarity(&self, s1: &str, s2: &str) -> f64 {
        // Simple Jaccard similarity for wallet addresses
        let chars1: std::collections::HashSet<char> = s1.chars().collect();
        let chars2: std::collections::HashSet<char> = s2.chars().collect();
        
        let intersection = chars1.intersection(&chars2).count();
        let union = chars1.union(&chars2).count();
        
        if union == 0 {
            0.0
        } else {
            intersection as f64 / union as f64
        }
    }

    fn calculate_timing_synchrony(&self, transactions: &[crate::analytics::Transaction]) -> Result<f64> {
        if transactions.len() < 2 {
            return Ok(0.0);
        }

        // Calculate time differences between consecutive transactions
        let mut sorted_txs = transactions.to_vec();
        sorted_txs.sort_by_key(|tx| tx.timestamp);

        let mut time_diffs = Vec::new();
        for i in 1..sorted_txs.len() {
            let diff = (sorted_txs[i].timestamp - sorted_txs[i-1].timestamp).num_seconds().abs();
            time_diffs.push(diff as f64);
        }

        // High synchrony = low variance in time differences
        let time_variance = self.calculate_variance(&time_diffs);
        let synchrony = if time_variance < 60.0 { // Within 1 minute
            1.0 - (time_variance / 3600.0) // Normalize by hour
        } else {
            (-time_variance / 3600.0).exp()
        };

        Ok(synchrony.clamp(0.0, 1.0))
    }

    fn calculate_whale_concentration(&self, transactions: &[crate::analytics::Transaction]) -> Result<f64> {
        if transactions.is_empty() {
            return Ok(0.0);
        }

        // Calculate total volume
        let total_volume: f64 = transactions.iter().map(|tx| tx.amount_sol).sum();
        
        // Find large transactions (whales)
        let whale_threshold = total_volume * 0.1; // 10% of total volume
        let whale_volume: f64 = transactions.iter()
            .filter(|tx| tx.amount_sol >= whale_threshold)
            .map(|tx| tx.amount_sol)
            .sum();

        let concentration = if total_volume > 0.0 {
            whale_volume / total_volume
        } else {
            0.0
        };

        Ok(concentration)
    }

    fn update_market_variables(&mut self, market_data: &crate::analytics::MarketSnapshot, timestamp: DateTime<Utc>) -> Result<()> {
        // Extract price and liquidity information from market data
        let avg_price_change = market_data.top_movers.iter()
            .map(|tm| tm.price_change_5m.abs())
            .sum::<f64>() / market_data.top_movers.len().max(1) as f64;

        self.update_variable("token_price", avg_price_change, timestamp)?;

        let total_volume = market_data.top_movers.iter()
            .map(|tm| tm.volume_sol)
            .sum::<f64>();
        
        self.update_variable("liquidity", total_volume.ln().max(0.0), timestamp)?;

        Ok(())
    }

    fn infer_latent_variables(&mut self, transactions: &[crate::analytics::Transaction]) -> Result<()> {
        let timestamp = Utc::now();

        // Infer bot coordination from transaction patterns
        let bot_coordination = if transactions.len() > 5 {
            let timing_precision = self.calculate_timing_precision(transactions);
            let amount_precision = self.calculate_amount_precision(transactions);
            
            // High precision suggests bot coordination
            if timing_precision > 0.8 && amount_precision > 0.8 {
                1.0
            } else {
                0.0
            }
        } else {
            0.0
        };

        self.update_variable("bot_coordination", bot_coordination, timestamp)?;

        // Infer market sentiment from transaction volume and patterns
        let volume_trend = self.calculate_volume_trend(transactions);
        let market_sentiment = volume_trend.tanh(); // Squash to [-1, 1]
        
        self.update_variable("market_sentiment", market_sentiment, timestamp)?;

        Ok(())
    }

    fn calculate_timing_precision(&self, transactions: &[crate::analytics::Transaction]) -> f64 {
        if transactions.len() < 2 {
            return 0.0;
        }

        // Check if transactions occur at regular intervals (bot-like)
        let mut sorted_txs = transactions.to_vec();
        sorted_txs.sort_by_key(|tx| tx.timestamp);

        let mut intervals = Vec::new();
        for i in 1..sorted_txs.len() {
            let interval = (sorted_txs[i].timestamp - sorted_txs[i-1].timestamp).num_seconds();
            intervals.push(interval as f64);
        }

        let interval_variance = self.calculate_variance(&intervals);
        let precision = (-interval_variance / 3600.0).exp(); // High precision = low variance
        
        precision.clamp(0.0, 1.0)
    }

    fn calculate_amount_precision(&self, transactions: &[crate::analytics::Transaction]) -> f64 {
        if transactions.is_empty() {
            return 0.0;
        }

        let amounts: Vec<f64> = transactions.iter().map(|tx| tx.amount_sol).collect();
        let amount_variance = self.calculate_variance(&amounts);
        
        // Check for round numbers (bot-like)
        let round_number_count = amounts.iter()
            .filter(|&&amount| amount.fract() == 0.0 || (amount * 10.0).fract() == 0.0)
            .count();
        
        let round_number_ratio = round_number_count as f64 / amounts.len() as f64;
        
        // Combine low variance with round numbers
        let precision = (-amount_variance / 1000.0).exp() * 0.7 + round_number_ratio * 0.3;
        
        precision.clamp(0.0, 1.0)
    }

    fn calculate_volume_trend(&self, transactions: &[crate::analytics::Transaction]) -> f64 {
        if transactions.len() < 2 {
            return 0.0;
        }

        let total_volume: f64 = transactions.iter().map(|tx| tx.amount_sol).sum();
        let time_span = if let (Some(first), Some(last)) = 
            (transactions.iter().min_by_key(|tx| tx.timestamp), 
             transactions.iter().max_by_key(|tx| tx.timestamp)) {
            (last.timestamp - first.timestamp).num_hours().max(1) as f64
        } else {
            1.0
        };

        // Volume per hour as trend indicator
        total_volume / time_span
    }

    async fn discover_causal_relationships(&mut self) -> Result<()> {
        // Simplified causal discovery using correlation and temporal precedence
        let variables = &self.causal_model.variables;
        let mut discovered_edges = Vec::new();

        // For each potential causal relationship
        for (var1_name, var1) in variables {
            for (var2_name, var2) in variables {
                if var1_name == var2_name {
                    continue;
                }

                // Check if there's enough data
                if var1.historical_values.len() < 5 || var2.historical_values.len() < 5 {
                    continue;
                }

                // Calculate causal strength using temporal correlation
                let causal_strength = self.calculate_causal_strength(var1, var2)?;
                
                if causal_strength > 0.3 { // Threshold for significant causation
                    let edge = CausalEdge {
                        from_variable: var1_name.clone(),
                        to_variable: var2_name.clone(),
                        causal_strength,
                        time_lag_seconds: self.estimate_time_lag(var1, var2),
                        confidence: self.calculate_causal_confidence(var1, var2),
                        mechanism: self.identify_causal_mechanism(var1_name, var2_name),
                    };
                    
                    discovered_edges.push(edge);
                }
            }
        }

        // Update causal graph
        self.causal_model.causal_graph = discovered_edges;
        
        Ok(())
    }

    fn calculate_causal_strength(&self, var1: &CausalVariable, var2: &CausalVariable) -> Result<f64> {
        // Extract aligned time series
        let (series1, series2) = self.align_time_series(&var1.historical_values, &var2.historical_values);
        
        if series1.len() < 3 {
            return Ok(0.0);
        }

        // Calculate cross-correlation with different lags
        let mut max_correlation = 0.0;
        
        for lag in 0..3 { // Check lags up to 3 time points
            let correlation = self.calculate_lagged_correlation(&series1, &series2, lag);
            max_correlation = max_correlation.max(correlation.abs());
        }

        // Apply temporal precedence constraint
        let temporal_precedence = self.check_temporal_precedence(&var1.historical_values, &var2.historical_values);
        
        let causal_strength = max_correlation * temporal_precedence;
        Ok(causal_strength)
    }

    fn align_time_series(&self, series1: &VecDeque<TimeSeriesPoint>, series2: &VecDeque<TimeSeriesPoint>) -> (Vec<f64>, Vec<f64>) {
        let mut aligned1 = Vec::new();
        let mut aligned2 = Vec::new();
        
        // Simple alignment by matching timestamps (within 1 minute tolerance)
        for point1 in series1 {
            for point2 in series2 {
                if (point1.timestamp - point2.timestamp).num_seconds().abs() <= 60 {
                    aligned1.push(point1.value);
                    aligned2.push(point2.value);
                    break;
                }
            }
        }
        
        (aligned1, aligned2)
    }

    fn calculate_lagged_correlation(&self, series1: &[f64], series2: &[f64], lag: usize) -> f64 {
        if series1.len() <= lag || series2.len() <= lag {
            return 0.0;
        }

        let n = series1.len() - lag;
        if n < 2 {
            return 0.0;
        }

        // Calculate correlation between series1[0..n] and series2[lag..lag+n]
        let x: Vec<f64> = series1[0..n].to_vec();
        let y: Vec<f64> = series2[lag..lag+n].to_vec();
        
        let mean_x = x.iter().sum::<f64>() / n as f64;
        let mean_y = y.iter().sum::<f64>() / n as f64;
        
        let numerator: f64 = x.iter().zip(y.iter())
            .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y))
            .sum();
        
        let var_x: f64 = x.iter().map(|&xi| (xi - mean_x).powi(2)).sum();
        let var_y: f64 = y.iter().map(|&yi| (yi - mean_y).powi(2)).sum();
        
        if var_x == 0.0 || var_y == 0.0 {
            return 0.0;
        }
        
        numerator / (var_x * var_y).sqrt()
    }

    fn check_temporal_precedence(&self, series1: &VecDeque<TimeSeriesPoint>, series2: &VecDeque<TimeSeriesPoint>) -> f64 {
        // Check if changes in var1 consistently precede changes in var2
        let mut precedence_count = 0;
        let mut total_count = 0;
        
        for point1 in series1 {
            for point2 in series2 {
                let time_diff = (point2.timestamp - point1.timestamp).num_seconds();
                if time_diff > 0 && time_diff <= 3600 { // var1 precedes var2 within 1 hour
                    precedence_count += 1;
                }
                total_count += 1;
            }
        }
        
        if total_count == 0 {
            0.0
        } else {
            precedence_count as f64 / total_count as f64
        }
    }

    fn estimate_time_lag(&self, var1: &CausalVariable, var2: &CausalVariable) -> i64 {
        // Estimate average time lag between cause and effect
        let mut lags = Vec::new();
        
        for point1 in &var1.historical_values {
            for point2 in &var2.historical_values {
                let lag = (point2.timestamp - point1.timestamp).num_seconds();
                if lag > 0 && lag <= 3600 { // Within reasonable range
                    lags.push(lag);
                }
            }
        }
        
        if lags.is_empty() {
            300 // Default 5 minutes
        } else {
            lags.iter().sum::<i64>() / lags.len() as i64
        }
    }

    fn calculate_causal_confidence(&self, var1: &CausalVariable, var2: &CausalVariable) -> f64 {
        // Confidence based on data quality and consistency
        let data_quality = ((var1.historical_values.len() + var2.historical_values.len()) as f64 / 20.0).min(1.0);
        let avg_confidence = (var1.historical_values.iter().map(|p| p.confidence).sum::<f64>() +
                            var2.historical_values.iter().map(|p| p.confidence).sum::<f64>()) /
                           (var1.historical_values.len() + var2.historical_values.len()) as f64;
        
        (data_quality + avg_confidence) / 2.0
    }

    fn identify_causal_mechanism(&self, var1_name: &str, var2_name: &str) -> CausalMechanism {
        // Simple rule-based mechanism identification
        match (var1_name, var2_name) {
            ("wallet_similarity", "coordination_score") => CausalMechanism::DirectCause,
            ("timing_synchrony", "coordination_score") => CausalMechanism::DirectCause,
            ("coordination_score", "rug_pull_event") => CausalMechanism::DirectCause,
            ("market_sentiment", _) => CausalMechanism::ConfounderEffect,
            ("bot_coordination", "timing_synchrony") => CausalMechanism::DirectCause,
            _ => CausalMechanism::MediatesCause,
        }
    }

    async fn fit_structural_equations(&mut self) -> Result<()> {
        // Fit structural equations for each variable based on its causal parents
        for (var_name, variable) in &self.causal_model.variables {
            if variable.causal_parents.is_empty() {
                continue;
            }

            let equation = self.fit_equation_for_variable(var_name, variable).await?;
            self.causal_model.structural_equations.insert(var_name.clone(), equation);
        }
        
        Ok(())
    }

    async fn fit_equation_for_variable(&self, var_name: &str, variable: &CausalVariable) -> Result<StructuralEquation> {
        // Collect data for regression
        let mut feature_matrix = Vec::new();
        let mut target_values = Vec::new();
        
        // Get parent variable data
        for point in &variable.historical_values {
            let mut features = Vec::new();
            let mut all_parents_have_data = true;
            
            for parent_name in &variable.causal_parents {
                if let Some(parent_var) = self.causal_model.variables.get(parent_name) {
                    // Find parent value at similar timestamp
                    if let Some(parent_point) = self.find_closest_point(&parent_var.historical_values, point.timestamp) {
                        features.push(parent_point.value);
                    } else {
                        all_parents_have_data = false;
                        break;
                    }
                } else {
                    all_parents_have_data = false;
                    break;
                }
            }
            
            if all_parents_have_data && !features.is_empty() {
                feature_matrix.push(features);
                target_values.push(point.value);
            }
        }
        
        // Fit linear regression (simplified)
        let coefficients = if feature_matrix.len() >= 3 {
            self.fit_linear_regression(&feature_matrix, &target_values)?
        } else {
            vec![0.0; variable.causal_parents.len() + 1] // +1 for intercept
        };
        
        Ok(StructuralEquation {
            target_variable: var_name.to_string(),
            equation_type: EquationType::Linear,
            coefficients,
            parent_variables: variable.causal_parents.clone(),
            noise_variance: 0.1, // Default noise
            nonlinear_terms: Vec::new(),
        })
    }

    fn find_closest_point(&self, points: &VecDeque<TimeSeriesPoint>, target_time: DateTime<Utc>) -> Option<&TimeSeriesPoint> {
        points.iter()
            .min_by_key(|point| (point.timestamp - target_time).num_seconds().abs())
    }

    fn fit_linear_regression(&self, features: &[Vec<f64>], targets: &[f64]) -> Result<Vec<f64>> {
        if features.is_empty() || targets.is_empty() || features.len() != targets.len() {
            return Ok(vec![0.0; features[0].len() + 1]);
        }

        let n = features.len();
        let p = features[0].len() + 1; // +1 for intercept
        
        // Create design matrix with intercept
        let mut x_matrix = Array2::zeros((n, p));
        for i in 0..n {
            x_matrix[[i, 0]] = 1.0; // Intercept
            for j in 0..features[i].len() {
                x_matrix[[i, j + 1]] = features[i][j];
            }
        }
        
        let y_vector = Array1::from_vec(targets.to_vec());
        
        // Solve normal equations: (X'X + λI)β = X'y
        let xt = x_matrix.t();
        let xtx = xt.dot(&x_matrix);
        let xty = xt.dot(&y_vector);
        
        // Add regularization
        let mut xtx_reg = xtx;
        for i in 0..p {
            xtx_reg[[i, i]] += self.regularization_strength;
        }
        
        // Simplified solution (would use proper matrix inverse in production)
        let mut coefficients = vec![0.0; p];
        
        // Simple gradient descent for demonstration
        for _ in 0..100 {
            let predictions = x_matrix.dot(&Array1::from_vec(coefficients.clone()));
            let errors = &predictions - &y_vector;
            let gradients = xt.dot(&errors) / n as f64;
            
            for i in 0..p {
                coefficients[i] -= self.learning_rate * gradients[i];
            }
        }
        
        Ok(coefficients)
    }

    async fn make_causal_prediction(&self) -> Result<CausalPrediction> {
        // Use structural equations to predict rug pull probability
        let mut predictions = HashMap::new();
        
        // Forward pass through causal graph
        for variable_name in self.get_causal_ordering() {
            let prediction = self.predict_variable_value(&variable_name).await?;
            predictions.insert(variable_name, prediction);
        }
        
        // Extract rug pull prediction
        let rug_pull_probability = predictions.get("rug_pull_event").unwrap_or(&0.0);
        
        // Calculate prediction confidence
        let confidence = self.calculate_prediction_confidence(&predictions)?;
        
        // Identify key causal factors
        let causal_factors = self.identify_key_causal_factors(&predictions)?;
        
        Ok(CausalPrediction {
            rug_pull_probability: *rug_pull_probability,
            confidence,
            causal_factors,
        })
    }

    fn get_causal_ordering(&self) -> Vec<String> {
        // Topological sort of causal graph (simplified)
        vec![
            "bot_coordination".to_string(),
            "market_sentiment".to_string(),
            "wallet_similarity".to_string(),
            "timing_synchrony".to_string(),
            "coordination_score".to_string(),
            "whale_concentration".to_string(),
            "token_price".to_string(),
            "liquidity".to_string(),
            "rug_pull_event".to_string(),
        ]
    }

    async fn predict_variable_value(&self, variable_name: &str) -> Result<f64> {
        if let Some(equation) = self.causal_model.structural_equations.get(variable_name) {
            // Use structural equation to predict
            let mut prediction = equation.coefficients[0]; // Intercept
            
            for (i, parent_name) in equation.parent_variables.iter().enumerate() {
                if let Some(parent_var) = self.causal_model.variables.get(parent_name) {
                    prediction += equation.coefficients[i + 1] * parent_var.current_value;
                }
            }
            
            // Apply activation function based on variable type
            if let Some(variable) = self.causal_model.variables.get(variable_name) {
                match variable.variable_type {
                    VariableType::Binary => Ok(prediction.tanh().abs()), // Sigmoid-like
                    VariableType::Continuous => Ok(prediction),
                    _ => Ok(prediction.clamp(0.0, 1.0)),
                }
            } else {
                Ok(prediction)
            }
        } else {
            // Use current value if no equation available
            if let Some(variable) = self.causal_model.variables.get(variable_name) {
                Ok(variable.current_value)
            } else {
                Ok(0.0)
            }
        }
    }

    fn calculate_prediction_confidence(&self, predictions: &HashMap<String, f64>) -> Result<f64> {
        // Confidence based on data quality and model fit
        let mut confidence_factors = Vec::new();
        
        // Data quality factor
        let avg_data_points = self.causal_model.variables.values()
            .map(|v| v.historical_values.len())
            .sum::<usize>() as f64 / self.causal_model.variables.len() as f64;
        
        let data_quality = (avg_data_points / 20.0).min(1.0);
        confidence_factors.push(data_quality);
        
        // Causal graph completeness
        let graph_completeness = self.causal_model.causal_graph.len() as f64 / 
                               (self.causal_model.variables.len() * 2) as f64;
        confidence_factors.push(graph_completeness);
        
        // Prediction consistency
        let consistency = self.calculate_prediction_consistency(predictions);
        confidence_factors.push(consistency);
        
        let overall_confidence = confidence_factors.iter().sum::<f64>() / confidence_factors.len() as f64;
        Ok(overall_confidence.clamp(0.0, 1.0))
    }

    fn calculate_prediction_consistency(&self, predictions: &HashMap<String, f64>) -> f64 {
        // Check if predictions are consistent with causal relationships
        let mut consistency_scores = Vec::new();
        
        for edge in &self.causal_model.causal_graph {
            if let (Some(&cause_value), Some(&effect_value)) = 
                (predictions.get(&edge.from_variable), predictions.get(&edge.to_variable)) {
                
                // Expected effect direction
                let expected_correlation = if edge.causal_strength > 0.0 { 1.0 } else { -1.0 };
                let actual_correlation = cause_value * effect_value;
                
                let consistency = if expected_correlation * actual_correlation > 0.0 { 1.0 } else { 0.0 };
                consistency_scores.push(consistency);
            }
        }
        
        if consistency_scores.is_empty() {
            0.5 // Default moderate consistency
        } else {
            consistency_scores.iter().sum::<f64>() / consistency_scores.len() as f64
        }
    }

    fn identify_key_causal_factors(&self, predictions: &HashMap<String, f64>) -> Result<Vec<CausalFactor>> {
        let mut factors = Vec::new();
        
        // Find variables with highest causal influence on rug pull event
        for edge in &self.causal_model.causal_graph {
            if edge.to_variable == "rug_pull_event" {
                if let Some(&factor_value) = predictions.get(&edge.from_variable) {
                    factors.push(CausalFactor {
                        variable_name: edge.from_variable.clone(),
                        current_value: factor_value,
                        causal_strength: edge.causal_strength,
                        contribution_to_risk: edge.causal_strength * factor_value,
                        time_to_effect_seconds: edge.time_lag_seconds,
                    });
                }
            }
        }
        
        // Sort by contribution to risk
        factors.sort_by(|a, b| b.contribution_to_risk.partial_cmp(&a.contribution_to_risk).unwrap());
        
        Ok(factors)
    }

    async fn generate_counterfactual_scenarios(&self) -> Result<Vec<CounterfactualScenario>> {
        let mut scenarios = Vec::new();
        
        // Generate "what if" scenarios by intervening on key variables
        let key_variables = vec!["coordination_score", "whale_concentration", "liquidity"];
        
        for var_name in key_variables {
            // Scenario 1: Variable is much higher
            let high_scenario = self.simulate_counterfactual(var_name, 0.9).await?;
            scenarios.push(CounterfactualScenario {
                scenario_name: format!("High {}", var_name),
                intervention_variable: var_name.to_string(),
                intervention_value: 0.9,
                predicted_rug_pull_probability: high_scenario,
                probability_change: high_scenario - self.causal_model.variables.get("rug_pull_event")
                    .map(|v| v.current_value).unwrap_or(0.0),
            });
            
            // Scenario 2: Variable is much lower
            let low_scenario = self.simulate_counterfactual(var_name, 0.1).await?;
            scenarios.push(CounterfactualScenario {
                scenario_name: format!("Low {}", var_name),
                intervention_variable: var_name.to_string(),
                intervention_value: 0.1,
                predicted_rug_pull_probability: low_scenario,
                probability_change: low_scenario - self.causal_model.variables.get("rug_pull_event")
                    .map(|v| v.current_value).unwrap_or(0.0),
            });
        }
        
        Ok(scenarios)
    }

    async fn simulate_counterfactual(&self, intervention_var: &str, intervention_value: f64) -> Result<f64> {
        // Create modified causal model with intervention
        let mut modified_variables = self.causal_model.variables.clone();
        
        if let Some(var) = modified_variables.get_mut(intervention_var) {
            var.current_value = intervention_value;
        }
        
        // Propagate effects through causal graph
        let ordering = self.get_causal_ordering();
        let mut predictions = HashMap::new();
        
        for variable_name in ordering {
            if variable_name == intervention_var {
                predictions.insert(variable_name, intervention_value);
            } else {
                // Use structural equation with modified parent values
                let prediction = if let Some(equation) = self.causal_model.structural_equations.get(&variable_name) {
                    let mut pred = equation.coefficients[0]; // Intercept
                    
                    for (i, parent_name) in equation.parent_variables.iter().enumerate() {
                        let parent_value = predictions.get(parent_name)
                            .or_else(|| modified_variables.get(parent_name).map(|v| &v.current_value))
                            .unwrap_or(&0.0);
                        pred += equation.coefficients[i + 1] * parent_value;
                    }
                    
                    pred
                } else {
                    modified_variables.get(&variable_name).map(|v| v.current_value).unwrap_or(0.0)
                };
                
                predictions.insert(variable_name, prediction);
            }
        }
        
        Ok(*predictions.get("rug_pull_event").unwrap_or(&0.0))
    }

    async fn simulate_interventions(&self) -> Result<Vec<InterventionRecommendation>> {
        let mut recommendations = Vec::new();
        
        // Simulate interventions to reduce rug pull probability
        let protective_interventions = vec![
            ("coordination_score", 0.1, "Implement coordination detection alerts"),
            ("whale_concentration", 0.2, "Monitor large wallet movements"),
            ("liquidity", 0.8, "Increase pool liquidity requirements"),
        ];
        
        for (var_name, target_value, description) in protective_interventions {
            let new_probability = self.simulate_counterfactual(var_name, target_value).await?;
            let current_probability = self.causal_model.variables.get("rug_pull_event")
                .map(|v| v.current_value).unwrap_or(0.0);
            
            let risk_reduction = current_probability - new_probability;
            
            recommendations.push(InterventionRecommendation {
                intervention_type: description.to_string(),
                target_variable: var_name.to_string(),
                target_value,
                expected_risk_reduction: risk_reduction.max(0.0),
                implementation_difficulty: self.assess_implementation_difficulty(var_name),
                time_to_effect_hours: self.estimate_intervention_time(var_name),
            });
        }
        
        // Sort by effectiveness
        recommendations.sort_by(|a, b| b.expected_risk_reduction.partial_cmp(&a.expected_risk_reduction).unwrap());
        
        Ok(recommendations)
    }

    fn assess_implementation_difficulty(&self, var_name: &str) -> f64 {
        match var_name {
            "coordination_score" => 0.3, // Easy - just alerting
            "whale_concentration" => 0.5, // Medium - requires monitoring
            "liquidity" => 0.8, // Hard - requires protocol changes
            _ => 0.5,
        }
    }

    fn estimate_intervention_time(&self, var_name: &str) -> i64 {
        match var_name {
            "coordination_score" => 1, // 1 hour to set up alerts
            "whale_concentration" => 6, // 6 hours to implement monitoring
            "liquidity" => 24, // 24 hours for protocol changes
            _ => 12,
        }
    }

    fn summarize_causal_graph(&self) -> CausalGraphSummary {
        CausalGraphSummary {
            total_variables: self.causal_model.variables.len(),
            total_edges: self.causal_model.causal_graph.len(),
            strongest_causal_relationship: self.find_strongest_relationship(),
            avg_causal_strength: self.calculate_avg_causal_strength(),
            key_variables: self.identify_key_variables(),
        }
    }

    fn find_strongest_relationship(&self) -> Option<String> {
        self.causal_model.causal_graph.iter()
            .max_by(|a, b| a.causal_strength.partial_cmp(&b.causal_strength).unwrap())
            .map(|edge| format!("{} → {}", edge.from_variable, edge.to_variable))
    }

    fn calculate_avg_causal_strength(&self) -> f64 {
        if self.causal_model.causal_graph.is_empty() {
            0.0
        } else {
            self.causal_model.causal_graph.iter()
                .map(|edge| edge.causal_strength)
                .sum::<f64>() / self.causal_model.causal_graph.len() as f64
        }
    }

    fn identify_key_variables(&self) -> Vec<String> {
        // Variables with the most causal connections
        let mut variable_importance: HashMap<String, usize> = HashMap::new();
        
        for edge in &self.causal_model.causal_graph {
            *variable_importance.entry(edge.from_variable.clone()).or_default() += 1;
            *variable_importance.entry(edge.to_variable.clone()).or_default() += 1;
        }
        
        let mut sorted_vars: Vec<_> = variable_importance.into_iter().collect();
        sorted_vars.sort_by(|a, b| b.1.cmp(&a.1));
        
        sorted_vars.into_iter().take(5).map(|(name, _)| name).collect()
    }
}

// Supporting structures and algorithms

pub struct CausalDiscoveryAlgorithm {
    algorithm_type: DiscoveryAlgorithm,
    significance_threshold: f64,
}

#[derive(Debug)]
pub enum DiscoveryAlgorithm {
    PC,           // Peter-Clark algorithm
    GES,          // Greedy Equivalence Search
    NOTEARS,      // Neural approach
    Simplified,   // Our simplified approach
}

impl CausalDiscoveryAlgorithm {
    pub fn new() -> Result<Self> {
        Ok(Self {
            algorithm_type: DiscoveryAlgorithm::Simplified,
            significance_threshold: 0.05,
        })
    }
}

pub struct InterventionSimulator {
    simulation_runs: usize,
    confidence_level: f64,
}

impl InterventionSimulator {
    pub fn new() -> Self {
        Self {
            simulation_runs: 1000,
            confidence_level: 0.95,
        }
    }
}

pub struct CounterfactualEngine {
    num_samples: usize,
    bootstrap_iterations: usize,
}

impl CounterfactualEngine {
    pub fn new() -> Self {
        Self {
            num_samples: 1000,
            bootstrap_iterations: 100,
        }
    }
}

// Result structures

#[derive(Debug, Serialize)]
pub struct CausalPredictionResult {
    pub rug_pull_probability: f64,
    pub confidence: f64,
    pub prediction_horizon_hours: i64,
    pub causal_factors: Vec<CausalFactor>,
    pub counterfactual_scenarios: Vec<CounterfactualScenario>,
    pub recommended_interventions: Vec<InterventionRecommendation>,
    pub causal_graph_summary: CausalGraphSummary,
    pub processing_time_ms: u64,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CausalFactor {
    pub variable_name: String,
    pub current_value: f64,
    pub causal_strength: f64,
    pub contribution_to_risk: f64,
    pub time_to_effect_seconds: i64,
}

#[derive(Debug, Clone, Serialize)]
pub struct CounterfactualScenario {
    pub scenario_name: String,
    pub intervention_variable: String,
    pub intervention_value: f64,
    pub predicted_rug_pull_probability: f64,
    pub probability_change: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct InterventionRecommendation {
    pub intervention_type: String,
    pub target_variable: String,
    pub target_value: f64,
    pub expected_risk_reduction: f64,
    pub implementation_difficulty: f64, // 0.0 (easy) to 1.0 (very hard)
    pub time_to_effect_hours: i64,
}

#[derive(Debug, Serialize)]
pub struct CausalGraphSummary {
    pub total_variables: usize,
    pub total_edges: usize,
    pub strongest_causal_relationship: Option<String>,
    pub avg_causal_strength: f64,
    pub key_variables: Vec<String>,
}

#[derive(Debug)]
struct CausalPrediction {
    rug_pull_probability: f64,
    confidence: f64,
    causal_factors: Vec<CausalFactor>,
}

#[derive(Debug)]
struct InterventionExperiment {
    experiment_id: String,
    intervention_variable: String,
    intervention_value: f64,
    outcome: f64,
    timestamp: DateTime<Utc>,
}

#[derive(Debug)]
struct CounterfactualResult {
    scenario_id: String,
    counterfactual_outcome: f64,
    confidence_interval: (f64, f64),
    timestamp: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    #[tokio::test]
    async fn test_causal_inference_engine_creation() {
        let engine = CausalInferenceEngine::new(24).unwrap();
        assert_eq!(engine.prediction_horizon_hours, 24);
        assert!(engine.causal_model.variables.contains_key("rug_pull_event"));
        assert!(engine.causal_model.variables.contains_key("coordination_score"));
    }

    #[tokio::test] 
    async fn test_causal_prediction() {
        let mut engine = CausalInferenceEngine::new(12).unwrap();
        
        // Create test transactions with coordination patterns
        let transactions = vec![
            crate::analytics::Transaction {
                signature: "causal_tx_1".to_string(),
                wallet: "coordinated_wallet_1".to_string(),
                token_mint: "test_token".to_string(),
                amount_sol: 100.0,
                transaction_type: crate::analytics::TransactionType::Buy,
                timestamp: Utc::now(),
            },
            crate::analytics::Transaction {
                signature: "causal_tx_2".to_string(),
                wallet: "coordinated_wallet_2".to_string(),
                token_mint: "test_token".to_string(),
                amount_sol: 100.0, // Same amount (coordinated)
                transaction_type: crate::analytics::TransactionType::Buy,
                timestamp: Utc::now() + Duration::minutes(2), // Close timing
            },
        ];

        let result = engine.predict_rug_pull_probability(&transactions, None).await.unwrap();
        
        assert!(result.rug_pull_probability >= 0.0 && result.rug_pull_probability <= 1.0);
        assert!(result.confidence > 0.0);
        assert_eq!(result.prediction_horizon_hours, 12);
        assert!(!result.causal_factors.is_empty());
        assert!(!result.counterfactual_scenarios.is_empty());
        
        println!("Causal prediction result:");
        println!("  Rug pull probability: {:.3}", result.rug_pull_probability);
        println!("  Confidence: {:.3}", result.confidence);
        println!("  Key causal factors: {}", result.causal_factors.len());
        println!("  Counterfactual scenarios: {}", result.counterfactual_scenarios.len());
    }

    #[tokio::test]
    async fn test_causal_variable_updates() {
        let mut engine = CausalInferenceEngine::new(24).unwrap();
        
        let transactions = vec![
            crate::analytics::Transaction {
                signature: "test_1".to_string(),
                wallet: "wallet_1".to_string(),
                token_mint: "token".to_string(),
                amount_sol: 50.0,
                transaction_type: crate::analytics::TransactionType::Buy,
                timestamp: Utc::now(),
            },
        ];

        engine.update_causal_variables(&transactions, None).await.unwrap();
        
        // Check that variables were updated
        let coordination_var = engine.causal_model.variables.get("coordination_score").unwrap();
        assert!(!coordination_var.historical_values.is_empty());
        
        let wallet_sim_var = engine.causal_model.variables.get("wallet_similarity").unwrap();
        assert!(!wallet_sim_var.historical_values.is_empty());
    }

    #[test]
    fn test_coordination_score_calculation() {
        let engine = CausalInferenceEngine::new(24).unwrap();
        
        // High coordination transactions (same amounts, close timing)
        let coordinated_txs = vec![
            crate::analytics::Transaction {
                signature: "coord_1".to_string(),
                wallet: "wallet_1".to_string(),
                token_mint: "token".to_string(),
                amount_sol: 100.0,
                transaction_type: crate::analytics::TransactionType::Buy,
                timestamp: Utc::now(),
            },
            crate::analytics::Transaction {
                signature: "coord_2".to_string(),
                wallet: "wallet_2".to_string(),
                token_mint: "token".to_string(),
                amount_sol: 100.0, // Same amount
                transaction_type: crate::analytics::TransactionType::Buy,
                timestamp: Utc::now() + Duration::seconds(30), // Close timing
            },
        ];

        let coordination_score = engine.calculate_coordination_score(&coordinated_txs).unwrap();
        
        // Should detect high coordination
        assert!(coordination_score > 0.5, "Expected high coordination score, got {}", coordination_score);
        
        // Random transactions should have low coordination
        let random_txs = vec![
            crate::analytics::Transaction {
                signature: "random_1".to_string(),
                wallet: "wallet_1".to_string(),
                token_mint: "token".to_string(),
                amount_sol: 37.5,
                transaction_type: crate::analytics::TransactionType::Buy,
                timestamp: Utc::now(),
            },
            crate::analytics::Transaction {
                signature: "random_2".to_string(),
                wallet: "wallet_2".to_string(),
                token_mint: "token".to_string(),
                amount_sol: 182.3,
                transaction_type: crate::analytics::TransactionType::Sell,
                timestamp: Utc::now() + Duration::hours(3),
            },
        ];

        let random_score = engine.calculate_coordination_score(&random_txs).unwrap();
        assert!(random_score < coordination_score, "Random transactions should have lower coordination");
    }

    #[tokio::test]
    async fn test_counterfactual_scenarios() {
        let mut engine = CausalInferenceEngine::new(24).unwrap();
        
        // Set up some causal relationships and equations
        engine.causal_model.structural_equations.insert("rug_pull_event".to_string(), StructuralEquation {
            target_variable: "rug_pull_event".to_string(),
            equation_type: EquationType::Linear,
            coefficients: vec![0.1, 0.8, 0.6], // intercept, coordination_score, whale_concentration
            parent_variables: vec!["coordination_score".to_string(), "whale_concentration".to_string()],
            noise_variance: 0.1,
            nonlinear_terms: Vec::new(),
        });
        
        let scenarios = engine.generate_counterfactual_scenarios().await.unwrap();
        
        assert!(!scenarios.is_empty());
        assert!(scenarios.iter().any(|s| s.intervention_variable == "coordination_score"));
        
        println!("Generated {} counterfactual scenarios", scenarios.len());
        for scenario in &scenarios {
            println!("  {}: P(rug pull) = {:.3} (change: {:+.3})", 
                scenario.scenario_name, 
                scenario.predicted_rug_pull_probability,
                scenario.probability_change);
        }
    }
}