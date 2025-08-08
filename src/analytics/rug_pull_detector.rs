//! Coordinated Rug Pull Detection System
//! 
//! This module implements advanced detection algorithms to identify coordinated
//! rug pull attacks by analyzing early investor behavior, transaction patterns,
//! and wallet clustering techniques.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc, Duration};
use std::collections::{HashMap, HashSet, BTreeMap};
use solana_sdk::pubkey::Pubkey;
use ndarray::{Array1, Array2};
use rayon::prelude::*;

use super::{Transaction, TransactionType, WhalePersonality, TradePatterns};
use crate::types::{AlphaWalletSignal, PendingTransaction};

/// Early investor behavior patterns that may indicate coordinated rug pulls
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyInvestorCluster {
    pub cluster_id: String,
    pub wallets: Vec<String>,
    pub token_mint: String,
    pub first_purchase_time: DateTime<Utc>,
    pub total_investment: f64,
    pub coordination_score: f64,
    pub behavioral_flags: Vec<CoordinationFlag>,
    pub risk_level: RiskLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoordinationFlag {
    SimilarTiming,        // Purchases within narrow time window
    SimilarAmounts,       // Similar purchase amounts
    NewWallets,           // Recently created wallet addresses
    BatchTransactions,    // Transactions in same block/batch
    SimilarSources,       // Funded from same source wallet
    UniformBehavior,      // Identical transaction patterns
    RapidAccumulation,    // Fast accumulation phase
    SuspiciousMetadata,   // Unusual transaction metadata
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RugPullAlert {
    pub alert_id: String,
    pub token_mint: String,
    pub clusters: Vec<EarlyInvestorCluster>,
    pub overall_risk_score: f64,
    pub confidence: f64,
    pub timestamp: DateTime<Utc>,
    pub recommended_actions: Vec<String>,
    pub evidence: Vec<String>,
}

/// Configuration for rug pull detection parameters
#[derive(Debug, Clone)]
pub struct DetectionConfig {
    pub timing_threshold_minutes: i64,
    pub amount_similarity_threshold: f64,
    pub cluster_min_size: usize,
    pub new_wallet_threshold_days: i64,
    pub coordination_score_threshold: f64,
}

impl Default for DetectionConfig {
    fn default() -> Self {
        Self {
            timing_threshold_minutes: 30,      // 30 minute window for coordinated timing
            amount_similarity_threshold: 0.15, // 15% variance in amounts
            cluster_min_size: 3,               // Minimum 3 wallets for a cluster
            new_wallet_threshold_days: 7,      // Wallets newer than 7 days are suspicious
            coordination_score_threshold: 0.7, // 70% coordination score triggers alert
        }
    }
}

pub struct RugPullDetector {
    config: DetectionConfig,
    wallet_history_cache: HashMap<String, WalletHistory>,
    active_clusters: HashMap<String, Vec<EarlyInvestorCluster>>,
}

#[derive(Debug, Clone)]
struct WalletHistory {
    first_seen: DateTime<Utc>,
    total_transactions: usize,
    funding_sources: HashSet<String>,
    behavior_patterns: Vec<String>,
}

impl RugPullDetector {
    pub fn new() -> Self {
        Self {
            config: DetectionConfig::default(),
            wallet_history_cache: HashMap::new(),
            active_clusters: HashMap::new(),
        }
    }

    pub fn with_config(config: DetectionConfig) -> Self {
        Self {
            config,
            wallet_history_cache: HashMap::new(),
            active_clusters: HashMap::new(),
        }
    }

    /// Main analysis function: detect coordinated rug pull patterns
    pub async fn analyze_transactions(&mut self, transactions: &[Transaction]) -> Result<Vec<RugPullAlert>> {
        // Group transactions by token
        let mut token_transactions: HashMap<String, Vec<&Transaction>> = HashMap::new();
        for tx in transactions {
            token_transactions.entry(tx.token_mint.clone()).or_default().push(tx);
        }

        let mut alerts = Vec::new();

        // Analyze each token separately
        for (token_mint, token_txs) in token_transactions {
            if let Some(alert) = self.analyze_token_transactions(&token_mint, &token_txs).await? {
                alerts.push(alert);
            }
        }

        Ok(alerts)
    }

    async fn analyze_token_transactions(
        &mut self, 
        token_mint: &str, 
        transactions: &[&Transaction]
    ) -> Result<Option<RugPullAlert>> {
        // Filter to buy transactions only (early investors)
        let buy_transactions: Vec<&Transaction> = transactions
            .iter()
            .filter(|tx| matches!(tx.transaction_type, TransactionType::Buy))
            .copied()
            .collect();

        if buy_transactions.len() < self.config.cluster_min_size {
            return Ok(None);
        }

        // Update wallet history cache
        self.update_wallet_history(&buy_transactions).await?;

        // Identify potential clusters
        let clusters = self.identify_coordination_clusters(token_mint, &buy_transactions).await?;
        
        if clusters.is_empty() {
            return Ok(None);
        }

        // Calculate overall risk score
        let overall_risk_score = self.calculate_overall_risk(&clusters).await?;
        
        if overall_risk_score < self.config.coordination_score_threshold {
            return Ok(None);
        }

        // Generate alert
        let alert = RugPullAlert {
            alert_id: format!("rugpull_{}_{}", token_mint, Utc::now().timestamp()),
            token_mint: token_mint.to_string(),
            clusters: clusters.clone(),
            overall_risk_score,
            confidence: self.calculate_confidence(&clusters).await?,
            timestamp: Utc::now(),
            recommended_actions: self.generate_recommendations(&clusters).await?,
            evidence: self.collect_evidence(&clusters).await?,
        };

        // Cache the clusters
        self.active_clusters.insert(token_mint.to_string(), clusters);

        Ok(Some(alert))
    }

    async fn update_wallet_history(&mut self, transactions: &[&Transaction]) -> Result<()> {
        for tx in transactions {
            let wallet = &tx.wallet;
            let history = self.wallet_history_cache.entry(wallet.clone()).or_insert(WalletHistory {
                first_seen: tx.timestamp,
                total_transactions: 0,
                funding_sources: HashSet::new(),
                behavior_patterns: Vec::new(),
            });

            history.total_transactions += 1;
            if tx.timestamp < history.first_seen {
                history.first_seen = tx.timestamp;
            }
        }
        Ok(())
    }

    async fn identify_coordination_clusters(
        &self,
        token_mint: &str,
        transactions: &[&Transaction]
    ) -> Result<Vec<EarlyInvestorCluster>> {
        let mut clusters = Vec::new();

        // Sort transactions by timestamp
        let mut sorted_txs = transactions.to_vec();
        sorted_txs.sort_by_key(|tx| tx.timestamp);

        // Use sliding window to find coordinated timing clusters
        let timing_clusters = self.find_timing_clusters(&sorted_txs).await?;
        
        for timing_cluster in timing_clusters {
            if timing_cluster.len() < self.config.cluster_min_size {
                continue;
            }

            // Analyze the cluster for coordination patterns
            let coordination_score = self.calculate_coordination_score(&timing_cluster).await?;
            let behavioral_flags = self.identify_behavioral_flags(&timing_cluster).await?;
            let risk_level = self.assess_risk_level(coordination_score, &behavioral_flags).await?;

            if coordination_score >= self.config.coordination_score_threshold {
                let cluster = EarlyInvestorCluster {
                    cluster_id: format!("cluster_{}_{}", token_mint, clusters.len()),
                    wallets: timing_cluster.iter().map(|tx| tx.wallet.clone()).collect(),
                    token_mint: token_mint.to_string(),
                    first_purchase_time: timing_cluster.iter().min_by_key(|tx| tx.timestamp).unwrap().timestamp,
                    total_investment: timing_cluster.iter().map(|tx| tx.amount_sol).sum(),
                    coordination_score,
                    behavioral_flags,
                    risk_level,
                };
                clusters.push(cluster);
            }
        }

        Ok(clusters)
    }

    async fn find_timing_clusters(&self, transactions: &[&Transaction]) -> Result<Vec<Vec<&Transaction>>> {
        let mut clusters = Vec::new();
        let timing_threshold = Duration::minutes(self.config.timing_threshold_minutes);

        let mut current_cluster = Vec::new();
        let mut cluster_start_time: Option<DateTime<Utc>> = None;

        for tx in transactions {
            match cluster_start_time {
                None => {
                    // Start new cluster
                    current_cluster.push(*tx);
                    cluster_start_time = Some(tx.timestamp);
                }
                Some(start_time) => {
                    if tx.timestamp - start_time <= timing_threshold {
                        // Add to current cluster
                        current_cluster.push(*tx);
                    } else {
                        // Finalize current cluster and start new one
                        if current_cluster.len() >= self.config.cluster_min_size {
                            clusters.push(current_cluster.clone());
                        }
                        current_cluster.clear();
                        current_cluster.push(*tx);
                        cluster_start_time = Some(tx.timestamp);
                    }
                }
            }
        }

        // Don't forget the last cluster
        if current_cluster.len() >= self.config.cluster_min_size {
            clusters.push(current_cluster);
        }

        Ok(clusters)
    }

    async fn calculate_coordination_score(&self, cluster: &[&Transaction]) -> Result<f64> {
        if cluster.len() < 2 {
            return Ok(0.0);
        }

        let mut coordination_factors = Vec::new();

        // 1. Timing coordination (already clustered by timing)
        let timing_score = 1.0; // High since we already clustered by timing
        coordination_factors.push(timing_score * 0.3);

        // 2. Amount similarity
        let amounts: Vec<f64> = cluster.iter().map(|tx| tx.amount_sol).collect();
        let amount_similarity = self.calculate_amount_similarity(&amounts).await?;
        coordination_factors.push(amount_similarity * 0.25);

        // 3. New wallet factor
        let new_wallet_score = self.calculate_new_wallet_score(cluster).await?;
        coordination_factors.push(new_wallet_score * 0.2);

        // 4. Behavioral uniformity
        let behavior_score = self.calculate_behavioral_uniformity(cluster).await?;
        coordination_factors.push(behavior_score * 0.15);

        // 5. Funding source similarity
        let funding_score = self.calculate_funding_similarity(cluster).await?;
        coordination_factors.push(funding_score * 0.1);

        let total_score: f64 = coordination_factors.iter().sum();
        Ok(total_score.min(1.0))
    }

    async fn calculate_amount_similarity(&self, amounts: &[f64]) -> Result<f64> {
        if amounts.len() < 2 {
            return Ok(0.0);
        }

        let amounts_array = Array1::from(amounts.to_vec());
        let mean = amounts_array.mean().unwrap_or(0.0);
        let std_dev = amounts_array.mapv(|x| (x - mean).powi(2)).mean().unwrap_or(0.0).sqrt();
        
        let coefficient_of_variation = if mean > 0.0 { std_dev / mean } else { 1.0 };
        
        // Lower CV = higher similarity
        let similarity = (1.0 - coefficient_of_variation.min(1.0)).max(0.0);
        Ok(similarity)
    }

    async fn calculate_new_wallet_score(&self, cluster: &[&Transaction]) -> Result<f64> {
        let threshold = Utc::now() - Duration::days(self.config.new_wallet_threshold_days);
        let new_wallet_count = cluster
            .iter()
            .filter(|tx| {
                self.wallet_history_cache
                    .get(&tx.wallet)
                    .map(|history| history.first_seen > threshold)
                    .unwrap_or(true)
            })
            .count();

        Ok(new_wallet_count as f64 / cluster.len() as f64)
    }

    async fn calculate_behavioral_uniformity(&self, cluster: &[&Transaction]) -> Result<f64> {
        if cluster.len() < 2 {
            return Ok(0.0);
        }

        // Analyze transaction patterns for uniformity
        let mut slippage_values: Vec<f64> = Vec::new();
        let mut amounts: Vec<f64> = Vec::new();

        for tx in cluster {
            amounts.push(tx.amount_sol);
            // Note: We'd need slippage data from the actual transaction, using amount as proxy
            slippage_values.push(tx.amount_sol.fract()); // Use fractional part as proxy
        }

        let amount_uniformity = 1.0 - (self.calculate_coefficient_of_variation(&amounts)?);
        let slippage_uniformity = 1.0 - (self.calculate_coefficient_of_variation(&slippage_values)?);

        Ok((amount_uniformity + slippage_uniformity) / 2.0)
    }

    async fn calculate_funding_similarity(&self, cluster: &[&Transaction]) -> Result<f64> {
        // This would require additional on-chain analysis to trace funding sources
        // For now, return a moderate score based on wallet age correlation
        let wallet_ages: Vec<f64> = cluster
            .iter()
            .map(|tx| {
                self.wallet_history_cache
                    .get(&tx.wallet)
                    .map(|history| (Utc::now() - history.first_seen).num_days() as f64)
                    .unwrap_or(365.0)
            })
            .collect();

        let age_similarity = 1.0 - self.calculate_coefficient_of_variation(&wallet_ages)?;
        Ok(age_similarity * 0.5) // Conservative estimate without full funding analysis
    }

    fn calculate_coefficient_of_variation(&self, values: &[f64]) -> Result<f64> {
        if values.len() < 2 {
            return Ok(0.0);
        }

        let values_array = Array1::from(values.to_vec());
        let mean = values_array.mean().unwrap_or(0.0);
        let std_dev = values_array.mapv(|x| (x - mean).powi(2)).mean().unwrap_or(0.0).sqrt();
        
        Ok(if mean > 0.0 { std_dev / mean } else { 0.0 })
    }

    async fn identify_behavioral_flags(&self, cluster: &[&Transaction]) -> Result<Vec<CoordinationFlag>> {
        let mut flags = Vec::new();

        // Check timing similarity (already done in clustering)
        flags.push(CoordinationFlag::SimilarTiming);

        // Check amount similarity
        let amounts: Vec<f64> = cluster.iter().map(|tx| tx.amount_sol).collect();
        let amount_cv = self.calculate_coefficient_of_variation(&amounts)?;
        if amount_cv < self.config.amount_similarity_threshold {
            flags.push(CoordinationFlag::SimilarAmounts);
        }

        // Check for new wallets
        let threshold = Utc::now() - Duration::days(self.config.new_wallet_threshold_days);
        let new_wallet_ratio = cluster
            .iter()
            .filter(|tx| {
                self.wallet_history_cache
                    .get(&tx.wallet)
                    .map(|history| history.first_seen > threshold)
                    .unwrap_or(true)
            })
            .count() as f64 / cluster.len() as f64;

        if new_wallet_ratio > 0.5 {
            flags.push(CoordinationFlag::NewWallets);
        }

        // Check for rapid accumulation
        let time_span = cluster.iter().max_by_key(|tx| tx.timestamp).unwrap().timestamp
            - cluster.iter().min_by_key(|tx| tx.timestamp).unwrap().timestamp;
        if time_span <= Duration::minutes(self.config.timing_threshold_minutes) {
            flags.push(CoordinationFlag::RapidAccumulation);
        }

        Ok(flags)
    }

    async fn assess_risk_level(&self, coordination_score: f64, flags: &[CoordinationFlag]) -> Result<RiskLevel> {
        let high_risk_flags = [
            CoordinationFlag::NewWallets,
            CoordinationFlag::SimilarSources,
            CoordinationFlag::UniformBehavior,
        ];

        let high_risk_count = flags.iter()
            .filter(|flag| high_risk_flags.contains(flag))
            .count();

        let risk_level = if coordination_score >= 0.9 && high_risk_count >= 2 {
            RiskLevel::Critical
        } else if coordination_score >= 0.8 || high_risk_count >= 2 {
            RiskLevel::High
        } else if coordination_score >= 0.6 || flags.len() >= 3 {
            RiskLevel::Medium
        } else {
            RiskLevel::Low
        };

        Ok(risk_level)
    }

    async fn calculate_overall_risk(&self, clusters: &[EarlyInvestorCluster]) -> Result<f64> {
        if clusters.is_empty() {
            return Ok(0.0);
        }

        // Weight clusters by size and coordination score
        let weighted_score: f64 = clusters
            .iter()
            .map(|cluster| {
                let size_weight = (cluster.wallets.len() as f64).ln() / 10.0; // Log scale for size
                let investment_weight = (cluster.total_investment / 1000.0).min(1.0); // Cap at 1000 SOL
                cluster.coordination_score * (1.0 + size_weight + investment_weight * 0.5)
            })
            .sum();

        let average_weighted_score = weighted_score / clusters.len() as f64;
        Ok(average_weighted_score.min(1.0))
    }

    async fn calculate_confidence(&self, clusters: &[EarlyInvestorCluster]) -> Result<f64> {
        if clusters.is_empty() {
            return Ok(0.0);
        }

        let total_evidence_points: usize = clusters
            .iter()
            .map(|cluster| cluster.behavioral_flags.len())
            .sum();

        let total_wallets: usize = clusters
            .iter()
            .map(|cluster| cluster.wallets.len())
            .sum();

        // More evidence and larger clusters = higher confidence
        let evidence_confidence = (total_evidence_points as f64 / (clusters.len() * 8) as f64).min(1.0);
        let size_confidence = ((total_wallets as f64).ln() / 10.0).min(1.0);
        
        Ok((evidence_confidence * 0.7 + size_confidence * 0.3).min(1.0))
    }

    async fn generate_recommendations(&self, clusters: &[EarlyInvestorCluster]) -> Result<Vec<String>> {
        let mut recommendations = Vec::new();

        let max_risk_level = clusters
            .iter()
            .map(|c| &c.risk_level)
            .max()
            .unwrap_or(&RiskLevel::Low);

        match max_risk_level {
            RiskLevel::Critical => {
                recommendations.push("IMMEDIATE ACTION: High probability coordinated rug pull detected".to_string());
                recommendations.push("Avoid all purchases of this token".to_string());
                recommendations.push("Monitor for mass sell-off patterns".to_string());
                recommendations.push("Alert other traders in network".to_string());
            }
            RiskLevel::High => {
                recommendations.push("HIGH RISK: Strong coordination patterns detected".to_string());
                recommendations.push("Exercise extreme caution with this token".to_string());
                recommendations.push("Monitor cluster behavior closely".to_string());
            }
            RiskLevel::Medium => {
                recommendations.push("MEDIUM RISK: Potential coordination detected".to_string());
                recommendations.push("Increase monitoring of this token".to_string());
                recommendations.push("Use smaller position sizes".to_string());
            }
            RiskLevel::Low => {
                recommendations.push("LOW RISK: Minor coordination patterns".to_string());
                recommendations.push("Continue normal monitoring".to_string());
            }
        }

        Ok(recommendations)
    }

    async fn collect_evidence(&self, clusters: &[EarlyInvestorCluster]) -> Result<Vec<String>> {
        let mut evidence = Vec::new();

        for cluster in clusters {
            evidence.push(format!(
                "Cluster {} has {} wallets with {:.1}% coordination score",
                cluster.cluster_id, cluster.wallets.len(), cluster.coordination_score * 100.0
            ));

            for flag in &cluster.behavioral_flags {
                let flag_description = match flag {
                    CoordinationFlag::SimilarTiming => "Purchases within narrow time window",
                    CoordinationFlag::SimilarAmounts => "Similar purchase amounts detected",
                    CoordinationFlag::NewWallets => "High proportion of recently created wallets",
                    CoordinationFlag::BatchTransactions => "Transactions grouped in same blocks",
                    CoordinationFlag::SimilarSources => "Wallets funded from similar sources",
                    CoordinationFlag::UniformBehavior => "Identical transaction patterns",
                    CoordinationFlag::RapidAccumulation => "Rapid accumulation phase detected",
                    CoordinationFlag::SuspiciousMetadata => "Unusual transaction metadata",
                };
                evidence.push(format!("- {}", flag_description));
            }
        }

        Ok(evidence)
    }

    /// Get active alerts for monitoring
    pub fn get_active_clusters(&self) -> &HashMap<String, Vec<EarlyInvestorCluster>> {
        &self.active_clusters
    }

    /// Clear old clusters (should be called periodically)
    pub fn cleanup_old_clusters(&mut self, max_age_hours: i64) {
        let cutoff = Utc::now() - Duration::hours(max_age_hours);
        self.active_clusters.retain(|_, clusters| {
            clusters.iter().any(|cluster| cluster.first_purchase_time > cutoff)
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_transaction(wallet: &str, amount: f64, timestamp: DateTime<Utc>) -> Transaction {
        Transaction {
            signature: format!("sig_{}", wallet),
            wallet: wallet.to_string(),
            token_mint: "test_token".to_string(),
            amount_sol: amount,
            transaction_type: TransactionType::Buy,
            timestamp,
        }
    }

    #[tokio::test]
    async fn test_coordination_detection() {
        let mut detector = RugPullDetector::new();
        
        let base_time = Utc::now();
        let transactions = vec![
            create_test_transaction("wallet1", 100.0, base_time),
            create_test_transaction("wallet2", 105.0, base_time + Duration::minutes(2)),
            create_test_transaction("wallet3", 98.0, base_time + Duration::minutes(5)),
            create_test_transaction("wallet4", 102.0, base_time + Duration::minutes(8)),
        ];

        let tx_refs: Vec<&Transaction> = transactions.iter().collect();
        let alerts = detector.analyze_transactions(&tx_refs).await.unwrap();
        
        assert!(!alerts.is_empty(), "Should detect coordination in test data");
        
        let alert = &alerts[0];
        assert!(alert.overall_risk_score > 0.5, "Should have significant risk score");
        assert!(!alert.clusters.is_empty(), "Should identify clusters");
    }
}