//! Rug Pull Monitoring Service
//! 
//! This service integrates the rug pull detector with the blockchain listener
//! and provides real-time monitoring and alerting capabilities.

use anyhow::Result;
use tokio::sync::{mpsc, broadcast};
use tokio::time::{interval, Duration as TokioDuration};
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use std::collections::HashMap;

use super::{RugPullDetector, RugPullAlert, EarlyInvestorCluster, Transaction};
use crate::infrastructure::mempool::{MempoolMonitor, PendingTransaction};
use crate::modules::shared::training::combat_logger::{self, CombatContext};

#[derive(Debug, Clone, Serialize)]
pub struct RugPullEvent {
    pub event_type: EventType,
    pub token_mint: String,
    pub alert: Option<RugPullAlert>,
    pub cluster_update: Option<EarlyInvestorCluster>,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize)]
pub enum EventType {
    NewAlert,
    ClusterUpdate,
    AlertResolved,
    HighRiskDetected,
}

pub struct RugPullMonitor {
    detector: RugPullDetector,
    mempool_monitor: MempoolMonitor,
    event_sender: broadcast::Sender<RugPullEvent>,
    transaction_buffer: Vec<Transaction>,
    buffer_size: usize,
    analysis_interval: TokioDuration,
    context: CombatContext,
}

impl RugPullMonitor {
    pub fn new() -> Result<(Self, broadcast::Receiver<RugPullEvent>)> {
        let (event_sender, event_receiver) = broadcast::channel(1000);
        
        let monitor = Self {
            detector: RugPullDetector::new(),
            mempool_monitor: MempoolMonitor::new()?,
            event_sender,
            transaction_buffer: Vec::new(),
            buffer_size: 1000,
            analysis_interval: TokioDuration::from_secs(30),
            context: CombatContext {
                operation_id: "rug_pull_recon".to_string(),
                squad: "cyber_warfare".to_string(),
                trace_id: uuid::Uuid::new_v4().to_string(),
            },
        };

        Ok((monitor, event_receiver))
    }

    /// Start the monitoring service
    pub async fn start_monitoring(&mut self) -> Result<()> {
        let mut analysis_timer = interval(self.analysis_interval);
        let mut cleanup_timer = interval(TokioDuration::from_secs(3600)); // 1 hour

        combat_logger::severity::info(
            "rug_pull_recon",
            "surveillance_initiated",
            serde_json::json!({
                "buffer_capacity": self.buffer_size,
                "scan_interval_secs": self.analysis_interval.as_secs(),
                "status": "weapons_hot"
            }),
            Some(self.context.clone())
        ).await?;

        loop {
            tokio::select! {
                // Collect new transactions
                _ = self.collect_transactions() => {},
                
                // Periodic analysis
                _ = analysis_timer.tick() => {
                    if let Err(e) = self.analyze_buffered_transactions().await {
                        combat_logger::severity::error(
                            "rug_pull_recon",
                            "tactical_analysis_failure",
                            serde_json::json!({ "error": e.to_string() }),
                            Some(self.context.clone())
                        ).await?;
                    }
                },
                
                // Periodic cleanup
                _ = cleanup_timer.tick() => {
                    self.detector.cleanup_old_clusters(24);
                    combat_logger::severity::debug(
                        "rug_pull_recon", 
                        "intel_cleanup_complete",
                        serde_json::json!({ "operation": "cluster_purge" }),
                        Some(self.context.clone())
                    ).await?;
                }
            }
        }
    }

    async fn collect_transactions(&mut self) -> Result<()> {
        // Get pending transactions from mempool
        let pending_txs = self.mempool_monitor.get_pending_transactions().await?;
        
        // Convert to our Transaction format
        for pending_tx in pending_txs {
            let transaction = self.convert_pending_transaction(pending_tx)?;
            self.add_to_buffer(transaction).await?;
        }

        Ok(())
    }

    fn convert_pending_transaction(&self, pending_tx: PendingTransaction) -> Result<Transaction> {
        // Determine transaction type based on program_id and amount
        let transaction_type = if pending_tx.program_id.contains("JUP") {
            super::TransactionType::Swap
        } else if pending_tx.amount > 0.0 {
            super::TransactionType::Buy
        } else {
            super::TransactionType::Sell
        };

        Ok(Transaction {
            signature: pending_tx.signature,
            wallet: "extracted_from_signature".to_string(), // Would extract from actual transaction
            token_mint: pending_tx.token_mint,
            amount_sol: pending_tx.amount,
            transaction_type,
            timestamp: pending_tx.timestamp,
        })
    }

    async fn add_to_buffer(&mut self, transaction: Transaction) -> Result<()> {
        self.transaction_buffer.push(transaction);
        
        // Maintain buffer size
        if self.transaction_buffer.len() > self.buffer_size {
            let overflow = self.transaction_buffer.len() - self.buffer_size;
            self.transaction_buffer.drain(0..overflow);
        }

        Ok(())
    }

    async fn analyze_buffered_transactions(&mut self) -> Result<()> {
        if self.transaction_buffer.is_empty() {
            return Ok(());
        }

        println!("🔍 Starting analysis of {} transactions", self.transaction_buffer.len());

        // Analyze transactions for rug pull patterns
        let alerts = self.detector.analyze_transactions(&self.transaction_buffer).await?;

        // Process each alert
        for alert in alerts {
            self.handle_rug_pull_alert(alert).await?;
        }

        // Clear processed transactions (keep recent ones for context)
        let keep_count = self.buffer_size / 4; // Keep 25% for context
        if self.transaction_buffer.len() > keep_count {
            let drain_count = self.transaction_buffer.len() - keep_count;
            self.transaction_buffer.drain(0..drain_count);
        }

        Ok(())
    }

    async fn handle_rug_pull_alert(&self, alert: RugPullAlert) -> Result<()> {
        // Log the alert
        let severity = match alert.overall_risk_score {
            score if score >= 0.9 => "🚨 CRITICAL",
            score if score >= 0.7 => "⚠️  WARNING",
            _ => "ℹ️  INFO",
        };

        println!("{} Rug Pull Alert Detected!", severity);
        println!("   Alert ID: {}", alert.alert_id);
        println!("   Token: {}", alert.token_mint);
        println!("   Risk Score: {:.1}%", alert.overall_risk_score * 100.0);
        println!("   Confidence: {:.1}%", alert.confidence * 100.0);
        println!("   Clusters: {}", alert.clusters.len());
        println!("   Total Wallets: {}", alert.clusters.iter().map(|c| c.wallets.len()).sum::<usize>());

        // Broadcast event
        let event = RugPullEvent {
            event_type: if alert.overall_risk_score >= 0.8 {
                EventType::HighRiskDetected
            } else {
                EventType::NewAlert
            },
            token_mint: alert.token_mint.clone(),
            alert: Some(alert),
            cluster_update: None,
            timestamp: Utc::now(),
        };

        // Send event (ignore if no receivers)
        let _ = self.event_sender.send(event);

        Ok(())
    }

    /// Get a snapshot of current monitoring status
    pub async fn get_monitoring_status(&self) -> Result<MonitoringStatus> {
        let active_clusters = self.detector.get_active_clusters();
        
        let total_clusters = active_clusters.values().map(|v| v.len()).sum();
        let total_monitored_tokens = active_clusters.len();
        
        let high_risk_tokens = active_clusters
            .values()
            .flatten()
            .filter(|cluster| matches!(cluster.risk_level, super::RiskLevel::High | super::RiskLevel::Critical))
            .count();

        Ok(MonitoringStatus {
            total_monitored_tokens,
            total_clusters,
            high_risk_tokens,
            buffer_size: self.transaction_buffer.len(),
            last_analysis: Utc::now(),
        })
    }

    /// Manually trigger analysis (useful for testing)
    pub async fn trigger_analysis(&mut self) -> Result<Vec<RugPullAlert>> {
        self.detector.analyze_transactions(&self.transaction_buffer).await
    }

    /// Subscribe to rug pull events
    pub fn subscribe(&self) -> broadcast::Receiver<RugPullEvent> {
        self.event_sender.subscribe()
    }
}

#[derive(Debug, Serialize)]
pub struct MonitoringStatus {
    pub total_monitored_tokens: usize,
    pub total_clusters: usize,
    pub high_risk_tokens: usize,
    pub buffer_size: usize,
    pub last_analysis: DateTime<Utc>,
}


#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::timeout;

    #[tokio::test]
    async fn test_rug_pull_monitor_creation() {
        let (monitor, _receiver) = RugPullMonitor::new().unwrap();
        assert_eq!(monitor.buffer_size, 1000);
        assert_eq!(monitor.transaction_buffer.len(), 0);
    }

    #[tokio::test]
    async fn test_transaction_buffering() {
        let (mut monitor, _receiver) = RugPullMonitor::new().unwrap();
        
        let test_tx = Transaction {
            signature: "test_sig".to_string(),
            wallet: "test_wallet".to_string(),
            token_mint: "test_token".to_string(),
            amount_sol: 100.0,
            transaction_type: super::super::TransactionType::Buy,
            timestamp: Utc::now(),
        };

        monitor.add_to_buffer(test_tx).await.unwrap();
        assert_eq!(monitor.transaction_buffer.len(), 1);
    }

    #[tokio::test]
    async fn test_monitoring_status() {
        let (monitor, _receiver) = RugPullMonitor::new().unwrap();
        let status = monitor.get_monitoring_status().await.unwrap();
        
        assert_eq!(status.total_monitored_tokens, 0);
        assert_eq!(status.buffer_size, 0);
    }
}