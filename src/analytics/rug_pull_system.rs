//! Complete Rug Pull Detection System
//! 
//! This module provides a unified interface to the coordinated rug pull detection system,
//! integrating detection, monitoring, and alerting capabilities.

use anyhow::Result;
use tokio::sync::{broadcast, mpsc};
use tokio::task::JoinHandle;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

use super::{
    RugPullDetector, RugPullMonitor, AlertManager, AlertDeliveryService,
    RugPullAlert, RugPullEvent, MonitoringStatus, AlertConfig, AlertStats
};

/// Complete rug pull detection and alerting system
pub struct RugPullSystem {
    monitor_handle: Option<JoinHandle<Result<()>>>,
    alert_handle: Option<JoinHandle<Result<()>>>,
    event_receiver: broadcast::Receiver<RugPullEvent>,
    system_stats: SystemStats,
}

#[derive(Debug, Clone, Serialize)]
pub struct SystemStats {
    pub system_uptime: DateTime<Utc>,
    pub total_alerts_sent: u64,
    pub tokens_monitored: u64,
    pub last_analysis: Option<DateTime<Utc>>,
    pub system_status: SystemStatus,
}

#[derive(Debug, Clone, Serialize)]
pub enum SystemStatus {
    Starting,
    Running,
    Paused,
    Error(String),
    Stopped,
}

impl RugPullSystem {
    /// Initialize and start the complete rug pull detection system
    pub async fn start(alert_config: AlertConfig) -> Result<Self> {
        // Create the monitoring system
        let (mut monitor, event_receiver) = RugPullMonitor::new()?;
        
        // Create the alert system
        let (mut alert_manager, alert_receiver) = AlertManager::new(alert_config.clone());
        let alert_delivery = AlertDeliveryService::new(alert_config);

        // Subscribe to events for alert processing
        let mut monitor_events = monitor.subscribe();

        // Start the monitoring service
        let monitor_handle = tokio::spawn(async move {
            monitor.start_monitoring().await
        });

        // Start the alert processing service
        let alert_handle = tokio::spawn(async move {
            // Process events and generate alerts
            let alert_processor = tokio::spawn(async move {
                while let Ok(event) = monitor_events.recv().await {
                    if let Err(e) = alert_manager.handle_rug_pull_event(event).await {
                        eprintln!("Error processing rug pull event: {}", e);
                    }
                }
                Ok::<(), anyhow::Error>(())
            });

            // Deliver alerts
            let alert_delivery_task = tokio::spawn(async move {
                alert_delivery.process_alerts(alert_receiver).await
            });

            // Wait for both tasks
            let (processor_result, delivery_result) = tokio::join!(alert_processor, alert_delivery_task);
            processor_result??;
            delivery_result??;
            
            Ok(())
        });

        let system = Self {
            monitor_handle: Some(monitor_handle),
            alert_handle: Some(alert_handle),
            event_receiver,
            system_stats: SystemStats {
                system_uptime: Utc::now(),
                total_alerts_sent: 0,
                tokens_monitored: 0,
                last_analysis: None,
                system_status: SystemStatus::Running,
            },
        };

        Ok(system)
    }

    /// Get system health and statistics
    pub async fn get_system_health(&self) -> SystemHealth {
        let is_monitor_healthy = self.monitor_handle
            .as_ref()
            .map(|h| !h.is_finished())
            .unwrap_or(false);

        let is_alert_healthy = self.alert_handle
            .as_ref()
            .map(|h| !h.is_finished())
            .unwrap_or(false);

        let overall_status = if is_monitor_healthy && is_alert_healthy {
            SystemStatus::Running
        } else if !is_monitor_healthy {
            SystemStatus::Error("Monitor service stopped".to_string())
        } else if !is_alert_healthy {
            SystemStatus::Error("Alert service stopped".to_string())
        } else {
            SystemStatus::Stopped
        };

        SystemHealth {
            overall_status,
            monitor_healthy: is_monitor_healthy,
            alert_system_healthy: is_alert_healthy,
            uptime_seconds: (Utc::now() - self.system_stats.system_uptime).num_seconds(),
            stats: self.system_stats.clone(),
        }
    }

    /// Subscribe to rug pull events
    pub fn subscribe_to_events(&self) -> broadcast::Receiver<RugPullEvent> {
        self.event_receiver.resubscribe()
    }

    /// Gracefully shutdown the system
    pub async fn shutdown(mut self) -> Result<()> {
        // Cancel monitoring
        if let Some(handle) = self.monitor_handle.take() {
            handle.abort();
        }

        // Cancel alerting
        if let Some(handle) = self.alert_handle.take() {
            handle.abort();
        }

        Ok(())
    }
}

#[derive(Debug, Serialize)]
pub struct SystemHealth {
    pub overall_status: SystemStatus,
    pub monitor_healthy: bool,
    pub alert_system_healthy: bool,
    pub uptime_seconds: i64,
    pub stats: SystemStats,
}

/// Simplified API for basic rug pull detection without full monitoring
pub struct SimpleRugPullDetector {
    detector: RugPullDetector,
}

impl SimpleRugPullDetector {
    pub fn new() -> Self {
        Self {
            detector: RugPullDetector::new(),
        }
    }

    /// Analyze a batch of transactions for rug pull patterns
    pub async fn analyze_transactions(&mut self, transactions: &[super::Transaction]) -> Result<Vec<RugPullAlert>> {
        self.detector.analyze_transactions(transactions).await
    }

    /// Get summary of any active clusters
    pub fn get_active_clusters_summary(&self) -> ClustersSummary {
        let active_clusters = self.detector.get_active_clusters();
        
        let total_tokens = active_clusters.len();
        let total_clusters: usize = active_clusters.values().map(|v| v.len()).sum();
        
        let high_risk_count = active_clusters
            .values()
            .flatten()
            .filter(|cluster| matches!(cluster.risk_level, super::RiskLevel::High | super::RiskLevel::Critical))
            .count();

        ClustersSummary {
            total_tokens_monitored: total_tokens,
            total_clusters,
            high_risk_clusters: high_risk_count,
            last_updated: Utc::now(),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ClustersSummary {
    pub total_tokens_monitored: usize,
    pub total_clusters: usize,
    pub high_risk_clusters: usize,
    pub last_updated: DateTime<Utc>,
}

/// Configuration builder for the rug pull system
pub struct RugPullSystemBuilder {
    alert_config: AlertConfig,
}

impl RugPullSystemBuilder {
    pub fn new() -> Self {
        Self {
            alert_config: AlertConfig::default(),
        }
    }

    pub fn with_telegram(mut self, chat_id: String) -> Self {
        self.alert_config.telegram_enabled = true;
        self.alert_config.telegram_chat_id = Some(chat_id);
        self
    }

    pub fn with_discord(mut self, webhook_url: String) -> Self {
        self.alert_config.discord_enabled = true;
        self.alert_config.discord_webhook_url = Some(webhook_url);
        self
    }

    pub fn with_email(mut self, recipients: Vec<String>) -> Self {
        self.alert_config.email_enabled = true;
        self.alert_config.email_recipients = recipients;
        self
    }

    pub fn with_risk_threshold(mut self, threshold: f64) -> Self {
        self.alert_config.min_risk_threshold = threshold;
        self
    }

    pub fn with_cooldown_minutes(mut self, minutes: i64) -> Self {
        self.alert_config.alert_cooldown_minutes = minutes;
        self
    }

    pub async fn build(self) -> Result<RugPullSystem> {
        RugPullSystem::start(self.alert_config).await
    }
}

impl Default for RugPullSystemBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_simple_rug_pull_detector() {
        let mut detector = SimpleRugPullDetector::new();
        let summary = detector.get_active_clusters_summary();
        
        assert_eq!(summary.total_tokens_monitored, 0);
        assert_eq!(summary.total_clusters, 0);
        assert_eq!(summary.high_risk_clusters, 0);
    }

    #[test]
    fn test_system_builder() {
        let builder = RugPullSystemBuilder::new()
            .with_telegram("123456789".to_string())
            .with_discord("https://discord.com/webhook".to_string())
            .with_risk_threshold(0.8)
            .with_cooldown_minutes(30);

        assert_eq!(builder.alert_config.min_risk_threshold, 0.8);
        assert_eq!(builder.alert_config.alert_cooldown_minutes, 30);
        assert!(builder.alert_config.telegram_enabled);
        assert!(builder.alert_config.discord_enabled);
    }

    // Integration test would require running services
    #[ignore]
    #[tokio::test]
    async fn test_full_system_integration() {
        let system = RugPullSystemBuilder::new()
            .with_risk_threshold(0.5)
            .build()
            .await
            .unwrap();

        let health = system.get_system_health().await;
        assert!(matches!(health.overall_status, SystemStatus::Running));
        
        // Cleanup
        system.shutdown().await.unwrap();
    }
}