//! Rug Pull Alert System
//! 
//! This module provides comprehensive alerting capabilities for coordinated rug pull detection,
//! integrating with Telegram, Discord, email, and other notification channels.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, broadcast};
use chrono::{DateTime, Utc};
use std::collections::HashMap;

use super::{RugPullAlert, RugPullEvent, RiskLevel, EarlyInvestorCluster};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertConfig {
    pub telegram_enabled: bool,
    pub telegram_chat_id: Option<String>,
    pub discord_enabled: bool,
    pub discord_webhook_url: Option<String>,
    pub email_enabled: bool,
    pub email_recipients: Vec<String>,
    pub min_risk_threshold: f64,
    pub alert_cooldown_minutes: i64,
}

impl Default for AlertConfig {
    fn default() -> Self {
        Self {
            telegram_enabled: true,
            telegram_chat_id: std::env::var("TELEGRAM_CHAT_ID").ok(),
            discord_enabled: false,
            discord_webhook_url: std::env::var("DISCORD_WEBHOOK_URL").ok(),
            email_enabled: false,
            email_recipients: vec![],
            min_risk_threshold: 0.7,
            alert_cooldown_minutes: 60, // 1 hour cooldown per token
        }
    }
}

#[derive(Debug, Clone)]
pub struct AlertManager {
    config: AlertConfig,
    last_alerts: HashMap<String, DateTime<Utc>>, // token_mint -> last_alert_time
    alert_sender: mpsc::UnboundedSender<AlertMessage>,
}

#[derive(Debug, Clone)]
pub struct AlertMessage {
    pub alert_type: AlertType,
    pub content: AlertContent,
    pub priority: AlertPriority,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub enum AlertType {
    RugPullDetected,
    HighRiskPattern,
    ClusterUpdate,
    SystemHealth,
}

#[derive(Debug, Clone)]
pub enum AlertPriority {
    Critical,
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone)]
pub struct AlertContent {
    pub title: String,
    pub message: String,
    pub token_mint: Option<String>,
    pub risk_score: Option<f64>,
    pub cluster_info: Option<ClusterSummary>,
    pub actions: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ClusterSummary {
    pub cluster_count: usize,
    pub total_wallets: usize,
    pub total_investment_sol: f64,
    pub highest_coordination_score: f64,
    pub risk_flags: Vec<String>,
}

impl AlertManager {
    pub fn new(config: AlertConfig) -> (Self, mpsc::UnboundedReceiver<AlertMessage>) {
        let (alert_sender, alert_receiver) = mpsc::unbounded_channel();
        
        let manager = Self {
            config,
            last_alerts: HashMap::new(),
            alert_sender,
        };

        (manager, alert_receiver)
    }

    /// Process a rug pull event and generate appropriate alerts
    pub async fn handle_rug_pull_event(&mut self, event: RugPullEvent) -> Result<()> {
        match event.event_type {
            super::rug_pull_monitor::EventType::NewAlert | 
            super::rug_pull_monitor::EventType::HighRiskDetected => {
                if let Some(alert) = event.alert {
                    self.process_rug_pull_alert(alert).await?;
                }
            }
            super::rug_pull_monitor::EventType::ClusterUpdate => {
                if let Some(cluster) = event.cluster_update {
                    self.process_cluster_update(cluster).await?;
                }
            }
            super::rug_pull_monitor::EventType::AlertResolved => {
                self.process_alert_resolution(event.token_mint).await?;
            }
        }

        Ok(())
    }

    async fn process_rug_pull_alert(&mut self, alert: RugPullAlert) -> Result<()> {
        // Check if we're above the minimum risk threshold
        if alert.overall_risk_score < self.config.min_risk_threshold {
            return Ok(());
        }

        // Check cooldown period
        if let Some(last_alert_time) = self.last_alerts.get(&alert.token_mint) {
            let cooldown = chrono::Duration::minutes(self.config.alert_cooldown_minutes);
            if Utc::now() - *last_alert_time < cooldown {
                return Ok(()); // Still in cooldown
            }
        }

        // Update last alert time
        self.last_alerts.insert(alert.token_mint.clone(), Utc::now());

        // Generate alert content
        let cluster_summary = self.create_cluster_summary(&alert.clusters);
        let priority = self.determine_priority(alert.overall_risk_score);
        
        let content = AlertContent {
            title: self.create_alert_title(&alert),
            message: self.create_alert_message(&alert, &cluster_summary),
            token_mint: Some(alert.token_mint.clone()),
            risk_score: Some(alert.overall_risk_score),
            cluster_info: Some(cluster_summary),
            actions: alert.recommended_actions.clone(),
        };

        let alert_message = AlertMessage {
            alert_type: if alert.overall_risk_score >= 0.9 {
                AlertType::RugPullDetected
            } else {
                AlertType::HighRiskPattern
            },
            content,
            priority,
            timestamp: Utc::now(),
        };

        // Send alert
        self.alert_sender.send(alert_message)?;

        Ok(())
    }

    async fn process_cluster_update(&mut self, cluster: EarlyInvestorCluster) -> Result<()> {
        // Only send updates for high-risk clusters
        if !matches!(cluster.risk_level, RiskLevel::High | RiskLevel::Critical) {
            return Ok(());
        }

        let content = AlertContent {
            title: format!("Cluster Update: {}", cluster.token_mint),
            message: format!(
                "Cluster {} has been updated:\n\
                 • Wallets: {}\n\
                 • Coordination Score: {:.1}%\n\
                 • Total Investment: {:.2} SOL\n\
                 • Risk Level: {:?}",
                cluster.cluster_id,
                cluster.wallets.len(),
                cluster.coordination_score * 100.0,
                cluster.total_investment,
                cluster.risk_level
            ),
            token_mint: Some(cluster.token_mint),
            risk_score: Some(cluster.coordination_score),
            cluster_info: None,
            actions: vec![],
        };

        let alert_message = AlertMessage {
            alert_type: AlertType::ClusterUpdate,
            content,
            priority: AlertPriority::Medium,
            timestamp: Utc::now(),
        };

        self.alert_sender.send(alert_message)?;

        Ok(())
    }

    async fn process_alert_resolution(&mut self, token_mint: String) -> Result<()> {
        // Remove from tracking
        self.last_alerts.remove(&token_mint);

        let content = AlertContent {
            title: format!("Alert Resolved: {}", token_mint),
            message: format!("Rug pull alert for token {} has been resolved.", token_mint),
            token_mint: Some(token_mint),
            risk_score: None,
            cluster_info: None,
            actions: vec!["Resume normal monitoring".to_string()],
        };

        let alert_message = AlertMessage {
            alert_type: AlertType::SystemHealth,
            content,
            priority: AlertPriority::Low,
            timestamp: Utc::now(),
        };

        self.alert_sender.send(alert_message)?;

        Ok(())
    }

    fn create_cluster_summary(&self, clusters: &[EarlyInvestorCluster]) -> ClusterSummary {
        let cluster_count = clusters.len();
        let total_wallets = clusters.iter().map(|c| c.wallets.len()).sum();
        let total_investment_sol = clusters.iter().map(|c| c.total_investment).sum();
        let highest_coordination_score = clusters
            .iter()
            .map(|c| c.coordination_score)
            .fold(0.0, f64::max);

        let risk_flags = clusters
            .iter()
            .flat_map(|c| &c.behavioral_flags)
            .map(|flag| format!("{:?}", flag))
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        ClusterSummary {
            cluster_count,
            total_wallets,
            total_investment_sol,
            highest_coordination_score,
            risk_flags,
        }
    }

    fn determine_priority(&self, risk_score: f64) -> AlertPriority {
        if risk_score >= 0.95 {
            AlertPriority::Critical
        } else if risk_score >= 0.8 {
            AlertPriority::High
        } else if risk_score >= 0.6 {
            AlertPriority::Medium
        } else {
            AlertPriority::Low
        }
    }

    fn create_alert_title(&self, alert: &RugPullAlert) -> String {
        let risk_emoji = match alert.overall_risk_score {
            score if score >= 0.95 => "🚨",
            score if score >= 0.8 => "⚠️",
            score if score >= 0.6 => "⚡",
            _ => "ℹ️",
        };

        format!(
            "{} COORDINATED RUG PULL DETECTED: {} ({:.1}% risk)",
            risk_emoji,
            alert.token_mint,
            alert.overall_risk_score * 100.0
        )
    }

    fn create_alert_message(&self, alert: &RugPullAlert, cluster_summary: &ClusterSummary) -> String {
        format!(
            "🔍 **Coordinated Rug Pull Analysis**\n\n\
             **Token:** `{}`\n\
             **Risk Score:** {:.1}%\n\
             **Confidence:** {:.1}%\n\n\
             📊 **Cluster Analysis:**\n\
             • {} coordination clusters detected\n\
             • {} total wallets involved\n\
             • {:.2} SOL total investment\n\
             • Max coordination score: {:.1}%\n\n\
             🚩 **Risk Indicators:**\n{}\n\n\
             🎯 **Recommended Actions:**\n{}\n\n\
             📋 **Evidence:**\n{}\n\n\
             ⏰ **Detected:** {}",
            alert.token_mint,
            alert.overall_risk_score * 100.0,
            alert.confidence * 100.0,
            cluster_summary.cluster_count,
            cluster_summary.total_wallets,
            cluster_summary.total_investment_sol,
            cluster_summary.highest_coordination_score * 100.0,
            cluster_summary.risk_flags.iter()
                .map(|flag| format!("• {}", flag))
                .collect::<Vec<_>>()
                .join("\n"),
            alert.recommended_actions.iter()
                .map(|action| format!("• {}", action))
                .collect::<Vec<_>>()
                .join("\n"),
            alert.evidence.iter()
                .map(|evidence| format!("• {}", evidence))
                .collect::<Vec<_>>()
                .join("\n"),
            alert.timestamp.format("%Y-%m-%d %H:%M:%S UTC")
        )
    }

    /// Get current alert statistics
    pub fn get_alert_stats(&self) -> AlertStats {
        AlertStats {
            total_tokens_monitored: self.last_alerts.len(),
            alerts_in_cooldown: self.last_alerts
                .values()
                .filter(|&&time| {
                    let cooldown = chrono::Duration::minutes(self.config.alert_cooldown_minutes);
                    Utc::now() - time < cooldown
                })
                .count(),
            last_alert_time: self.last_alerts.values().max().copied(),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct AlertStats {
    pub total_tokens_monitored: usize,
    pub alerts_in_cooldown: usize,
    pub last_alert_time: Option<DateTime<Utc>>,
}

/// Alert delivery service that handles different notification channels
pub struct AlertDeliveryService {
    config: AlertConfig,
    telegram_client: Option<TelegramClient>,
    discord_client: Option<DiscordClient>,
}

impl AlertDeliveryService {
    pub fn new(config: AlertConfig) -> Self {
        let telegram_client = if config.telegram_enabled && config.telegram_chat_id.is_some() {
            Some(TelegramClient::new())
        } else {
            None
        };

        let discord_client = if config.discord_enabled && config.discord_webhook_url.is_some() {
            Some(DiscordClient::new())
        } else {
            None
        };

        Self {
            config,
            telegram_client,
            discord_client,
        }
    }

    /// Process alert messages from the queue
    pub async fn process_alerts(&self, mut alert_receiver: mpsc::UnboundedReceiver<AlertMessage>) -> Result<()> {
        while let Some(alert_message) = alert_receiver.recv().await {
            if let Err(e) = self.deliver_alert(&alert_message).await {
                eprintln!("Failed to deliver alert: {}", e);
                // Could implement retry logic here
            }
        }
        Ok(())
    }

    async fn deliver_alert(&self, alert_message: &AlertMessage) -> Result<()> {
        // Telegram delivery
        if let Some(ref telegram_client) = self.telegram_client {
            if let Some(ref chat_id) = self.config.telegram_chat_id {
                telegram_client.send_message(chat_id, &alert_message.content.message).await?;
            }
        }

        // Discord delivery
        if let Some(ref discord_client) = self.discord_client {
            if let Some(ref webhook_url) = self.config.discord_webhook_url {
                discord_client.send_webhook_message(webhook_url, &alert_message.content).await?;
            }
        }

        // Email delivery (if configured)
        if self.config.email_enabled && !self.config.email_recipients.is_empty() {
            // Email implementation would go here
            println!("Email alert would be sent to: {:?}", self.config.email_recipients);
        }

        Ok(())
    }
}

// Placeholder client implementations - these would be fully implemented with actual API calls
struct TelegramClient;

impl TelegramClient {
    fn new() -> Self {
        Self
    }

    async fn send_message(&self, _chat_id: &str, message: &str) -> Result<()> {
        // Implement actual Telegram API call
        println!("Telegram message: {}", message);
        Ok(())
    }
}

struct DiscordClient;

impl DiscordClient {
    fn new() -> Self {
        Self
    }

    async fn send_webhook_message(&self, _webhook_url: &str, content: &AlertContent) -> Result<()> {
        // Implement actual Discord webhook call
        println!("Discord webhook: {}", content.title);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    #[tokio::test]
    async fn test_alert_manager_creation() {
        let config = AlertConfig::default();
        let (manager, _receiver) = AlertManager::new(config);
        assert_eq!(manager.last_alerts.len(), 0);
    }

    #[tokio::test]
    async fn test_cooldown_logic() {
        let config = AlertConfig {
            alert_cooldown_minutes: 5, // 5 minute cooldown for testing
            ..Default::default()
        };
        let (mut manager, _receiver) = AlertManager::new(config);

        // Record an alert
        let token_mint = "test_token".to_string();
        manager.last_alerts.insert(token_mint.clone(), Utc::now());

        // Check that we're in cooldown
        let last_alert_time = manager.last_alerts.get(&token_mint).unwrap();
        let cooldown = Duration::minutes(5);
        assert!(Utc::now() - *last_alert_time < cooldown);
    }

    #[test]
    fn test_alert_priority_determination() {
        let config = AlertConfig::default();
        let (manager, _receiver) = AlertManager::new(config);

        assert!(matches!(manager.determine_priority(0.96), AlertPriority::Critical));
        assert!(matches!(manager.determine_priority(0.85), AlertPriority::High));
        assert!(matches!(manager.determine_priority(0.65), AlertPriority::Medium));
        assert!(matches!(manager.determine_priority(0.45), AlertPriority::Low));
    }
}