use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock, broadcast};
use tracing::{info, warn, error, debug};

use crate::infrastructure::{
    solana_rpc::{SolanaRpc, SolanaRpcConfig, SolanaHealthStatus},
    transaction_monitor::{TransactionMonitor, MonitorConfig, TransactionEvent},
};
use crate::analytics::rug_pull_detector::RugPullDetector;
use crate::ai_engines::AIEngineManager;

#[derive(Debug, Clone)]
pub struct BlockchainServiceConfig {
    pub solana_config: SolanaRpcConfig,
    pub monitor_config: MonitorConfig,
    pub enable_ai_analysis: bool,
    pub analysis_batch_size: usize,
}

impl Default for BlockchainServiceConfig {
    fn default() -> Self {
        Self {
            solana_config: SolanaRpcConfig::default(),
            monitor_config: MonitorConfig::default(),
            enable_ai_analysis: bool::default(),
            analysis_batch_size: 50,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    pub transaction_signature: String,
    pub rug_pull_risk: f64,
    pub confidence: f64,
    pub patterns_detected: Vec<String>,
    pub recommendation: TradingRecommendation,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TradingRecommendation {
    Hold,
    Buy { confidence: f64 },
    Sell { confidence: f64 },
    AvoidToken { reason: String },
    CounterAttack { strategy: String },
}

pub struct BlockchainService {
    config: BlockchainServiceConfig,
    solana_rpc: Arc<SolanaRpc>,
    transaction_monitor: Arc<TransactionMonitor>,
    rug_pull_detector: Arc<RugPullDetector>,
    ai_engine_manager: Arc<AIEngineManager>,
    analysis_results: Arc<RwLock<Vec<AnalysisResult>>>,
    event_sender: broadcast::Sender<AnalysisResult>,
}

impl BlockchainService {
    pub async fn new(config: BlockchainServiceConfig) -> Result<Self> {
        // Initialize Solana RPC
        let solana_rpc = Arc::new(SolanaRpc::new(config.solana_config.clone()));
        
        // Test connection
        match solana_rpc.health_check().await? {
            SolanaHealthStatus::Healthy { .. } => {
                info!("Solana RPC connection established successfully");
            }
            SolanaHealthStatus::Slow { .. } => {
                warn!("Solana RPC connection is slow but functional");
            }
            SolanaHealthStatus::Unhealthy { error } => {
                return Err(anyhow!("Solana RPC unhealthy: {}", error));
            }
        }

        // Initialize transaction monitor
        let transaction_monitor = Arc::new(TransactionMonitor::new(
            config.monitor_config.clone(),
            Arc::clone(&solana_rpc),
        ));

        // Initialize rug pull detector
        let rug_pull_detector = Arc::new(RugPullDetector::new());

        // Initialize AI engine manager
        let ai_engine_manager = Arc::new(AIEngineManager::new());

        let (event_sender, _) = broadcast::channel(1000);

        Ok(Self {
            config,
            solana_rpc,
            transaction_monitor,
            rug_pull_detector,
            ai_engine_manager,
            analysis_results: Arc::new(RwLock::new(Vec::new())),
            event_sender,
        })
    }

    pub async fn start(&self) -> Result<()> {
        info!("Starting Blockchain Service");

        // Subscribe to transaction events
        let mut event_receiver = self.transaction_monitor
            .subscribe("blockchain_service".to_string())
            .await?;

        // Start transaction monitor
        let monitor_task = {
            let monitor = Arc::clone(&self.transaction_monitor);
            tokio::spawn(async move {
                if let Err(e) = monitor.start().await {
                    error!("Transaction monitor failed: {}", e);
                }
            })
        };

        // Start analysis task
        let analysis_task = {
            let service = self.clone_for_task();
            tokio::spawn(async move {
                service.run_analysis_loop(event_receiver).await;
            })
        };

        // Wait for tasks to complete
        tokio::select! {
            _ = monitor_task => {
                warn!("Transaction monitor task completed");
            }
            _ = analysis_task => {
                warn!("Analysis task completed");
            }
        }

        Ok(())
    }

    fn clone_for_task(&self) -> BlockchainServiceTask {
        BlockchainServiceTask {
            config: self.config.clone(),
            rug_pull_detector: Arc::clone(&self.rug_pull_detector),
            ai_engine_manager: Arc::clone(&self.ai_engine_manager),
            analysis_results: Arc::clone(&self.analysis_results),
            event_sender: self.event_sender.clone(),
        }
    }

    pub async fn get_health_status(&self) -> Result<ServiceHealthStatus> {
        let solana_health = self.solana_rpc.health_check().await?;
        let monitor_stats = self.transaction_monitor.get_statistics().await?;
        
        let analysis_count = {
            let results = self.analysis_results.read().await;
            results.len()
        };

        Ok(ServiceHealthStatus {
            solana_rpc: solana_health,
            monitor_running: monitor_stats.is_running,
            total_events_processed: monitor_stats.total_events,
            analysis_results_count: analysis_count,
            ai_engines_active: self.config.enable_ai_analysis,
        })
    }

    pub async fn subscribe_to_analysis(&self) -> broadcast::Receiver<AnalysisResult> {
        self.event_sender.subscribe()
    }

    pub async fn get_recent_analysis(&self, limit: Option<usize>) -> Result<Vec<AnalysisResult>> {
        let results = self.analysis_results.read().await;
        let limit = limit.unwrap_or(100).min(results.len());
        Ok(results.iter().rev().take(limit).cloned().collect())
    }

    pub async fn analyze_transaction(&self, signature: &str) -> Result<AnalysisResult> {
        // Get transaction details
        let transaction = self.solana_rpc.get_transaction(signature).await?;
        
        // Perform rug pull analysis
        let rug_pull_risk = self.rug_pull_detector.analyze_transaction(&transaction).await?;
        
        // Generate recommendation
        let recommendation = if rug_pull_risk > 0.8 {
            TradingRecommendation::CounterAttack {
                strategy: "SCAMMER GET SCAMMED - Deploy counter-rug strategy".to_string()
            }
        } else if rug_pull_risk > 0.6 {
            TradingRecommendation::AvoidToken {
                reason: "High rug pull risk detected".to_string()
            }
        } else if rug_pull_risk < 0.3 {
            TradingRecommendation::Buy { confidence: 1.0 - rug_pull_risk }
        } else {
            TradingRecommendation::Hold
        };

        let result = AnalysisResult {
            transaction_signature: signature.to_string(),
            rug_pull_risk,
            confidence: 0.95, // Static for now
            patterns_detected: vec!["early_investor_clustering".to_string()],
            recommendation,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };

        Ok(result)
    }
}

#[derive(Clone)]
struct BlockchainServiceTask {
    config: BlockchainServiceConfig,
    rug_pull_detector: Arc<RugPullDetector>,
    ai_engine_manager: Arc<AIEngineManager>,
    analysis_results: Arc<RwLock<Vec<AnalysisResult>>>,
    event_sender: broadcast::Sender<AnalysisResult>,
}

impl BlockchainServiceTask {
    async fn run_analysis_loop(&self, mut event_receiver: broadcast::Receiver<TransactionEvent>) {
        info!("Starting analysis loop");
        let mut batch = Vec::new();

        while let Ok(event) = event_receiver.recv().await {
            batch.push(event);

            if batch.len() >= self.config.analysis_batch_size {
                self.process_batch(batch.drain(..).collect()).await;
            }
        }

        // Process remaining batch
        if !batch.is_empty() {
            self.process_batch(batch).await;
        }
    }

    async fn process_batch(&self, events: Vec<TransactionEvent>) {
        if !self.config.enable_ai_analysis {
            return;
        }

        debug!("Processing batch of {} events", events.len());

        for event in events {
            match self.analyze_event(&event).await {
                Ok(result) => {
                    // Store result
                    {
                        let mut results = self.analysis_results.write().await;
                        results.push(result.clone());
                        
                        // Keep only recent results
                        if results.len() > 10000 {
                            results.drain(0..1000);
                        }
                    }

                    // Notify subscribers
                    if let Err(e) = self.event_sender.send(result) {
                        debug!("Failed to send analysis result: {}", e);
                    }
                }
                Err(e) => {
                    error!("Failed to analyze event {}: {}", event.signature, e);
                }
            }
        }
    }

    async fn analyze_event(&self, event: &TransactionEvent) -> Result<AnalysisResult> {
        // Simulate analysis - in production this would use the actual AI engines
        let rug_pull_risk = fastrand::f64() * 0.5; // Random risk between 0-0.5 for demo
        
        let recommendation = match rug_pull_risk {
            r if r > 0.4 => TradingRecommendation::AvoidToken {
                reason: "Suspicious pattern detected".to_string()
            },
            r if r < 0.2 => TradingRecommendation::Buy { confidence: 1.0 - r },
            _ => TradingRecommendation::Hold,
        };

        Ok(AnalysisResult {
            transaction_signature: event.signature.clone(),
            rug_pull_risk,
            confidence: 0.90,
            patterns_detected: vec!["swap_pattern".to_string()],
            recommendation,
            timestamp: event.timestamp,
        })
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ServiceHealthStatus {
    pub solana_rpc: SolanaHealthStatus,
    pub monitor_running: bool,
    pub total_events_processed: usize,
    pub analysis_results_count: usize,
    pub ai_engines_active: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_blockchain_service_creation() {
        let config = BlockchainServiceConfig {
            solana_config: SolanaRpcConfig {
                rpc_url: "https://api.devnet.solana.com".to_string(),
                ..Default::default()
            },
            ..Default::default()
        };
        
        let service = BlockchainService::new(config).await;
        assert!(service.is_ok());
    }

    #[tokio::test]
    async fn test_health_status() {
        let config = BlockchainServiceConfig {
            solana_config: SolanaRpcConfig {
                rpc_url: "https://api.devnet.solana.com".to_string(),
                ..Default::default()
            },
            ..Default::default()
        };
        
        let service = BlockchainService::new(config).await.unwrap();
        let health = service.get_health_status().await.unwrap();
        assert!(!health.ai_engines_active); // Default is false
    }
}