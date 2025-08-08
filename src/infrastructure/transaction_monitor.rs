use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::{mpsc, RwLock, broadcast};
use tokio::time::{interval, sleep};
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tracing::{info, warn, error, debug};

use crate::infrastructure::solana_rpc::{SolanaRpc, SolanaRpcConfig};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionEvent {
    pub signature: String,
    pub slot: u64,
    pub block_time: Option<i64>,
    pub fee: Option<u64>,
    pub accounts: Vec<String>,
    pub program_ids: Vec<String>,
    pub success: bool,
    pub event_type: TransactionEventType,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransactionEventType {
    TokenTransfer { 
        from: String, 
        to: String, 
        amount: u64, 
        token_mint: String 
    },
    LiquidityAdd { 
        pool: String, 
        token_a: String, 
        token_b: String, 
        amount_a: u64, 
        amount_b: u64 
    },
    LiquidityRemove { 
        pool: String, 
        token_a: String, 
        token_b: String, 
        amount_a: u64, 
        amount_b: u64 
    },
    Swap { 
        pool: String, 
        token_in: String, 
        token_out: String, 
        amount_in: u64, 
        amount_out: u64 
    },
    ProgramInvocation { 
        program_id: String, 
        instruction_data: Vec<u8> 
    },
    Unknown,
}

#[derive(Debug, Clone)]
pub struct MonitorConfig {
    pub websocket_url: String,
    pub programs_to_monitor: Vec<String>,
    pub accounts_to_monitor: Vec<String>,
    pub batch_size: usize,
    pub buffer_size: usize,
    pub health_check_interval: Duration,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            websocket_url: "wss://api.mainnet-beta.solana.com".to_string(),
            programs_to_monitor: vec![
                "9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM".to_string(), // Raydium
                "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8".to_string(), // Raydium AMM
                "5Q544fKrFoe6tsEbD7S8EmxGTJYAKtTVhAW5Q5pge4j1".to_string(), // Raydium Liquidity
            ],
            accounts_to_monitor: vec![],
            batch_size: 100,
            buffer_size: 10000,
            health_check_interval: Duration::from_secs(30),
        }
    }
}

#[derive(Debug)]
pub struct TransactionMonitor {
    config: MonitorConfig,
    solana_rpc: Arc<SolanaRpc>,
    event_buffer: Arc<RwLock<VecDeque<TransactionEvent>>>,
    subscribers: Arc<RwLock<HashMap<String, broadcast::Sender<TransactionEvent>>>>,
    is_running: Arc<RwLock<bool>>,
}

impl TransactionMonitor {
    pub fn new(config: MonitorConfig, solana_rpc: Arc<SolanaRpc>) -> Self {
        Self {
            config,
            solana_rpc,
            event_buffer: Arc::new(RwLock::new(VecDeque::new())),
            subscribers: Arc::new(RwLock::new(HashMap::new())),
            is_running: Arc::new(RwLock::new(false)),
        }
    }

    pub async fn start(&self) -> Result<()> {
        {
            let mut running = self.is_running.write().await;
            if *running {
                return Err(anyhow!("Transaction monitor already running"));
            }
            *running = true;
        }

        info!("Starting transaction monitor with {} programs", self.config.programs_to_monitor.len());

        // Start WebSocket subscription
        let ws_task = self.start_websocket_monitor();
        
        // Start health check task
        let health_task = self.start_health_check();
        
        // Start buffer maintenance task
        let buffer_task = self.start_buffer_maintenance();

        // Wait for all tasks
        tokio::select! {
            result = ws_task => {
                error!("WebSocket monitor stopped: {:?}", result);
                result
            }
            result = health_task => {
                error!("Health check stopped: {:?}", result);
                result
            }
            result = buffer_task => {
                error!("Buffer maintenance stopped: {:?}", result);
                result
            }
        }
    }

    pub async fn stop(&self) -> Result<()> {
        let mut running = self.is_running.write().await;
        *running = false;
        info!("Stopping transaction monitor");
        Ok(())
    }

    pub async fn subscribe(&self, subscriber_id: String) -> Result<broadcast::Receiver<TransactionEvent>> {
        let (tx, rx) = broadcast::channel(1000);
        let mut subscribers = self.subscribers.write().await;
        subscribers.insert(subscriber_id, tx);
        Ok(rx)
    }

    pub async fn unsubscribe(&self, subscriber_id: &str) -> Result<()> {
        let mut subscribers = self.subscribers.write().await;
        subscribers.remove(subscriber_id);
        Ok(())
    }

    pub async fn get_recent_events(&self, limit: Option<usize>) -> Result<Vec<TransactionEvent>> {
        let buffer = self.event_buffer.read().await;
        let limit = limit.unwrap_or(100).min(buffer.len());
        
        Ok(buffer.iter().rev().take(limit).cloned().collect())
    }

    async fn start_websocket_monitor(&self) -> Result<()> {
        let url = &self.config.websocket_url;
        let (ws_stream, _) = connect_async(url).await?;
        info!("Connected to WebSocket: {}", url);

        // Subscribe to program account updates
        for program_id in &self.config.programs_to_monitor {
            let subscribe_msg = serde_json::json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "programSubscribe",
                "params": [
                    program_id,
                    {
                        "encoding": "jsonParsed",
                        "commitment": "confirmed"
                    }
                ]
            });

            // In a real implementation, you'd send this message to the WebSocket
            debug!("Subscribing to program: {}", program_id);
        }

        // Simulate receiving transaction events
        // In production, this would parse actual WebSocket messages
        self.simulate_transaction_stream().await
    }

    async fn simulate_transaction_stream(&self) -> Result<()> {
        let mut interval = interval(Duration::from_millis(500));
        
        while *self.is_running.read().await {
            interval.tick().await;
            
            // Simulate a transaction event
            let event = TransactionEvent {
                signature: format!("sig_{}", SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()),
                slot: self.solana_rpc.get_slot().await.unwrap_or(0),
                block_time: Some(SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as i64),
                fee: Some(5000),
                accounts: vec!["11111111111111111111111111111112".to_string()],
                program_ids: vec!["9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM".to_string()],
                success: true,
                event_type: TransactionEventType::Swap {
                    pool: "pool123".to_string(),
                    token_in: "SOL".to_string(),
                    token_out: "USDC".to_string(),
                    amount_in: 1000000000, // 1 SOL
                    amount_out: 100000000, // 100 USDC
                },
                timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            };

            self.process_event(event).await?;
        }

        Ok(())
    }

    async fn process_event(&self, event: TransactionEvent) -> Result<()> {
        // Add to buffer
        {
            let mut buffer = self.event_buffer.write().await;
            buffer.push_back(event.clone());
            
            // Maintain buffer size
            while buffer.len() > self.config.buffer_size {
                buffer.pop_front();
            }
        }

        // Notify subscribers
        {
            let subscribers = self.subscribers.read().await;
            for (subscriber_id, sender) in subscribers.iter() {
                if let Err(e) = sender.send(event.clone()) {
                    debug!("Failed to send event to subscriber {}: {}", subscriber_id, e);
                }
            }
        }

        debug!("Processed transaction event: {}", event.signature);
        Ok(())
    }

    async fn start_health_check(&self) -> Result<()> {
        let mut interval = interval(self.config.health_check_interval);
        
        while *self.is_running.read().await {
            interval.tick().await;
            
            match self.solana_rpc.health_check().await {
                Ok(health) => {
                    if !health.is_healthy() {
                        warn!("Solana RPC health check warning: {:?}", health);
                    } else {
                        debug!("Solana RPC health check passed: {:?}", health);
                    }
                }
                Err(e) => {
                    error!("Solana RPC health check failed: {}", e);
                }
            }
        }

        Ok(())
    }

    async fn start_buffer_maintenance(&self) -> Result<()> {
        let mut interval = interval(Duration::from_secs(60));
        
        while *self.is_running.read().await {
            interval.tick().await;
            
            let buffer_size = {
                let buffer = self.event_buffer.read().await;
                buffer.len()
            };

            let subscriber_count = {
                let subscribers = self.subscribers.read().await;
                subscribers.len()
            };

            info!(
                "Buffer maintenance - Events: {}, Subscribers: {}", 
                buffer_size, 
                subscriber_count
            );
        }

        Ok(())
    }

    pub async fn get_statistics(&self) -> Result<MonitorStatistics> {
        let buffer = self.event_buffer.read().await;
        let subscribers = self.subscribers.read().await;
        
        let recent_events = buffer.iter().rev().take(1000);
        let mut event_type_counts = HashMap::new();
        let mut success_count = 0;
        let mut failure_count = 0;

        for event in recent_events {
            let event_type_key = match &event.event_type {
                TransactionEventType::TokenTransfer { .. } => "token_transfer",
                TransactionEventType::LiquidityAdd { .. } => "liquidity_add",
                TransactionEventType::LiquidityRemove { .. } => "liquidity_remove",
                TransactionEventType::Swap { .. } => "swap",
                TransactionEventType::ProgramInvocation { .. } => "program_invocation",
                TransactionEventType::Unknown => "unknown",
            };
            
            *event_type_counts.entry(event_type_key.to_string()).or_insert(0) += 1;
            
            if event.success {
                success_count += 1;
            } else {
                failure_count += 1;
            }
        }

        Ok(MonitorStatistics {
            total_events: buffer.len(),
            active_subscribers: subscribers.len(),
            event_type_counts,
            success_rate: if success_count + failure_count > 0 {
                success_count as f64 / (success_count + failure_count) as f64
            } else {
                0.0
            },
            is_running: *self.is_running.read().await,
        })
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MonitorStatistics {
    pub total_events: usize,
    pub active_subscribers: usize,
    pub event_type_counts: HashMap<String, u32>,
    pub success_rate: f64,
    pub is_running: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infrastructure::solana_rpc::SolanaRpcConfig;

    #[tokio::test]
    async fn test_transaction_monitor_creation() {
        let config = MonitorConfig::default();
        let solana_rpc = Arc::new(SolanaRpc::new(SolanaRpcConfig::default()));
        let monitor = TransactionMonitor::new(config, solana_rpc);
        
        let stats = monitor.get_statistics().await.unwrap();
        assert_eq!(stats.total_events, 0);
        assert!(!stats.is_running);
    }

    #[tokio::test]
    async fn test_subscribe_unsubscribe() {
        let config = MonitorConfig::default();
        let solana_rpc = Arc::new(SolanaRpc::new(SolanaRpcConfig::default()));
        let monitor = TransactionMonitor::new(config, solana_rpc);
        
        let rx = monitor.subscribe("test_subscriber".to_string()).await.unwrap();
        let stats = monitor.get_statistics().await.unwrap();
        assert_eq!(stats.active_subscribers, 1);
        
        monitor.unsubscribe("test_subscriber").await.unwrap();
        let stats = monitor.get_statistics().await.unwrap();
        assert_eq!(stats.active_subscribers, 0);
    }
}