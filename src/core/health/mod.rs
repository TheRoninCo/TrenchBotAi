//! Health monitoring and alerting system
use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthStatus {
    Healthy,
    Warning { message: String },
    Critical { message: String },
    Down { message: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheck {
    pub name: String,
    pub status: HealthStatus,
    pub last_check: DateTime<Utc>,
    pub check_interval_seconds: u64,
}

pub struct HealthMonitor {
    checks: HashMap<String, HealthCheck>,
}

impl HealthMonitor {
    pub fn new() -> Self {
        let mut monitor = Self {
            checks: HashMap::new(),
        };
        
        // Add default health checks
        monitor.add_check("rpc_connection", 30);
        monitor.add_check("wallet_balance", 300);
        monitor.add_check("mempool_data", 60);
        monitor.add_check("gas_prices", 120);
        monitor.add_check("circuit_breaker", 10);
        
        monitor
    }
    
    fn add_check(&mut self, name: &str, interval_seconds: u64) {
        let check = HealthCheck {
            name: name.to_string(),
            status: HealthStatus::Healthy,
            last_check: Utc::now(),
            check_interval_seconds: interval_seconds,
        };
        self.checks.insert(name.to_string(), check);
    }
    
    pub async fn run_all_checks(&mut self) -> Result<()> {
        for (name, check) in &mut self.checks {
            let should_check = Utc::now().timestamp() - check.last_check.timestamp() 
                >= check.check_interval_seconds as i64;
            
            if should_check {
                check.status = self.perform_health_check(name).await;
                check.last_check = Utc::now();
                
                // Alert on status changes
                if let HealthStatus::Critical { ref message } = check.status {
                    self.send_alert(&format!("CRITICAL: {} - {}", name, message)).await?;
                }
            }
        }
        Ok(())
    }
    
    async fn perform_health_check(&self, check_name: &str) -> HealthStatus {
        match check_name {
            "rpc_connection" => self.check_rpc_connection().await,
            "wallet_balance" => self.check_wallet_balance().await,
            "mempool_data" => self.check_mempool_data().await,
            "gas_prices" => self.check_gas_prices().await,
            "circuit_breaker" => self.check_circuit_breaker().await,
            _ => HealthStatus::Warning { message: "Unknown check".to_string() },
        }
    }
    
    async fn check_rpc_connection(&self) -> HealthStatus {
        // TODO: Actually check RPC connection
        HealthStatus::Healthy
    }
    
    async fn check_wallet_balance(&self) -> HealthStatus {
        // TODO: Check if wallet has sufficient balance
        HealthStatus::Healthy
    }
    
    async fn check_mempool_data(&self) -> HealthStatus {
        // TODO: Check if receiving fresh mempool data
        HealthStatus::Healthy
    }
    
    async fn check_gas_prices(&self) -> HealthStatus {
        // TODO: Check if gas prices are reasonable
        HealthStatus::Healthy
    }
    
    async fn check_circuit_breaker(&self) -> HealthStatus {
        // TODO: Check circuit breaker status
        HealthStatus::Healthy
    }
    
    async fn send_alert(&self, message: &str) -> Result<()> {
        // TODO: Implement alerting (Discord, Slack, email, etc.)
        tracing::error!("🚨 ALERT: {}", message);
        Ok(())
    }
    
    pub fn get_overall_status(&self) -> HealthStatus {
        let mut has_warning = false;
        
        for check in self.checks.values() {
            match &check.status {
                HealthStatus::Critical { .. } | HealthStatus::Down { .. } => {
                    return check.status.clone();
                }
                HealthStatus::Warning { .. } => {
                    has_warning = true;
                }
                HealthStatus::Healthy => {}
            }
        }
        
        if has_warning {
            HealthStatus::Warning { message: "Some checks have warnings".to_string() }
        } else {
            HealthStatus::Healthy
        }
    }
}
