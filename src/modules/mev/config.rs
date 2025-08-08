use serde::{Deserialize, Serialize};
use std::{path::Path, time::Duration};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MevConfig {
    #[serde(default = "default_simulation")]
    pub simulation: SimulationConfig,
    #[serde(default = "default_detection")]
    pub detection: DetectionConfig,
    #[serde(default = "default_alerting")]
    pub alerting: AlertingConfig,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SimulationConfig {
    #[serde(default = "default_max_retries")]
    pub max_retries: usize,
    #[serde(default = "default_rpc_timeout_ms")]
    pub rpc_timeout: Duration,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DetectionConfig {
    #[serde(default = "default_sandwich")]
    pub sandwich: SandwichConfig,
    #[serde(default = "default_statistical")]
    pub statistical: StatisticalConfig,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AlertingConfig {
    #[serde(default = "default_alert_threshold")]
    pub profit_threshold_lamports: u64,
    #[serde(default = "default_confidence_threshold")]
    pub min_confidence: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SandwichConfig {
    pub enabled: bool,
    pub min_profit_threshold: u64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct StatisticalConfig {
    pub enabled: bool,
    pub window_size: usize,
}

fn default_simulation() -> SimulationConfig {
    SimulationConfig {
        max_retries: default_max_retries(),
        rpc_timeout: default_rpc_timeout_ms(),
    }
}

fn default_detection() -> DetectionConfig {
    DetectionConfig {
        sandwich: default_sandwich(),
        statistical: default_statistical(),
    }
}

fn default_alerting() -> AlertingConfig {
    AlertingConfig {
        profit_threshold_lamports: default_alert_threshold(),
        min_confidence: default_confidence_threshold(),
    }
}

fn default_max_retries() -> usize {
    3
}

fn default_rpc_timeout_ms() -> Duration {
    Duration::from_millis(5000)
}

fn default_sandwich() -> SandwichConfig {
    SandwichConfig {
        enabled: true,
        min_profit_threshold: 1000000,
    }
}

fn default_statistical() -> StatisticalConfig {
    StatisticalConfig {
        enabled: true,
        window_size: 100,
    }
}

fn default_alert_threshold() -> u64 {
    500000
}

fn default_confidence_threshold() -> f32 {
    0.8
}