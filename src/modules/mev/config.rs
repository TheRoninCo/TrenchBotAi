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
    #[serde(default = "default_rpc_timeout_ms", with = "humantime_serde")]
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

// Default implementations and constants omitted for brevity