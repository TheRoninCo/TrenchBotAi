use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use anyhow::Result;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricEvent {
    pub name: String,
    pub value: f64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub tags: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CombatLog {
    pub id: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub level: String,
    pub message: String,
    pub context: HashMap<String, String>,
}

#[derive(Debug, thiserror::Error)]
pub enum LogError {
    #[error("Serialization error: {0}")]
    SerializationError(String),
    #[error("Storage error: {0}")]
    StorageError(String),
    #[error("Query error: {0}")]
    QueryError(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogQuery {
    pub start_time: Option<chrono::DateTime<chrono::Utc>>,
    pub end_time: Option<chrono::DateTime<chrono::Utc>>,
    pub level: Option<String>,
    pub limit: Option<usize>,
}

pub mod sinks {
    use super::*;
    
    pub trait LogSink {
        async fn write_log(&self, log: &CombatLog) -> Result<(), LogError>;
        async fn query_logs(&self, query: &LogQuery) -> Result<Vec<CombatLog>, LogError>;
    }
}

#[derive(Debug, Clone)]
pub struct ObservabilityConfig {
    pub enabled: bool,
    pub sampling_rate: f64,
}

impl ObservabilityConfig {
    pub fn new() -> Self {
        Self {
            enabled: true,
            sampling_rate: 1.0,
        }
    }
}

impl Default for ObservabilityConfig {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub enum LogSeverity {
    Debug,
    Info,
    Warning,
    Error,
    Critical,
}

pub struct OmniLogger {
    pub enabled: bool,
}

impl OmniLogger {
    pub fn new() -> Self {
        Self { enabled: true }
    }
}