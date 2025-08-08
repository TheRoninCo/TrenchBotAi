pub mod mongo;
pub mod s3;
pub mod kafka;
pub mod local;

use async_trait::async_trait;
use serde::Serialize;
use anyhow::Result;

#[async_trait]
pub trait LogSink {
    async fn write_log<T: Serialize + Send + Sync>(&self, log: T) -> Result<()>;
    async fn flush(&self) -> Result<()>;
}

#[derive(Debug, Clone)]
pub enum SinkHealth {
    Healthy,
    Degraded { reason: String },
    Failing { reason: String },
}

#[derive(Debug, Clone)]
pub struct RetryPolicy {
    pub max_retries: usize,
    pub initial_delay_ms: u64,
    pub max_delay_ms: u64,
    pub backoff_multiplier: f64,
}

#[derive(Debug, Clone)]
pub struct SinkConfig {
    pub enabled: bool,
    pub batch_size: usize,
    pub flush_interval_ms: u64,
    pub retry_policy: RetryPolicy,
}

pub struct InstrumentedSink<T: LogSink> {
    inner: T,
    metrics: std::sync::Arc<std::sync::Mutex<SinkMetrics>>,
}

#[derive(Debug, Default)]
struct SinkMetrics {
    writes_total: u64,
    writes_failed: u64,
    flush_total: u64,
}

pub use mongo::MongoSink;
pub use s3::S3Sink;
pub use kafka::KafkaSink;
pub use local::LocalSink;
