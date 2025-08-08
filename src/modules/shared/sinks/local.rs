use super::{LogSink, SinkHealth};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tokio::fs;
use std::sync::Arc;
use async_trait::async_trait;
use anyhow::Result;

#[derive(Debug, Deserialize)]
pub struct LocalConfig {
    pub path: PathBuf,
}

#[derive(Debug)]
pub struct LocalSink {
    base: PathBuf,
}

impl LocalSink {
    pub async fn new(cfg: &LocalConfig) -> Result<Self> {
        fs::create_dir_all(&cfg.path).await
            .map_err(|e| anyhow::anyhow!("Failed to create directory: {}", e))?;
        Ok(Self { base: cfg.path.clone() })
    }
}

#[async_trait]
impl LogSink for LocalSink {
    async fn write_log<T: Serialize + Send + Sync>(&self, log: T) -> Result<()> {
        let file = self.base.join(format!("{}.json", chrono::Utc::now().timestamp()));
        let data = serde_json::to_vec(&log)
            .map_err(|e| anyhow::anyhow!("Failed to serialize log: {}", e))?;
        fs::write(file, data).await
            .map_err(|e| anyhow::anyhow!("Failed to write log file: {}", e))?;
        Ok(())
    }

    async fn flush(&self) -> Result<()> {
        // no-op for local files
        Ok(())
    }
}

impl LocalSink {
    pub fn health_check(&self) -> SinkHealth {
        if self.base.exists() && self.base.is_dir() {
            SinkHealth::Healthy
        } else {
            SinkHealth::Failing { reason: "Directory not accessible".to_string() }
        }
    }
}
