use super::{LogSink, SinkHealth};
use crate::observability::{CombatLog, LogError};
use serde::Deserialize;
use std::path::PathBuf;
use tokio::fs;
use std::sync::Arc;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub path: PathBuf,
}

#[derive(Debug)]
pub struct LocalSink {
    base: PathBuf,
}

impl LocalSink {
    pub fn new(cfg: &Config) -> Result<Self, LogError> {
        fs::create_dir_all(&cfg.path)?;
        Ok(Self { base: cfg.path.clone() })
    }
}

#[async_trait::async_trait]
impl LogSink for LocalSink {
    async fn write(&self, log: &CombatLog) -> Result<(), LogError> {
        let file = self.base.join(format!("{}.json", log.timestamp));
        fs::write(file, serde_json::to_vec(log)?).await?;
        Ok(())
    }

    async fn flush(&self) -> Result<(), LogError> {
        // no-op
        Ok(())
    }

    fn health_check(&self) -> SinkHealth {
        SinkHealth { alive: true, last_error: None }
    }
}
