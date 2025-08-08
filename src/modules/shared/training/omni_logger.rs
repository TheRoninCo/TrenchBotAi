use crate::observability::{CombatLog, LogError, LogQuery, sinks::LogSink as ObservabilityLogSink};
use super::super::sinks::{self, LogSink, SinkConfig, InstrumentedSink};
use chrono::Utc;
use std::{
    net::SocketAddr,
    path::PathBuf,
    sync::Arc,
    any::Any
};
use tokio::sync::{Mutex, mpsc};

pub struct OmniLogger {
    sinks: Vec<Arc<dyn LogSink>>,
    metrics: Arc<LogMetrics>,
    crypto: Mutex<CryptoEngine>,
    alert_rx: mpsc::Receiver<Alert>,
    combat_mode: bool,
    config_path: Option<PathBuf>,
}

impl OmniLogger {
    pub fn builder() -> OmniLoggerBuilder {
        OmniLoggerBuilder::new()
    }

    pub async fn log(&self, mut entry: CombatLog) -> Result<(), LogError> {
        self.metrics.logs_total.inc();
        
        entry.timestamp = Utc::now().timestamp_nanos();
        entry.signature = self.crypto.lock().await.sign(&entry)?;

        if self.crypto.lock().await.is_encryption_enabled() {
            entry.data = self.crypto.lock().await.encrypt(entry.data)?;
            entry.encrypted = Some(true);
        }

        let results = futures::future::join_all(
            self.sinks.iter().map(|sink| sink.write(&entry))
        ).await;

        if self.combat_mode {
            results.into_iter().collect::<Result<Vec<_>, _>>()?;
        }

        Ok(())
    }

    pub async fn retrieve(&self, query: LogQuery) -> Result<Vec<CombatLog>, LogError> {
        let mongo_sink = self.sinks.iter()
            .find_map(|s| s.as_any().downcast_ref::<sinks::MongoSink>())
            .ok_or(LogError::SinkError("No queryable sink available".into()))?;

        let mut logs = mongo_sink.query(&query).await?;
        
        for log in &mut logs {
            if log.encrypted.unwrap_or(false) {
                log.data = self.crypto.lock().await.decrypt(log.data.clone())?;
            }
        }
        
        Ok(logs)
    }
}

pub struct OmniLoggerBuilder {
    config_path: Option<PathBuf>,
    sink_config: Option<SinkConfig>,
    encryption_key: Option<[u8; 32]>,
    combat_mode: bool,
    metrics_addr: Option<String>,
}

impl OmniLoggerBuilder {
    pub fn new() -> Self {
        Self {
            config_path: None,
            sink_config: None,
            encryption_key: None,
            combat_mode: false,
            metrics_addr: Some("0.0.0.0:8080".into()),
        }
    }

    pub fn with_config(mut self, path: impl Into<PathBuf>) -> Self {
        self.config_path = Some(path.into());
        self
    }

    pub fn with_sink_config(mut self, cfg: SinkConfig) -> Self {
        self.sink_config = Some(cfg);
        self
    }

    pub fn with_encryption(mut self, key: [u8; 32]) -> Self {
        self.encryption_key = Some(key);
        self
    }

    pub fn combat_mode(mut self) -> Self {
        self.combat_mode = true;
        self
    }

    pub fn with_metrics_addr(mut self, addr: impl Into<String>) -> Self {
        self.metrics_addr = Some(addr.into());
        self
    }

    pub async fn build(self) -> Result<OmniLogger, LogError> {
        // Initialize metrics
        let metrics = Arc::new(LogMetrics::new());

        // Start metrics server if configured
        if let Some(addr_str) = &self.metrics_addr {
            let addr: SocketAddr = addr_str.parse()
                .map_err(|e| LogError::ConfigError(e.to_string()))?;
            let metrics_clone = metrics.clone();
            tokio::spawn(async move {
                serve_metrics(addr, metrics_clone).await
                    .expect("Metrics server failed");
            });
        }

        // Load sink configuration
        let sink_config = if let Some(cfg) = self.sink_config {
            cfg
        } else if let Some(path) = &self.config_path {
            let config_str = tokio::fs::read_to_string(path).await?;
            toml::from_str(&config_str)?
        } else {
            return Err(LogError::ConfigError("No sink configuration provided".into()));
        };

        // Build and instrument sinks
        let raw_sinks = sink_config.load_all().await?;
        let sinks = raw_sinks.into_iter()
            .map(|s| Arc::new(InstrumentedSink::new(s, metrics.clone())))
            .collect();

        // Initialize crypto
        let crypto = CryptoEngine::new(self.encryption_key)?;
        let (_, alert_rx) = mpsc::channel(100);

        Ok(OmniLogger {
            sinks,
            metrics,
            crypto: Mutex::new(crypto),
            alert_rx,
            combat_mode: self.combat_mode,
            config_path: self.config_path,
        })
    }
}