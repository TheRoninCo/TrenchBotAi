use super::{LogSink, SinkHealth, RetryPolicy};
#[cfg(feature = "mongodb")]
use mongodb::{options::ClientOptions, Client};
use std::time::Instant;
use serde::{Deserialize, Serialize};
use async_trait::async_trait;
use anyhow::Result;

#[derive(Debug, Deserialize)]
pub struct MongoConfig {
    pub uri: String,
    pub collection: Option<String>,
}

pub struct MongoSink {
    client:     Client,
    collection: String,
}

impl MongoSink {
    pub async fn new(cfg: &MongoConfig) -> Result<Self> {
        let opts = ClientOptions::parse(&cfg.uri).await
            .map_err(|e| anyhow::anyhow!("Failed to parse MongoDB URI: {}", e))?;
        let client = Client::with_options(opts)
            .map_err(|e| anyhow::anyhow!("Failed to create MongoDB client: {}", e))?;
        Ok(Self {
            client,
            collection: cfg.collection.clone().unwrap_or_else(|| "combat_logs".into()),
        })
    }
}

#[async_trait]
impl LogSink for MongoSink {
    async fn write_log<T: Serialize + Send + Sync>(&self, log: T) -> Result<()> {
        let _start = Instant::now();
        self.client
            .database("trenchbot")
            .collection(&self.collection)
            .insert_one(&log, None)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to write to MongoDB: {}", e))?;
        Ok(())
    }

    async fn flush(&self) -> Result<()> {
        // MongoDB auto-flushes, so this is a no-op
        Ok(())
    }
}

impl MongoSink {
    pub fn health_check(&self) -> SinkHealth {
        // Try a cheap ping with list_collections (or similar)
        let alive = match futures::executor::block_on(
            self.client.database("trenchbot").list_collection_names(None)
        ) {
            Ok(_) => true,
            Err(e) => {
                eprintln!("Mongo health check failed: {}", e);
                false
            }
        };

        if alive {
            SinkHealth::Healthy
        } else {
            SinkHealth::Failing { reason: "Connection failed".to_string() }
        }
    }

    pub fn get_retry_policy(&self) -> RetryPolicy {
        RetryPolicy { 
            max_retries: 3,
            initial_delay_ms: 100,
            max_delay_ms: 5000,
            backoff_multiplier: 2.0,
        }
    }
}
