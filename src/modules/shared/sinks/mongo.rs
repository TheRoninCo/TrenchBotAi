use crate::observability::{
    CombatLog, LogError, LogQuery, sinks::{LogSink, SinkHealth, RetryPolicy}
};
use mongodb::{options::{ClientOptions, FindOptions}, Client};
use std::{any::Any, time::Instant};
use futures::stream::TryStreamExt;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub uri: String,
    pub collection: Option<String>,
}

pub struct MongoSink {
    client:     Client,
    collection: String,
}

impl MongoSink {
    pub async fn new(cfg: &Config) -> Result<Self, LogError> {
        let opts = ClientOptions::parse(&cfg.uri).await?;
        let client = Client::with_options(opts)?;
        Ok(Self {
            client,
            collection: cfg.collection.clone().unwrap_or_else(|| "combat_logs".into()),
        })
    }
}

#[async_trait::async_trait]
impl LogSink for MongoSink {
    fn name(&self) -> &'static str {
        "MongoSink"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    async fn write(&self, log: &CombatLog) -> Result<(), LogError> {
        let _start = Instant::now();
        self.client
            .database("trenchbot")
            .collection(&self.collection)
            .insert_one(log, None)
            .await?;
        // you could record latency here if desired
        Ok(())
    }

    async fn flush(&self) -> Result<(), LogError> {
        // no-op
        Ok(())
    }

    fn health_check(&self) -> SinkHealth {
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

        SinkHealth {
            alive,
            latency:   None,                // we don't track this here
            queue_depth: 0,                 // no internal queue
            last_error: if alive { None } else { Some("ping failed".into()) },
        }
    }

    fn retry_policy(&self) -> RetryPolicy {
        // Mongo is generally reliable; fewer retries
        RetryPolicy { max_retries: 1, backoff: std::time::Duration::from_millis(50) }
    }

    async fn query(&self, query: &LogQuery) -> Result<Vec<CombatLog>, LogError> {
        let mut filter = query.into_document();
        let options = FindOptions::builder()
            .limit(query.limit)
            .build();

        let mut cursor = self.client
            .database("trenchbot")
            .collection::<CombatLog>(&self.collection)
            .find(filter, options)
            .await?;

        let mut logs = Vec::new();
        while let Some(log) = cursor.try_next().await? {
            logs.push(log);
        }
        Ok(logs)
    }
}
