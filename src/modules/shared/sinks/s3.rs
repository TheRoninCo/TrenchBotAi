use super::{LogSink, SinkHealth};
use serde::{Deserialize, Serialize};
#[cfg(feature = "aws")]
use aws_sdk_s3::{Client as S3Client, Error as SdkError};
use std::time::Duration;
use tokio::time::sleep;
use std::sync::Arc;
use async_trait::async_trait;
use anyhow::Result;

#[derive(Debug, Deserialize)]
pub struct S3Config {
    pub bucket: String,
    pub region: Option<String>,
}

#[derive(Debug)]
pub struct S3Sink {
    client: S3Client,
    bucket: String,
}

impl S3Sink {
    pub async fn new(cfg: &S3Config) -> Result<Self> {
        let config = aws_config::load_from_env().await;
        let client = if let Some(r) = &cfg.region {
            let mut builder = aws_sdk_s3::config::Builder::from(&config);
            builder.region(r.parse().map_err(|e| anyhow::anyhow!("Invalid region: {:?}", e))?);
            S3Client::from_conf(builder.build())
        } else {
            S3Client::new(&config)
        };
        Ok(Self { client, bucket: cfg.bucket.clone() })
    }
}

#[async_trait]
impl LogSink for S3Sink {
    async fn write_log<T: Serialize + Send + Sync>(&self, log: T) -> Result<()> {
        let key = format!("logs/{}/{}.json",
            chrono::Utc::now().format("%Y-%m-%d"),
            uuid::Uuid::new_v4());
        let body = serde_json::to_vec(&log)
            .map_err(|e| anyhow::anyhow!("Failed to serialize log: {}", e))?;
        
        #[cfg(feature = "aws")]
        self.client.put_object()
            .bucket(&self.bucket)
            .key(key)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to write to S3: {:?}", e))?;
        
        #[cfg(not(feature = "aws"))]
        {
            // Mock implementation for testing
            println!("S3 sink would write to bucket: {}, key: {}", self.bucket, key);
        }
        
        Ok(())
    }

    async fn flush(&self) -> Result<()> {
        // no-op; AWS SDK buffers internally
        Ok(())
    }
}

impl S3Sink {
    pub fn health_check(&self) -> SinkHealth {
        // TODO: perform a lightweight head_object or list_objects_one
        SinkHealth::Healthy
    }
}
