use super::{LogSink, SinkHealth};
use crate::observability::{CombatLog, LogError};
use serde::Deserialize;
use aws_sdk_s3::{Client as S3Client, Error as SdkError};
use std::time::Duration;
use tokio::time::sleep;
use std::sync::Arc;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub bucket: String,
    pub region: Option<String>,
}

#[derive(Debug)]
pub struct S3Sink {
    client: S3Client,
    bucket: String,
}

impl S3Sink {
    pub async fn new(cfg: &Config) -> Result<Self, LogError> {
        let config = aws_config::load_from_env().await;
        let client = if let Some(r) = &cfg.region {
            let mut builder = aws_sdk_s3::config::Builder::from(&config);
            builder.region(r.parse().unwrap());
            S3Client::from_conf(builder.build())
        } else {
            S3Client::new(&config)
        };
        Ok(Self { client, bucket: cfg.bucket.clone() })
    }
}

#[async_trait::async_trait]
impl LogSink for S3Sink {
    async fn write(&self, log: &CombatLog) -> Result<(), LogError> {
        let key = format!("logs/{}/{}.json.zst",
            chrono::Utc::now().format("%Y-%m-%d"),
            log.timestamp);
        let body = zstd::encode_all(
            serde_json::to_vec(log)?.as_slice(),
            3
        )?;
        self.client.put_object()
            .bucket(&self.bucket)
            .key(key)
            .body(body.into())
            .send()
            .await
            .map_err(LogError::AwsError)?;
        Ok(())
    }

    async fn flush(&self) -> Result<(), LogError> {
        // no-op; AWS SDK buffers internally
        Ok(())
    }

    fn health_check(&self) -> SinkHealth {
        // TODO: perform a lightweight head_object or list_objects_one
        SinkHealth { alive: true, last_error: None }
    }
}
