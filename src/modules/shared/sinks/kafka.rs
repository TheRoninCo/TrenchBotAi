// src/modules/shared/sinks/kafka.rs
use super::{LogSink, SinkHealth};
#[cfg(feature = "kafka")]
use rdkafka::{producer::{FutureProducer, FutureRecord}, ClientConfig};
use serde::Serialize;
use async_trait::async_trait;
use anyhow::Result;
use std::time::Duration;

pub struct KafkaSink {
    #[cfg(feature = "kafka")]
    producer: FutureProducer,
    #[cfg(not(feature = "kafka"))]
    producer: (),
    topic: String,
}

impl KafkaSink {
    pub fn new(brokers: &str, topic: &str) -> Result<Self> {
        #[cfg(feature = "kafka")]
        let producer = ClientConfig::new()
            .set("bootstrap.servers", brokers)
            .create()
            .map_err(|e| anyhow::anyhow!("Failed to create Kafka producer: {:?}", e))?;
        
        #[cfg(not(feature = "kafka"))]
        let producer = ();
        
        Ok(Self {
            producer,
            topic: topic.to_string(),
        })
    }

    pub async fn push<T: Serialize>(&self, key: &str, payload: &T) -> Result<()> {
        let json = serde_json::to_string(payload)
            .map_err(|e| anyhow::anyhow!("Failed to serialize payload: {}", e))?;
        
        #[cfg(feature = "kafka")]
        self.producer
            .send(
                FutureRecord::to(&self.topic)
                    .key(key)
                    .payload(&json),
                Duration::from_secs(0),
            )
            .await
            .map_err(|e| anyhow::anyhow!("Failed to send to Kafka: {:?}", e))?;
            
        #[cfg(not(feature = "kafka"))]
        {
            println!("Kafka sink would send to topic {}: key={}, payload={}", self.topic, key, json);
        }
        
        Ok(())
    }
}

#[async_trait]
impl LogSink for KafkaSink {
    async fn write_log<T: Serialize + Send + Sync>(&self, log: T) -> Result<()> {
        let key = uuid::Uuid::new_v4().to_string();
        self.push(&key, &log).await
    }

    async fn flush(&self) -> Result<()> {
        #[cfg(feature = "kafka")]
        self.producer.flush(Duration::from_secs(5))
            .map_err(|e| anyhow::anyhow!("Failed to flush Kafka producer: {:?}", e))?;
        Ok(())
    }
}

impl KafkaSink {
    pub fn health_check(&self) -> SinkHealth {
        // For Kafka, we'd need to check broker connectivity
        // For now, just return healthy
        SinkHealth::Healthy
    }
}