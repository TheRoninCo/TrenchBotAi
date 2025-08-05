// src/modules/shared/sinks/kafka.rs
use rdkafka::{producer::FutureProducer, ClientConfig};
use serde::Serialize;

pub struct KafkaSink {
    producer: FutureProducer,
    topic: String,
}

impl KafkaSink {
    pub fn new(brokers: &str, topic: &str) -> Self {
        let producer = ClientConfig::new()
            .set("bootstrap.servers", brokers)
            .create()
            .unwrap();
        
        Self {
            producer,
            topic: topic.to_string(),
        }
    }

    pub async fn push<T: Serialize>(&self, key: &str, payload: &T) -> anyhow::Result<()> {
        let json = serde_json::to_string(payload)?;
        self.producer
            .send(
                FutureRecord::to(&self.topic)
                    .key(key)
                    .payload(&json),
                Duration::from_secs(0),
            )
            .await?;
        Ok(())
    }
}