#[cfg(feature = "redis")]
use redis::{Client as RedisClient, Commands};
use serde_json::Value;
use std::sync::Arc;
use futures::Stream; // Added import for Stream

pub struct RedisCoordinator {
    client: Arc<RedisClient>,
    channel: String,
}

impl RedisCoordinator {
    pub fn new(uri: &str, channel: &str) -> Self {
        Self {
            client: Arc::new(RedisClient::open(uri).unwrap()),
            channel: channel.to_string(),
        }
    }

    pub async fn publish(&self, message: &Value) -> anyhow::Result<()> {
        let mut conn = self.client.get_async_connection().await?;
        redis::cmd("PUBLISH")
            .arg(&self.channel)
            .arg(message.to_string())
            .query_async(&mut conn)
            .await?;
        Ok(())
    }

    pub async fn subscribe(&self) -> impl Stream<Item = String> {
        let mut pubsub = self.client.get_async_connection().await.unwrap().into_pubsub();
        pubsub.subscribe(&self.channel).await.unwrap();
        pubsub.into_on_message().map(|msg| msg.get_payload().unwrap())
    }
}