#[cfg(feature = "redis")]
use redis::{Client, Commands};
use solana_sdk::pubkey::Pubkey;
use serde_json;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenMeta {
    pub name: String,
    pub symbol: String,
    pub decimals: u8,
    pub supply: u64,
}

#[derive(Clone)]
pub struct TokenCache {
    #[cfg(feature = "redis")]
    client: Client,
    #[cfg(not(feature = "redis"))]
    client: String,
    ttl: usize,
}

impl TokenCache {
    pub fn new(redis_url: &str, ttl: usize) -> Self {
        Self {
            #[cfg(feature = "redis")]
            client: Client::open(redis_url).unwrap(),
            #[cfg(not(feature = "redis"))]
            client: redis_url.to_string(),
            ttl,
        }
    }

    pub fn get_token_meta(&self, mint: &Pubkey) -> Option<TokenMeta> {
        let mut conn = self.client.get_connection().ok()?;
        let key = format!("token:{}", mint);
        
        if let Ok(data) = conn.get::<_, String>(&key) {
            return serde_json::from_str(&data).ok();
        }
        
        None
    }

    pub fn set_token_meta(&self, mint: &Pubkey, meta: &TokenMeta) {
        let mut conn = match self.client.get_connection() {
            Ok(c) => c,
            Err(_) => return,
        };
        
        let key = format!("token:{}", mint);
        let _ = conn.set_ex(
            key,
            serde_json::to_string(meta).unwrap(),
            self.ttl
        );
    }
}