use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RebateReceipt {
    pub id: String,
    pub amount: f64,
    pub currency: String,
    pub issued_at: DateTime<Utc>,
    pub recipient: String,
    pub transaction_hash: Option<String>,
}

impl RebateReceipt {
    pub fn new(id: String, amount: f64, currency: String, recipient: String) -> Self {
        Self {
            id,
            amount,
            currency,
            issued_at: Utc::now(),
            recipient,
            transaction_hash: None,
        }
    }

    pub fn set_transaction_hash(&mut self, hash: String) {
        self.transaction_hash = Some(hash);
    }
}