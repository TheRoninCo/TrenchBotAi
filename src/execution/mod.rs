use solana_client::rpc_client::RpcClient;
use solana_sdk::{signature::Signature, transaction::Transaction};
use anyhow::Result;
use crate::types::TradeResult;

pub struct ExecutionEngine {
    rpc_client: RpcClient,
}

impl ExecutionEngine {
    pub fn new(rpc_url: String) -> Self {
        Self {
            rpc_client: RpcClient::new(rpc_url),
        }
    }

    pub async fn execute_transaction(&self, tx: Transaction) -> Result<TradeResult> {
        let signature = self.rpc_client.send_and_confirm_transaction(&tx)?;
        Ok(TradeResult {
            success: true,
            profit: None,  // Will be filled after execution
            gas_used: 0,   // Will be updated
            execution_time_ms: 0,
        })
    }
}