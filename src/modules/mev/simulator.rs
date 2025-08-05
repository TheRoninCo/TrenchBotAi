use super::{error::MevError, RpcSimulateTransactionResult};
use solana_client::rpc_client::RpcClient;
use solana_sdk::transaction::Transaction;
use std::sync::Arc;
use tokio::time::{sleep, Duration};

pub struct BundleSimulator {
    rpc_client: Arc<RpcClient>,
    config: super::config::SimulationConfig,
}

impl BundleSimulator {
    pub fn new(rpc_client: Arc<RpcClient>, config: super::config::SimulationConfig) -> Self {
        Self { rpc_client, config }
    }

    pub async fn simulate_with_retry(
        &self,
        tx: &Transaction,
    ) -> Result<RpcSimulateTransactionResult, MevError> {
        let mut last_error = None;
        
        for attempt in 0..self.config.max_retries {
            match self.rpc_client.simulate_transaction(tx.clone()).await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    last_error = Some(e);
                    if attempt < self.config.max_retries - 1 {
                        sleep(self.config.rpc_timeout).await;
                    }
                }
            }
        }
        
        Err(last_error.map(MevError::from).unwrap_or(MevError::SimulationFailed))
    }
}