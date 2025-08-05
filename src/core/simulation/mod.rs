use solana_client::rpc_client::RpcClient;
use solana_sdk::{commitment_config::CommitmentConfig, transaction::Transaction};
use anyhow::{Context, Result};

#[derive(Debug, Clone)]
pub struct TransactionSimulator {
    rpc_client: RpcClient,
}

impl TransactionSimulator {
    pub fn new(rpc_url: &str) -> Self {
        Self {
            rpc_client: RpcClient::new_with_commitment(
                rpc_url.to_string(),
                CommitmentConfig::confirmed(),
            ),
        }
    }

    pub async fn simulate_sandwich_attack(
        &self,
        frontrun_tx: &Transaction,
        target_tx: &Transaction,
        backrun_tx: &Transaction,
    ) -> Result<SandwichSimulationResult> {
        let frontrun_result = self.simulate_transaction(frontrun_tx).await
            .context("Failed to simulate frontrun transaction")?;
        
        if !frontrun_result.success {
            return Ok(SandwichSimulationResult {
                frontrun_result,
                target_result: None,
                backrun_result: None,
                estimated_profit: 0.0,
                success: false,
            });
        }

        let target_result = self.simulate_transaction(target_tx).await
            .context("Failed to simulate target transaction")?;
        
        let backrun_result = self.simulate_transaction(backrun_tx).await
            .context("Failed to simulate backrun transaction")?;

        let profit = self.calculate_profit(&frontrun_result, &backrun_result);

        Ok(SandwichSimulationResult {
            frontrun_result,
            target_result: Some(target_result),
            backrun_result: Some(backrun_result),
            estimated_profit: profit,
            success: backrun_result.success && profit > 0.0,
        })
    }

    async fn simulate_transaction(&self, tx: &Transaction) -> Result<TransactionSimulationResult> {
        let sim_result = self.rpc_client.simulate_transaction(tx)?;
        
        Ok(TransactionSimulationResult {
            success: sim_result.value.err.is_none(),
            gas_used: sim_result.value.units_consumed.unwrap_or(0),
            logs: sim_result.value.logs.unwrap_or_default(),
            error: sim_result.value.err,
        })
    }
}