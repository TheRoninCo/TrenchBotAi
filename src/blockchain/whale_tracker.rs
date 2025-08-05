use solana_client::nonblocking::rpc_client::RpcClient;
use solana_sdk::pubkey::Pubkey;
use serde::Serialize;
use std::time::Duration;
use tokio::time;
use crate::observability::{combat_logger, CombatContext};

#[derive(Clone)]
pub struct WhaleTracker {
    rpc: RpcClient,
    config: TierConfig,
    wallet_source: WalletSource,
    interval: Duration,
    context: CombatContext,
    alert_rules: AlertRules,
}

impl WhaleTracker {
    pub async fn scan_and_log(&self) -> Result<(), Box<dyn std::error::Error>> {
        let wallets = self.get_wallets().await?;
        
        for wallet in wallets {
            let bal = self.rpc.get_balance(&wallet).await?;
            let (risk_score, flags) = self.assess_risk(&wallet).await;
            let tier = Self::classify(bal, &self.config);

            // Enhanced payload
            #[derive(Serialize)]
            struct Payload {
                wallet: String,
                balance: u64,
                tier: String,
                risk_score: f32,
                flags: Vec<String>,
                sol_value: f64,  // USD estimate
            }

            let payload = Payload {
                wallet: wallet.to_string(),
                balance: bal,
                tier: tier.to_string(),
                risk_score,
                flags: flags.into_iter().map(|s| s.to_string()).collect(),
                sol_value: bal as f64 / 1_000_000_000.0 * current_sol_price(),
            };

            // Log with severity based on risk
            let severity = if risk_score > 0.7 { 
                LogSeverity::Warn 
            } else { 
                LogSeverity::Info 
            };

            combat_logger::severity::log(
                severity,
                "whale_tracker",
                "wallet_scan",
                payload,
                Some(self.context.clone())
            ).await?;
        }
        Ok(())
    }
}