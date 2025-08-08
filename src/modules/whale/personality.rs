use solana_sdk::pubkey::Pubkey;
use std::collections::{VecDeque, HashMap};
use serde::{Serialize, Deserialize};
use solana_client::rpc_client::RpcClient;
use solana_client::rpc_response::RpcConfirmedTransactionStatusWithSignature;
use solana_client::rpc_config::RpcTransactionConfig;
use std::sync::Arc;
use anyhow::Result;
use solana_sdk::transaction::Transaction;
#[cfg(feature = "spl-token")]
use spl_token::{instruction::TokenInstruction, state::Account as TokenAccount};
use solana_sdk::{instruction::CompiledInstruction, message::Message};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum WhalePersonality {
    Gorilla,    // High-volume FOMO buyer
    Shark,      // Quick-profit taker
    Kraken,     // Market manipulator
    Accumulator,// Slow, steady accumulation
    Bait,       // Fake-out patterns
    Ghost       // Stealthy, small txns
}

#[derive(Debug, Clone, Serialize)]
pub struct WalletBehavior {
    pub velocity: f64,          // SOL moved per hour
    pub holding_period: f64,    // Avg hours held
    pub trade_size_stddev: f64, // Consistency of trade sizes
    pub token_diversity: u8,    // Number of unique tokens held
    pub reversal_rate: f64,     // % of trades that reverse direction
}

impl Default for WalletBehavior {
    fn default() -> Self {
        Self {
            velocity: 0.0,
            holding_period: 0.0,
            trade_size_stddev: 0.0,
            token_diversity: 0,
            reversal_rate: 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalityConfig {
    pub gorilla_threshold: f64,     // min $ volume to qualify
    pub shark_holding_period: f64,  // max hours before selling
    pub kraken_wash_trade_ratio: f64, // % of self-trades
    // ... other thresholds
}

impl PersonalityConfig {
    pub fn default() -> Self {
        Self {
            gorilla_threshold: 50_000.0, // $50k+ moves
            shark_holding_period: 2.0,    // sells within 2 hours
            kraken_wash_trade_ratio: 0.3, // 30% self-trades
        }
    }
}

pub struct PersonalityEngine {
    config: PersonalityConfig,
    behavior_history: HashMap<Pubkey, VecDeque<WalletBehavior>>,
    rpc: Arc<RpcClient>,
}

impl PersonalityEngine {
    pub fn new(config: PersonalityConfig, rpc: Arc<RpcClient>) -> Self {
        Self {
            config,
            behavior_history: HashMap::new(),
            rpc,
        }
    }

    pub async fn classify(&mut self, wallet: &Pubkey) -> WhalePersonality {
        let behavior = self.analyze_behavior(wallet).await;
        self.update_history(wallet, behavior.clone());
        
        match (
            behavior.velocity > self.config.gorilla_threshold,
            behavior.holding_period < self.config.shark_holding_period,
            self.calculate_wash_ratio(wallet) > self.config.kraken_wash_trade_ratio,
        ) {
            (true, _, _) => WhalePersonality::Gorilla,
            (_, true, _) => WhalePersonality::Shark,
            (_, _, true) => WhalePersonality::Kraken,
            _ if behavior.holding_period > 168.0 => WhalePersonality::Accumulator,
            _ if behavior.reversal_rate > 0.7 => WhalePersonality::Bait,
            _ => WhalePersonality::Ghost,
        }
    }

    async fn analyze_behavior(&self, wallet: &Pubkey) -> WalletBehavior {
        let txs = self.rpc.get_signatures_for_address(wallet).await.unwrap();
        let mut behavior = WalletBehavior::default();
        
        // Analyze last 30 transactions
        for tx in txs.iter().take(30) {
            // ... calculate metrics ...
        }
        
        behavior
    }

    fn calculate_wash_ratio(&self, wallet: &Pubkey) -> f64 {
        // Implement wash trade detection
        0.0
    }
    
    fn update_history(&mut self, wallet: &Pubkey, behavior: WalletBehavior) {
        let history = self.behavior_history.entry(*wallet).or_insert_with(VecDeque::new);
        history.push_back(behavior);
        if history.len() > 100 {
            history.pop_front();
        }
    }
}

#[derive(Debug)]
struct TokenTransfer {
        from: Pubkey,
        to: Pubkey,
        mint: Pubkey,
        amount: f64,
        timestamp: i64,
        is_buy: bool,  // Relative to tracked wallet
    }

impl PersonalityEngine {
        async fn fetch_transfers(
            &self,
            wallet: &Pubkey,
            signatures: &[RpcConfirmedTransactionStatusWithSignature],
        ) -> Result<Vec<TokenTransfer>> {
            // Batch fetch transactions
            let txns = self.rpc.get_transactions(
                signatures.iter().map(|s| s.signature).collect(),
                RpcTransactionConfig::default(),
            ).await?;
    
            // Parallel parse
            let mut transfers = Vec::new();
            for tx in txns {
                if let Some(parsed) = self.parse_transaction(wallet, &tx).await? {
                    transfers.push(parsed);
                }
            }
            Ok(transfers)
        }
    
        async fn parse_transaction(
            &self,
            wallet: &Pubkey,
            tx: &Transaction,
        ) -> Result<Option<Vec<TokenTransfer>>> {
            let mut transfers = Vec::new();
            let account_keys = &tx.message.account_keys;
    
            for (i, ix) in tx.message.instructions.iter().enumerate() {
                #[cfg(feature = "spl-token")]
                if let Ok(token_ix) = TokenInstruction::unpack(&ix.data) {
                    match token_ix {
                        TokenInstruction::Transfer { amount } => {
                            let from = account_keys[ix.accounts[0] as usize];
                            let to = account_keys[ix.accounts[1] as usize];
                            let mint = self.resolve_mint(&ix.accounts, &tx).await?;
    
                            transfers.push(TokenTransfer {
                                from,
                                to,
                                mint,
                                amount: amount as f64,
                                timestamp: tx.block_time.unwrap_or_default(),
                                is_buy: &to == wallet,
                            });
                        }
                        _ => continue,
                    }
                }
            }
    
            Ok(if transfers.is_empty() { None } else { Some(transfers) })
        }
    
        #[cfg(feature = "spl-token")]
        async fn resolve_mint(
            &self,
            accounts: &[u8],
            tx: &Transaction,
        ) -> Result<Pubkey> {
            // Get token account and resolve mint
            let token_account = &tx.message.account_keys[accounts[0] as usize];
            let account_data = self.rpc.get_account_data(token_account).await?;
            let token_account = TokenAccount::unpack(&account_data)?;
            Ok(token_account.mint)
        }
        
        #[cfg(not(feature = "spl-token"))]
        async fn resolve_mint(
            &self,
            _accounts: &[u8],
            _tx: &Transaction,
        ) -> Result<Pubkey> {
            Ok(Pubkey::default())
        }
    }