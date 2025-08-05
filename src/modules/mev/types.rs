use serde::{Deserialize, Serialize};
use solana_sdk::{pubkey::Pubkey, transaction::Transaction};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MevType {
    Arbitrage,
    Sandwich,
    Liquidation,
    JitoFlash,
    CexArb,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MevScore {
    pub confidence: f32,
    pub risk: f32,
    pub expected_profit: u64,
    pub ty: MevType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleFeatures {
    pub slot: u64,
    pub tip_lamports: u64,
    pub cu_limit: u32,
    pub cu_price: u32,
    pub accounts: Vec<Pubkey>,
    pub programs: Vec<String>,
    #[serde(skip)]
    pub signer: Pubkey,
}