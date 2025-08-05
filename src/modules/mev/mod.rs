pub mod analyzer;
pub mod config;
pub mod detector;
pub mod error;
pub mod simulator;
pub mod types;

pub use analyzer::BundleAnalyzer;
pub use config::MevConfig;
pub use detector::{DetectorPipeline, MevDetector};
pub use error::MevError;
pub use simulator::BundleSimulator;
pub use types::{BundleFeatures, MevScore, MevType};

// Re-export common types
pub use solana_sdk::{
    pubkey::Pubkey,
    signature::Signature,
    transaction::Transaction,
};
pub use solana_client::rpc_response::RpcSimulateTransactionResult;

/// Core detection trait for legacy systems
pub trait LegacyDetector: Send + Sync {
    fn detect_mev_type(
        &self,
        txs: &[Transaction],
        sim: &RpcSimulateTransactionResult,
    ) -> Result<MevType, MevError>;
}