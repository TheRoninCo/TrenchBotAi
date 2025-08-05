use thiserror::Error;
use solana_client::client_error::ClientError;

#[derive(Debug, Error)]
pub enum MevError {
    #[error("RPC error: {0}")]
    RpcError(#[from] ClientError),
    #[error("Simulation failed after retries")]
    SimulationFailed,
    #[error("Invalid transaction bundle: {0}")]
    InvalidBundle(String),
    #[error("Detection error: {0}")]
    DetectionError(String),
    #[error("Configuration error: {0}")]
    ConfigError(String),
}

impl MevError {
    pub fn is_retriable(&self) -> bool {
        matches!(self, Self::RpcError(e) if e.is_retriable())
    }
}