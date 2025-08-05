use thiserror::Error;

#[derive(Error, Debug)]
pub enum CombatError {
    #[error("Safety lock engaged: {0}")]
    SafetyLock(String),
    #[error("Ammo depleted: {0} rounds left")]
    AmmoFailure(f64),
    #[error("Target acquisition failed: {0}")]
    TargetError(String),
    #[error("Config load failed: {0}")]
    ConfigError(String),
    #[error("Logging system error: {0}")]
    LogError(#[from] crate::core::observability::LogError),
}