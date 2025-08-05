//! Military-Grade Observability Suite
//! - Cryptographic logging
//! - Distributed tracing
//! - Real-time anomaly detection

pub mod omni_logger;
pub mod combat_logger;
pub mod crypto;
pub mod metrics;
pub mod sinks;
pub mod replay_engine;

// Core exports
pub use omni_logger::{OmniLogger, OmniLoggerBuilder};
pub use combat_logger::{
    init_combat_logger, CombatContext,
    ops::{recon, engage, system_failure},
    severity::{debug, info, warn, error, critical},
    trace
};
pub use sinks::{LogSink, SinkConfig};