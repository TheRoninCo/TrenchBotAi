use crate::observability::{OmniLogger, CombatLog, LogSeverity, LogError};
use serde::Serialize;
use serde_json::json;
use once_cell::sync::Lazy;
use std::sync::{Arc, Mutex};
use uuid::Uuid;

/// Global thread-safe logger instance
static COMBAT_LOGGER: Lazy<Arc<Mutex<Option<OmniLogger>>>> = Lazy::new(|| {
    Arc::new(Mutex::new(None))
});

/// Operational context for combat logs
#[derive(Debug, Clone)]
pub struct CombatContext {
    pub operation_id: String,
    pub squad: String,
    pub trace_id: String,
}

impl Default for CombatContext {
    fn default() -> Self {
        Self {
            operation_id: "unspecified".into(),
            squad: "unassigned".into(),
            trace_id: Uuid::new_v4().to_string(),
        }
    }
}

/// Initialize the combat logger with default context
pub fn init_combat_logger(
    logger: OmniLogger,
    default_context: CombatContext
) -> Result<(), LogError> {
    let mut guard = COMBAT_LOGGER.lock().unwrap();
    *guard = Some((logger, default_context));
    Ok(())
}

/// Core logging function with context injection
async fn log_event<T: Serialize>(
    severity: LogSeverity,
    system: &str,
    event_type: &str,
    payload: T,
    context: Option<CombatContext>
) -> Result<(), LogError> {
    if let Some((logger, default_ctx)) = COMBAT_LOGGER.lock().unwrap().as_ref() {
        let ctx = context.unwrap_or_else(|| default_ctx.clone());
        
        let mut log = CombatLog::new(system, event_type)
            .with_severity(severity)
            .with_data(json!({
                "payload": payload,
                "operation_id": ctx.operation_id,
                "squad": ctx.squad,
                "trace_id": ctx.trace_id
            }))?;

        logger.log(log).await?;
    }
    Ok(())
}

/// Combat-specific operations
pub mod ops {
    use super::*;

    pub async fn recon<T: Serialize>(
        system: &str,
        coordinates: (f64, f64),
        payload: T,
        context: Option<CombatContext>
    ) -> Result<(), LogError> {
        log_event(
            LogSeverity::Debug,
            system,
            "recon",
            json!({
                "coordinates": coordinates,
                "details": payload
            }),
            context
        ).await
    }

    // ... other combat operations ...
}

/// Traditional severity-based logging
pub mod severity {
    use super::*;

    pub async fn debug<T: Serialize>(
        system: &str,
        event: &str,
        payload: T,
        context: Option<CombatContext>
    ) -> Result<(), LogError> {
        log_event(LogSeverity::Debug, system, event, payload, context).await
    }

    // ... info, warn, error, critical ...
}

/// Distributed tracing utilities
pub mod trace {
    use super::*;

    pub fn new_context(operation: &str, squad: &str) -> CombatContext {
        CombatContext {
            operation_id: operation.into(),
            squad: squad.into(),
            trace_id: Uuid::new_v4().to_string(),
        }
    }
}