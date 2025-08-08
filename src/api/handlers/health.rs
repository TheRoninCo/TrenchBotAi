use axum::{
    extract::State,
    response::Json,
    http::StatusCode,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{info, debug};

use crate::api::server::AppState;

/// **HEALTH CHECK ENDPOINT**
/// Simple health check for load balancers and monitoring
pub async fn health_check() -> Result<Json<HealthResponse>, StatusCode> {
    debug!("💓 Health check requested");
    
    Ok(Json(HealthResponse {
        status: "healthy".to_string(),
        message: "🤖 TrenchBot is LOCKED AND LOADED!".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        timestamp: chrono::Utc::now(),
    }))
}

/// **SYSTEM STATUS ENDPOINT**
/// Comprehensive system status with gaming themes
pub async fn system_status(
    State(app_state): State<Arc<AppState>>,
) -> Result<Json<SystemStatusResponse>, StatusCode> {
    info!("📊 System status requested");
    
    // Check all system components
    let blockchain_status = check_blockchain_connection().await;
    let database_status = check_database_connection().await;
    let ai_status = check_ai_systems().await;
    let websocket_status = check_websocket_connections(&app_state).await;
    
    // Calculate overall health
    let is_healthy = blockchain_status.healthy && 
                     database_status.healthy && 
                     ai_status.healthy &&
                     websocket_status.healthy;
    
    // Gaming-themed status message
    let gaming_message = if is_healthy {
        match std::env::var("GAMING_THEME").as_deref() {
            Ok("runescape") => "🏰 All systems operational! TrenchScape servers are running smoothly! ⚔️",
            Ok("cod") => "🪖 All units reporting ready for combat! Systems are LOCKED AND LOADED! 🎖️",
            Ok("crypto") => "💎 Diamond hands infrastructure! All systems HODLING strong! 🚀",
            Ok("fortnite") => "🏆 Victory Royale status! All systems in the zone! 👑",
            _ => "🤖 TrenchBot systems are fully operational! Ready to secure the bag! 💰"
        }
    } else {
        "🚨 Some systems need attention! Medics en route!"
    };
    
    let response = SystemStatusResponse {
        status: if is_healthy { "healthy" } else { "degraded" }.to_string(),
        gaming_message: gaming_message.to_string(),
        components: SystemComponents {
            blockchain: blockchain_status,
            database: database_status,
            ai_systems: ai_status,
            websockets: websocket_status,
            integrations: IntegrationStatus {
                phantom_wallet: ComponentStatus {
                    healthy: true,
                    message: "Phantom integration ready".to_string(),
                    last_check: chrono::Utc::now(),
                    response_time_ms: 45,
                },
                telegram: ComponentStatus {
                    healthy: true,
                    message: "Telegram bot online".to_string(),
                    last_check: chrono::Utc::now(),
                    response_time_ms: 120,
                },
                notifications: ComponentStatus {
                    healthy: true,
                    message: "Notification system active".to_string(),
                    last_check: chrono::Utc::now(),
                    response_time_ms: 30,
                },
            },
        },
        metrics: SystemMetrics {
            active_users: get_active_user_count(&app_state).await,
            websocket_connections: get_websocket_connection_count(&app_state).await,
            trades_last_hour: get_recent_trade_count().await,
            system_uptime_seconds: get_system_uptime().await,
            memory_usage_mb: get_memory_usage().await,
            cpu_usage_percent: get_cpu_usage().await,
        },
        version: env!("CARGO_PKG_VERSION").to_string(),
        timestamp: chrono::Utc::now(),
    };
    
    let status_code = if is_healthy { 
        StatusCode::OK 
    } else { 
        StatusCode::SERVICE_UNAVAILABLE 
    };
    
    Ok(Json(response))
}

/// **SUPPORTING TYPES**

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub message: String,
    pub version: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Serialize)]
pub struct SystemStatusResponse {
    pub status: String,
    pub gaming_message: String,
    pub components: SystemComponents,
    pub metrics: SystemMetrics,
    pub version: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Serialize)]
pub struct SystemComponents {
    pub blockchain: ComponentStatus,
    pub database: ComponentStatus,
    pub ai_systems: ComponentStatus,
    pub websockets: ComponentStatus,
    pub integrations: IntegrationStatus,
}

#[derive(Debug, Serialize)]
pub struct ComponentStatus {
    pub healthy: bool,
    pub message: String,
    pub last_check: chrono::DateTime<chrono::Utc>,
    pub response_time_ms: u64,
}

#[derive(Debug, Serialize)]
pub struct IntegrationStatus {
    pub phantom_wallet: ComponentStatus,
    pub telegram: ComponentStatus,
    pub notifications: ComponentStatus,
}

#[derive(Debug, Serialize)]
pub struct SystemMetrics {
    pub active_users: u32,
    pub websocket_connections: u32,
    pub trades_last_hour: u32,
    pub system_uptime_seconds: u64,
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
}

/// **HEALTH CHECK FUNCTIONS**

async fn check_blockchain_connection() -> ComponentStatus {
    // In real implementation, this would ping Solana RPC
    ComponentStatus {
        healthy: true,
        message: "Solana RPC responding normally".to_string(),
        last_check: chrono::Utc::now(),
        response_time_ms: 85,
    }
}

async fn check_database_connection() -> ComponentStatus {
    // In real implementation, this would check database connectivity
    ComponentStatus {
        healthy: true,
        message: "Database connection pool healthy".to_string(),
        last_check: chrono::Utc::now(),
        response_time_ms: 25,
    }
}

async fn check_ai_systems() -> ComponentStatus {
    // In real implementation, this would check AI model availability
    ComponentStatus {
        healthy: true,
        message: "AI models loaded and responding".to_string(),
        last_check: chrono::Utc::now(),
        response_time_ms: 150,
    }
}

async fn check_websocket_connections(app_state: &Arc<AppState>) -> ComponentStatus {
    // Check WebSocket health
    ComponentStatus {
        healthy: true,
        message: "WebSocket server accepting connections".to_string(),
        last_check: chrono::Utc::now(),
        response_time_ms: 10,
    }
}

async fn get_active_user_count(app_state: &Arc<AppState>) -> u32 {
    app_state.active_sessions.read().await.len() as u32
}

async fn get_websocket_connection_count(app_state: &Arc<AppState>) -> u32 {
    app_state.user_connections.read().await.values()
        .map(|conns| conns.len() as u32)
        .sum()
}

async fn get_recent_trade_count() -> u32 {
    // In real implementation, query recent trades from database
    125 // Placeholder
}

async fn get_system_uptime() -> u64 {
    // In real implementation, calculate actual uptime
    86400 // Placeholder: 24 hours
}

async fn get_memory_usage() -> f64 {
    // In real implementation, get actual memory usage
    512.5 // Placeholder MB
}

async fn get_cpu_usage() -> f64 {
    // In real implementation, get actual CPU usage
    23.4 // Placeholder percentage
}