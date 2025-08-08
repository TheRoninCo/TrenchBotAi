use anyhow::Result;
use axum::{
    extract::{State, Json},
    response::Json as ResponseJson,
    http::StatusCode,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{info, warn, error, debug};

use crate::api::server::AppState;
use crate::integrations::phantom_wallet::PhantomWalletIntegration;

/// **WALLET CONNECTION ENDPOINT**
/// Connect Phantom wallet and create authenticated session
pub async fn connect_wallet(
    State(app_state): State<Arc<AppState>>,
    Json(payload): Json<ConnectWalletRequest>,
) -> Result<ResponseJson<ConnectWalletResponse>, StatusCode> {
    info!("👻 Phantom wallet connection request from: {}", 
          payload.public_key.get(0..8).unwrap_or("unknown"));

    // Verify wallet signature
    let verification_result = app_state
        .phantom_wallet
        .verify_wallet_signature(&payload.public_key, &payload.signature, &payload.message)
        .await;

    match verification_result {
        Ok(true) => {
            debug!("✅ Wallet signature verified successfully");
            
            // Create user session
            let session_token = app_state.auth_manager.create_session(
                &payload.public_key,
                payload.gaming_theme.clone(),
            ).await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
            
            // Get user rank and gaming profile
            let user_rank = app_state.ranking_system
                .get_user_rank(&payload.public_key)
                .await
                .unwrap_or_default();
            
            // Generate welcome message with gaming theme
            let welcome_message = app_state.trenchware_ui.get_gaming_message(
                "wallet_connected",
                crate::integrations::trenchware_ui::MessageContext::default()
            );
            
            let response = ConnectWalletResponse {
                success: true,
                session_token,
                user_id: payload.public_key.clone(),
                wallet_address: payload.public_key,
                rank: user_rank.rank_name,
                gaming_theme: payload.gaming_theme,
                welcome_message,
                permissions: vec!["VIEW_PORTFOLIO".to_string(), "EXECUTE_TRADES".to_string()],
                expires_at: chrono::Utc::now() + chrono::Duration::hours(24),
            };
            
            info!("🎉 Wallet connected successfully - gaming mode activated!");
            Ok(ResponseJson(response))
        },
        Ok(false) => {
            warn!("🚫 Invalid wallet signature");
            Err(StatusCode::UNAUTHORIZED)
        },
        Err(e) => {
            error!("💥 Wallet verification failed: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// **SESSION VERIFICATION ENDPOINT**
/// Verify existing session token
pub async fn verify_session(
    State(app_state): State<Arc<AppState>>,
    Json(payload): Json<VerifySessionRequest>,
) -> Result<ResponseJson<VerifySessionResponse>, StatusCode> {
    debug!("🔍 Session verification request");
    
    match app_state.auth_manager.verify_session(&payload.session_token).await {
        Ok(session) => {
            let response = VerifySessionResponse {
                valid: true,
                user_id: session.user_id,
                wallet_address: session.wallet_address,
                permissions: session.permissions.iter().map(|p| format!("{:?}", p)).collect(),
                expires_at: session.last_active + chrono::Duration::hours(24),
            };
            
            debug!("✅ Session verified successfully");
            Ok(ResponseJson(response))
        },
        Err(_) => {
            warn!("🚫 Invalid session token");
            let response = VerifySessionResponse {
                valid: false,
                user_id: None,
                wallet_address: None,
                permissions: vec![],
                expires_at: chrono::Utc::now(),
            };
            Ok(ResponseJson(response))
        }
    }
}

/// **TOKEN REFRESH ENDPOINT**
/// Refresh authentication token
pub async fn refresh_token(
    State(app_state): State<Arc<AppState>>,
    Json(payload): Json<RefreshTokenRequest>,
) -> Result<ResponseJson<RefreshTokenResponse>, StatusCode> {
    info!("🔄 Token refresh request");
    
    match app_state.auth_manager.refresh_session(&payload.session_token).await {
        Ok(new_token) => {
            let gaming_message = app_state.trenchware_ui.get_gaming_message(
                "session_refreshed",
                crate::integrations::trenchware_ui::MessageContext::default()
            );
            
            let response = RefreshTokenResponse {
                success: true,
                new_session_token: new_token,
                gaming_message,
                expires_at: chrono::Utc::now() + chrono::Duration::hours(24),
            };
            
            info!("🎯 Token refreshed - ready for battle!");
            Ok(ResponseJson(response))
        },
        Err(e) => {
            warn!("🚫 Token refresh failed: {}", e);
            Err(StatusCode::UNAUTHORIZED)
        }
    }
}

/// **WALLET DISCONNECT ENDPOINT**
/// Disconnect wallet and end session
pub async fn disconnect_wallet(
    State(app_state): State<Arc<AppState>>,
    Json(payload): Json<DisconnectWalletRequest>,
) -> Result<ResponseJson<DisconnectWalletResponse>, StatusCode> {
    info!("👋 Wallet disconnect request");
    
    match app_state.auth_manager.end_session(&payload.session_token).await {
        Ok(_) => {
            let gaming_message = match payload.gaming_theme.as_deref() {
                Some("runescape") => "🏰 You have logged out of TrenchScape! See you next time, adventurer! ⚔️",
                Some("cod") => "🪖 Mission complete! You've been discharged from duty! 🎖️", 
                Some("crypto") => "💎 Diamond hands rest! You've exited the trading zone! 🚀",
                Some("fortnite") => "🏆 Victory achieved! You've left the battle royale! 👑",
                _ => "🤖 TrenchBot systems offline! Thanks for trading with us! 💰"
            };
            
            let response = DisconnectWalletResponse {
                success: true,
                gaming_message: gaming_message.to_string(),
                final_stats: Some(FinalStats {
                    session_duration_minutes: 45, // Placeholder
                    trades_executed: 12,          // Placeholder
                    profit_loss: 156.78,          // Placeholder
                }),
            };
            
            info!("✅ Session ended successfully");
            Ok(ResponseJson(response))
        },
        Err(e) => {
            error!("💥 Failed to end session: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// **REQUEST/RESPONSE TYPES**

#[derive(Debug, Deserialize)]
pub struct ConnectWalletRequest {
    pub public_key: String,
    pub signature: String,
    pub message: String,
    pub gaming_theme: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ConnectWalletResponse {
    pub success: bool,
    pub session_token: String,
    pub user_id: String,
    pub wallet_address: String,
    pub rank: String,
    pub gaming_theme: Option<String>,
    pub welcome_message: String,
    pub permissions: Vec<String>,
    pub expires_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Deserialize)]
pub struct VerifySessionRequest {
    pub session_token: String,
}

#[derive(Debug, Serialize)]
pub struct VerifySessionResponse {
    pub valid: bool,
    pub user_id: Option<String>,
    pub wallet_address: Option<String>,
    pub permissions: Vec<String>,
    pub expires_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Deserialize)]
pub struct RefreshTokenRequest {
    pub session_token: String,
}

#[derive(Debug, Serialize)]
pub struct RefreshTokenResponse {
    pub success: bool,
    pub new_session_token: String,
    pub gaming_message: String,
    pub expires_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Deserialize)]
pub struct DisconnectWalletRequest {
    pub session_token: String,
    pub gaming_theme: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct DisconnectWalletResponse {
    pub success: bool,
    pub gaming_message: String,
    pub final_stats: Option<FinalStats>,
}

#[derive(Debug, Serialize)]
pub struct FinalStats {
    pub session_duration_minutes: u32,
    pub trades_executed: u32,
    pub profit_loss: f64,
}