use anyhow::Result;
use axum::{
    extract::{State, Query, Json as ExtractJson},
    response::Json,
    http::StatusCode,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{info, warn, error, debug};

use crate::api::server::AppState;

/// **GET TRADES ENDPOINT**
/// Get user's trading history with gaming themes
pub async fn get_trades(
    State(app_state): State<Arc<AppState>>,
    Query(params): Query<GetTradesParams>,
) -> Result<Json<GetTradesResponse>, StatusCode> {
    info!("📈 Trading history request from user");
    
    let user_session = if let Some(token) = &params.session_token {
        app_state.auth_manager.verify_session(token).await
            .map_err(|_| StatusCode::UNAUTHORIZED)?
    } else {
        return Err(StatusCode::UNAUTHORIZED);
    };
    
    // Get trading data
    let trades = get_user_trades(&app_state, &user_session, &params).await;
    let trading_stats = calculate_trading_stats(&trades).await;
    
    // Generate gaming-themed messages
    let gaming_messages = generate_trading_gaming_messages(
        &app_state,
        &params.gaming_theme,
        &trading_stats
    ).await;
    
    let response = GetTradesResponse {
        trades,
        stats: trading_stats,
        gaming_messages,
        pagination: PaginationInfo {
            page: params.page.unwrap_or(1),
            per_page: params.per_page.unwrap_or(50),
            total_pages: 10, // Placeholder
            total_trades: 487, // Placeholder
        },
        filters_applied: params.status.clone(),
        timestamp: chrono::Utc::now(),
    };
    
    debug!("✅ Trading data compiled successfully");
    Ok(Json(response))
}

/// **EXECUTE TRADE ENDPOINT**
/// Execute a new trade with gaming-themed responses
pub async fn execute_trade(
    State(app_state): State<Arc<AppState>>,
    ExtractJson(payload): ExtractJson<ExecuteTradeRequest>,
) -> Result<Json<ExecuteTradeResponse>, StatusCode> {
    info!("⚡ Trade execution request: {} {} {} at ${}", 
          payload.action, payload.amount, payload.token_symbol, payload.price);
    
    let user_session = app_state.auth_manager.verify_session(&payload.session_token).await
        .map_err(|_| StatusCode::UNAUTHORIZED)?;
    
    // Validate trading request
    let validation_result = validate_trade_request(&app_state, &user_session, &payload).await;
    if let Err(error_msg) = validation_result {
        warn!("🚫 Trade validation failed: {}", error_msg);
        return Ok(Json(ExecuteTradeResponse {
            success: false,
            trade_id: None,
            status: "REJECTED".to_string(),
            error_message: Some(error_msg),
            gaming_message: get_rejection_gaming_message(&payload.gaming_theme, &error_msg),
            estimated_gas: None,
            confirmation_time: None,
            timestamp: chrono::Utc::now(),
        }));
    }
    
    // Execute the trade
    match execute_trade_on_blockchain(&app_state, &user_session, &payload).await {
        Ok(trade_result) => {
            info!("🎉 Trade executed successfully: {}", trade_result.trade_id);
            
            // Generate success gaming message
            let gaming_message = generate_success_gaming_message(
                &payload.gaming_theme,
                &payload.action,
                &payload.token_symbol,
                payload.amount,
                &trade_result.status
            );
            
            // Send notifications
            let _ = send_trade_notifications(&app_state, &user_session, &trade_result, &gaming_message).await;
            
            // Update leaderboards and achievements
            let _ = update_trading_achievements(&app_state, &user_session.user_id, &trade_result).await;
            
            Ok(Json(ExecuteTradeResponse {
                success: true,
                trade_id: Some(trade_result.trade_id),
                status: trade_result.status,
                error_message: None,
                gaming_message,
                estimated_gas: Some(trade_result.gas_used),
                confirmation_time: Some(trade_result.confirmation_time),
                timestamp: chrono::Utc::now(),
            }))
        },
        Err(e) => {
            error!("💥 Trade execution failed: {}", e);
            
            let gaming_message = match payload.gaming_theme.as_deref() {
                Some("runescape") => "⚔️ Trade failed! The market gods were not in your favor this time!",
                Some("cod") => "🪖 Mission failed! Regroup and try again, soldier!",
                Some("crypto") => "💎 Diamond hands shaken! Market rejected your move!",
                Some("fortnite") => "🏆 Trade eliminated! Better luck next drop!",
                _ => "📈 Trade execution failed - check your settings and try again!"
            }.to_string();
            
            Ok(Json(ExecuteTradeResponse {
                success: false,
                trade_id: None,
                status: "FAILED".to_string(),
                error_message: Some(e.to_string()),
                gaming_message,
                estimated_gas: None,
                confirmation_time: None,
                timestamp: chrono::Utc::now(),
            }))
        }
    }
}

/// **GET TRADE HISTORY ENDPOINT**
/// Get detailed trading history with analytics
pub async fn get_trade_history(
    State(app_state): State<Arc<AppState>>,
    Query(params): Query<TradeHistoryParams>,
) -> Result<Json<TradeHistoryResponse>, StatusCode> {
    info!("📊 Trade history analytics request - period: {}", params.period);
    
    let user_session = if let Some(token) = &params.session_token {
        app_state.auth_manager.verify_session(token).await
            .map_err(|_| StatusCode::UNAUTHORIZED)?
    } else {
        return Err(StatusCode::UNAUTHORIZED);
    };
    
    // Get historical trades based on period
    let trades = get_historical_trades(&app_state, &user_session, &params).await;
    let analytics = calculate_trade_analytics(&trades, &params.period).await;
    let performance = calculate_trading_performance(&trades).await;
    
    // Generate insights and gaming messages
    let insights = generate_trading_insights(&app_state, &params.gaming_theme, &analytics, &performance).await;
    
    let response = TradeHistoryResponse {
        period: params.period,
        total_trades: trades.len() as u32,
        successful_trades: trades.iter().filter(|t| t.status == "COMPLETED").count() as u32,
        failed_trades: trades.iter().filter(|t| t.status == "FAILED").count() as u32,
        analytics,
        performance,
        insights,
        recent_trades: trades.into_iter().take(10).collect(), // Last 10 trades
        timestamp: chrono::Utc::now(),
    };
    
    Ok(Json(response))
}

/// **SUPPORTING FUNCTIONS**

async fn get_user_trades(
    _app_state: &Arc<AppState>,
    user_session: &crate::api::server::UserSession,
    params: &GetTradesParams,
) -> Vec<Trade> {
    // Placeholder implementation - would query database
    let mut trades = vec![
        Trade {
            trade_id: "trade_001".to_string(),
            user_id: user_session.user_id.clone(),
            token_symbol: "BONK".to_string(),
            token_address: "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263".to_string(),
            action: "BUY".to_string(),
            amount: 1_000_000.0,
            price: 0.0000234,
            total_value: 23.40,
            status: "COMPLETED".to_string(),
            transaction_hash: "5KJp...xB7s".to_string(),
            gas_used: 0.001234,
            realized_pnl: Some(4.56),
            gaming_message: "🎯 Spawn kill successful! BONK acquired at perfect entry!".to_string(),
            created_at: chrono::Utc::now() - chrono::Duration::minutes(45),
            completed_at: Some(chrono::Utc::now() - chrono::Duration::minutes(44)),
        },
        Trade {
            trade_id: "trade_002".to_string(),
            user_id: user_session.user_id.clone(),
            token_symbol: "SOL".to_string(),
            token_address: "So11111111111111111111111111111111111111112".to_string(),
            action: "SELL".to_string(),
            amount: 5.0,
            price: 195.50,
            total_value: 977.50,
            status: "COMPLETED".to_string(),
            transaction_hash: "8NmP...yT9k".to_string(),
            gas_used: 0.001456,
            realized_pnl: Some(87.50),
            gaming_message: "💰 Profit secured! SOL position closed with epic gains!".to_string(),
            created_at: chrono::Utc::now() - chrono::Duration::hours(2),
            completed_at: Some(chrono::Utc::now() - chrono::Duration::hours(2)),
        }
    ];
    
    // Apply status filter
    if let Some(status_filter) = &params.status {
        trades.retain(|t| t.status == *status_filter);
    }
    
    // Apply pagination
    let per_page = params.per_page.unwrap_or(50);
    let page = params.page.unwrap_or(1);
    let start = ((page - 1) * per_page) as usize;
    let end = (start + per_page as usize).min(trades.len());
    
    if start < trades.len() {
        trades[start..end].to_vec()
    } else {
        vec![]
    }
}

async fn calculate_trading_stats(trades: &[Trade]) -> TradingStats {
    let completed_trades: Vec<&Trade> = trades.iter().filter(|t| t.status == "COMPLETED").collect();
    let total_volume: f64 = completed_trades.iter().map(|t| t.total_value).sum();
    let winning_trades: Vec<&Trade> = completed_trades.iter().filter(|t| t.realized_pnl.unwrap_or(0.0) > 0.0).collect();
    let losing_trades: Vec<&Trade> = completed_trades.iter().filter(|t| t.realized_pnl.unwrap_or(0.0) < 0.0).collect();
    
    let total_pnl: f64 = completed_trades.iter().map(|t| t.realized_pnl.unwrap_or(0.0)).sum();
    let win_rate = if completed_trades.is_empty() { 0.0 } else { 
        (winning_trades.len() as f64 / completed_trades.len() as f64) * 100.0 
    };
    
    let average_win = if winning_trades.is_empty() { 0.0 } else {
        winning_trades.iter().map(|t| t.realized_pnl.unwrap_or(0.0)).sum::<f64>() / winning_trades.len() as f64
    };
    
    let average_loss = if losing_trades.is_empty() { 0.0 } else {
        losing_trades.iter().map(|t| t.realized_pnl.unwrap_or(0.0)).sum::<f64>() / losing_trades.len() as f64
    };
    
    TradingStats {
        total_trades: trades.len() as u32,
        completed_trades: completed_trades.len() as u32,
        pending_trades: trades.iter().filter(|t| t.status == "PENDING").count() as u32,
        failed_trades: trades.iter().filter(|t| t.status == "FAILED").count() as u32,
        total_volume,
        total_pnl,
        win_rate,
        profit_factor: if average_loss == 0.0 { 0.0 } else { average_win / average_loss.abs() },
        average_win,
        average_loss,
        largest_win: winning_trades.iter().map(|t| t.realized_pnl.unwrap_or(0.0)).fold(0.0, f64::max),
        largest_loss: losing_trades.iter().map(|t| t.realized_pnl.unwrap_or(0.0)).fold(0.0, f64::min),
        total_gas_used: trades.iter().map(|t| t.gas_used).sum(),
    }
}

async fn generate_trading_gaming_messages(
    app_state: &Arc<AppState>,
    gaming_theme: &Option<String>,
    stats: &TradingStats,
) -> TradingGamingMessages {
    let context = crate::integrations::trenchware_ui::MessageContext {
        profit_amount: Some(stats.total_pnl),
        ..Default::default()
    };
    
    TradingGamingMessages {
        performance_summary: app_state.trenchware_ui.get_gaming_message("trading_performance", context),
        win_rate_message: if stats.win_rate > 70.0 {
            match gaming_theme.as_deref() {
                Some("runescape") => "⚔️ LEGENDARY STATUS! Your win rate is worthy of the Hall of Fame!",
                Some("cod") => "🎖️ ACE PILOT! Your accuracy is unmatched on the battlefield!",
                Some("crypto") => "💎 DIAMOND HANDS CHAMPION! Your trading skills are elite!",
                _ => "🏆 TRADING MASTER! Your win rate is absolutely crushing it!"
            }
        } else if stats.win_rate > 50.0 {
            "📈 Solid performance! You're beating the market average!"
        } else {
            "🎯 Keep grinding! Every pro trader started somewhere!"
        }.to_string(),
        volume_message: format!("💰 Total volume traded: ${:.2} - You're actively engaging the market!", stats.total_volume),
        gas_efficiency: if stats.total_gas_used < 0.01 {
            "⚡ Gas efficiency: EXCELLENT! You're minimizing transaction costs like a pro!"
        } else {
            "⛽ Consider optimizing gas usage - every satoshi counts!"
        }.to_string(),
        motivational_tip: match gaming_theme.as_deref() {
            Some("runescape") => "💡 Pro tip: Just like skilling in RS, consistent trading levels up your abilities! ⚔️".to_string(),
            Some("cod") => "💡 Remember: Know your enemy (the market) and choose your battles wisely! 🪖".to_string(),
            _ => "💡 Trading wisdom: Plan your trades and trade your plan! 📈".to_string(),
        },
    }
}

async fn validate_trade_request(
    _app_state: &Arc<AppState>,
    user_session: &crate::api::server::UserSession,
    payload: &ExecuteTradeRequest,
) -> Result<(), String> {
    // Check if user has necessary permissions
    if !user_session.permissions.iter().any(|p| matches!(p, crate::api::server::Permission::ExecuteTrades)) {
        return Err("Insufficient permissions to execute trades".to_string());
    }
    
    // Validate trade parameters
    if payload.amount <= 0.0 {
        return Err("Trade amount must be positive".to_string());
    }
    
    if payload.price <= 0.0 {
        return Err("Trade price must be positive".to_string());
    }
    
    // Check for minimum trade size
    let trade_value = payload.amount * payload.price;
    if trade_value < 1.0 {
        return Err("Minimum trade value is $1.00".to_string());
    }
    
    // Add more validation logic as needed
    Ok(())
}

async fn execute_trade_on_blockchain(
    _app_state: &Arc<AppState>,
    _user_session: &crate::api::server::UserSession,
    payload: &ExecuteTradeRequest,
) -> Result<TradeResult> {
    // Simulate blockchain interaction
    // In real implementation, this would interact with Solana blockchain
    
    let trade_id = format!("trade_{}", uuid::Uuid::new_v4().to_string()[0..8].to_string());
    let transaction_hash = format!("{}...{}", 
        &uuid::Uuid::new_v4().to_string()[0..4],
        &uuid::Uuid::new_v4().to_string()[32..36]
    );
    
    // Simulate network delay
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
    
    // Simulate success/failure (95% success rate)
    if rand::random::<f64>() < 0.95 {
        Ok(TradeResult {
            trade_id,
            transaction_hash,
            status: "COMPLETED".to_string(),
            gas_used: 0.001234,
            confirmation_time: 3.2, // seconds
            realized_pnl: Some(calculate_simulated_pnl(&payload.action, payload.amount, payload.price)),
        })
    } else {
        Err(anyhow::anyhow!("Network congestion - transaction failed"))
    }
}

fn calculate_simulated_pnl(action: &str, amount: f64, price: f64) -> f64 {
    // Simulate PnL based on action and market conditions
    let market_movement = (rand::random::<f64>() - 0.5) * 0.1; // ±5% movement
    let trade_value = amount * price;
    
    match action {
        "BUY" => -trade_value * 0.001, // Small negative for immediate PnL (spread cost)
        "SELL" => trade_value * market_movement,
        _ => 0.0,
    }
}

fn get_rejection_gaming_message(gaming_theme: &Option<String>, error_msg: &str) -> String {
    match gaming_theme.as_deref() {
        Some("runescape") => format!("⚔️ Quest failed: {}! Check your inventory and try again!", error_msg),
        Some("cod") => format!("🪖 Mission aborted: {}! Recheck your loadout, soldier!", error_msg),
        Some("crypto") => format!("💎 Diamond hands rejected: {}! HODL and reassess!", error_msg),
        Some("fortnite") => format!("🏆 Storm caught you: {}! Drop again with better gear!", error_msg),
        _ => format!("🚫 Trade rejected: {}!", error_msg),
    }
}

fn generate_success_gaming_message(
    gaming_theme: &Option<String>,
    action: &str,
    token_symbol: &str,
    amount: f64,
    status: &str,
) -> String {
    let base_message = match gaming_theme.as_deref() {
        Some("runescape") => match action {
            "BUY" => format!("⚔️ QUEST COMPLETE! {} {} acquired! Your trading level increases!"),
            "SELL" => format!("💰 LOOT SECURED! {} {} sold for epic profits!"),
            _ => format!("🏰 Trade executed successfully! {} {} {}!"),
        },
        Some("cod") => match action {
            "BUY" => format!("🎯 TARGET ACQUIRED! {} {} locked and loaded!"),
            "SELL" => format!("💥 MISSION ACCOMPLISHED! {} {} eliminated with precision!"),
            _ => format!("🪖 Operation successful! {} {} {}!"),
        },
        Some("crypto") => match action {
            "BUY" => format!("🚀 TO THE MOON! {} {} acquired with diamond hands!"),
            "SELL" => format!("💎 PROFIT SECURED! {} {} sold at peak performance!"),
            _ => format!("💰 Trade executed! {} {} {}!"),
        },
        Some("fortnite") => match action {
            "BUY" => format!("🏆 LOOT ACQUIRED! {} {} added to inventory!"),
            "SELL" => format!("👑 VICTORY ROYALE! {} {} eliminated for the win!"),
            _ => format!("🎮 Play executed! {} {} {}!"),
        },
        _ => match action {
            "BUY" => format!("📈 Purchase successful! {} {} acquired!"),
            "SELL" => format!("💰 Sale completed! {} {} sold!"),
            _ => format!("✅ Trade executed! {} {} {}!"),
        }
    };
    
    format!(base_message, amount, token_symbol, status)
}

async fn send_trade_notifications(
    app_state: &Arc<AppState>,
    user_session: &crate::api::server::UserSession,
    trade_result: &TradeResult,
    gaming_message: &str,
) -> Result<()> {
    // Send through notification system
    // This would integrate with the notification system
    info!("📢 Sending trade notification to user {}: {}", user_session.user_id, gaming_message);
    Ok(())
}

async fn update_trading_achievements(
    _app_state: &Arc<AppState>,
    user_id: &str,
    _trade_result: &TradeResult,
) -> Result<()> {
    // Update achievements and leaderboards
    // This would integrate with the ranking system
    info!("🏆 Updating achievements for user {}", user_id);
    Ok(())
}

async fn get_historical_trades(
    _app_state: &Arc<AppState>,
    user_session: &crate::api::server::UserSession,
    params: &TradeHistoryParams,
) -> Vec<Trade> {
    // Generate historical trades based on period
    let days = match params.period.as_str() {
        "1d" => 1,
        "7d" => 7,
        "30d" => 30,
        "1y" => 365,
        _ => 30,
    };
    
    let mut trades = vec![];
    for i in 0..days.min(100) { // Limit to 100 trades for demo
        let trade = Trade {
            trade_id: format!("trade_hist_{}", i),
            user_id: user_session.user_id.clone(),
            token_symbol: if i % 3 == 0 { "SOL" } else if i % 3 == 1 { "BONK" } else { "USDC" }.to_string(),
            token_address: "placeholder".to_string(),
            action: if i % 2 == 0 { "BUY" } else { "SELL" }.to_string(),
            amount: 100.0 + (i as f64 * 10.0),
            price: 1.0 + (rand::random::<f64>() * 0.1),
            total_value: 100.0 + (i as f64 * 10.0),
            status: "COMPLETED".to_string(),
            transaction_hash: format!("hash_{}", i),
            gas_used: 0.001,
            realized_pnl: Some((rand::random::<f64>() - 0.5) * 50.0),
            gaming_message: "Trade completed!".to_string(),
            created_at: chrono::Utc::now() - chrono::Duration::days(i as i64),
            completed_at: Some(chrono::Utc::now() - chrono::Duration::days(i as i64)),
        };
        trades.push(trade);
    }
    
    trades
}

async fn calculate_trade_analytics(trades: &[Trade], _period: &str) -> TradeAnalytics {
    let completed_trades: Vec<&Trade> = trades.iter().filter(|t| t.status == "COMPLETED").collect();
    let total_pnl: f64 = completed_trades.iter().map(|t| t.realized_pnl.unwrap_or(0.0)).sum();
    let winning_trades = completed_trades.iter().filter(|t| t.realized_pnl.unwrap_or(0.0) > 0.0).count();
    
    TradeAnalytics {
        total_volume: completed_trades.iter().map(|t| t.total_value).sum(),
        net_pnl: total_pnl,
        gross_profit: completed_trades.iter().map(|t| t.realized_pnl.unwrap_or(0.0).max(0.0)).sum(),
        gross_loss: completed_trades.iter().map(|t| t.realized_pnl.unwrap_or(0.0).min(0.0)).sum(),
        win_rate: if completed_trades.is_empty() { 0.0 } else { 
            (winning_trades as f64 / completed_trades.len() as f64) * 100.0 
        },
        profit_factor: 2.45, // Placeholder calculation
        average_trade: if completed_trades.is_empty() { 0.0 } else {
            total_pnl / completed_trades.len() as f64
        },
        max_consecutive_wins: 5, // Placeholder
        max_consecutive_losses: 3, // Placeholder
        total_fees: completed_trades.iter().map(|t| t.gas_used).sum::<f64>() * 200.0, // Estimate fee in USD
    }
}

async fn calculate_trading_performance(_trades: &[Trade]) -> TradingPerformance {
    TradingPerformance {
        sharpe_ratio: 2.34,
        sortino_ratio: 2.67,
        calmar_ratio: 5.33,
        max_drawdown: -8.5,
        recovery_factor: 1.85,
        profit_to_max_dd_ratio: 2.89,
        risk_reward_ratio: 1.67,
        expectancy: 12.45,
        kelly_criterion: 0.25,
        var_95: -0.08,
    }
}

async fn generate_trading_insights(
    _app_state: &Arc<AppState>,
    gaming_theme: &Option<String>,
    analytics: &TradeAnalytics,
    _performance: &TradingPerformance,
) -> Vec<TradingInsight> {
    let mut insights = vec![];
    
    if analytics.win_rate > 70.0 {
        insights.push(TradingInsight {
            insight_type: "PERFORMANCE".to_string(),
            title: "Exceptional Win Rate".to_string(),
            description: format!("Your {:.1}% win rate is outstanding! You're clearly skilled at market timing.", analytics.win_rate),
            gaming_message: match gaming_theme.as_deref() {
                Some("runescape") => "🏆 LEGENDARY TRADER! Your combat level is maxed out! ⚔️".to_string(),
                Some("cod") => "🎖️ ELITE OPERATOR! Your K/D ratio is off the charts! 🪖".to_string(),
                _ => "🌟 TRADING MASTER! Your skills are truly exceptional! 📈".to_string(),
            },
            actionable: false,
        });
    }
    
    if analytics.total_fees > 100.0 {
        insights.push(TradingInsight {
            insight_type: "COST_OPTIMIZATION".to_string(),
            title: "High Trading Fees".to_string(),
            description: format!("You've paid ${:.2} in fees. Consider optimizing your trade timing.", analytics.total_fees),
            gaming_message: "⚠️ Your ammo costs are adding up! Consider more strategic shot placement to save resources!".to_string(),
            actionable: true,
        });
    }
    
    insights
}

/// **REQUEST/RESPONSE TYPES**

#[derive(Debug, Deserialize)]
pub struct GetTradesParams {
    pub session_token: Option<String>,
    pub gaming_theme: Option<String>,
    pub status: Option<String>, // "COMPLETED", "PENDING", "FAILED"
    pub page: Option<u32>,
    pub per_page: Option<u32>,
}

#[derive(Debug, Deserialize)]
pub struct ExecuteTradeRequest {
    pub session_token: String,
    pub gaming_theme: Option<String>,
    pub token_symbol: String,
    pub token_address: String,
    pub action: String, // "BUY", "SELL"
    pub amount: f64,
    pub price: f64,
    pub order_type: String, // "MARKET", "LIMIT", "STOP"
    pub slippage: Option<f64>,
    pub gas_preference: Option<String>, // "SLOW", "NORMAL", "FAST"
}

#[derive(Debug, Deserialize)]
pub struct TradeHistoryParams {
    pub session_token: Option<String>,
    pub gaming_theme: Option<String>,
    pub period: String, // "1d", "7d", "30d", "1y"
}

#[derive(Debug, Serialize)]
pub struct GetTradesResponse {
    pub trades: Vec<Trade>,
    pub stats: TradingStats,
    pub gaming_messages: TradingGamingMessages,
    pub pagination: PaginationInfo,
    pub filters_applied: Option<String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Serialize)]
pub struct ExecuteTradeResponse {
    pub success: bool,
    pub trade_id: Option<String>,
    pub status: String,
    pub error_message: Option<String>,
    pub gaming_message: String,
    pub estimated_gas: Option<f64>,
    pub confirmation_time: Option<f64>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Serialize)]
pub struct TradeHistoryResponse {
    pub period: String,
    pub total_trades: u32,
    pub successful_trades: u32,
    pub failed_trades: u32,
    pub analytics: TradeAnalytics,
    pub performance: TradingPerformance,
    pub insights: Vec<TradingInsight>,
    pub recent_trades: Vec<Trade>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Serialize)]
pub struct Trade {
    pub trade_id: String,
    pub user_id: String,
    pub token_symbol: String,
    pub token_address: String,
    pub action: String,
    pub amount: f64,
    pub price: f64,
    pub total_value: f64,
    pub status: String,
    pub transaction_hash: String,
    pub gas_used: f64,
    pub realized_pnl: Option<f64>,
    pub gaming_message: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub completed_at: Option<chrono::DateTime<chrono::Utc>>,
}

#[derive(Debug, Serialize)]
pub struct TradingStats {
    pub total_trades: u32,
    pub completed_trades: u32,
    pub pending_trades: u32,
    pub failed_trades: u32,
    pub total_volume: f64,
    pub total_pnl: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub average_win: f64,
    pub average_loss: f64,
    pub largest_win: f64,
    pub largest_loss: f64,
    pub total_gas_used: f64,
}

#[derive(Debug, Serialize)]
pub struct TradingGamingMessages {
    pub performance_summary: String,
    pub win_rate_message: String,
    pub volume_message: String,
    pub gas_efficiency: String,
    pub motivational_tip: String,
}

#[derive(Debug, Serialize)]
pub struct PaginationInfo {
    pub page: u32,
    pub per_page: u32,
    pub total_pages: u32,
    pub total_trades: u32,
}

#[derive(Debug)]
pub struct TradeResult {
    pub trade_id: String,
    pub transaction_hash: String,
    pub status: String,
    pub gas_used: f64,
    pub confirmation_time: f64,
    pub realized_pnl: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct TradeAnalytics {
    pub total_volume: f64,
    pub net_pnl: f64,
    pub gross_profit: f64,
    pub gross_loss: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub average_trade: f64,
    pub max_consecutive_wins: u32,
    pub max_consecutive_losses: u32,
    pub total_fees: f64,
}

#[derive(Debug, Serialize)]
pub struct TradingPerformance {
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub calmar_ratio: f64,
    pub max_drawdown: f64,
    pub recovery_factor: f64,
    pub profit_to_max_dd_ratio: f64,
    pub risk_reward_ratio: f64,
    pub expectancy: f64,
    pub kelly_criterion: f64,
    pub var_95: f64,
}

#[derive(Debug, Serialize)]
pub struct TradingInsight {
    pub insight_type: String,
    pub title: String,
    pub description: String,
    pub gaming_message: String,
    pub actionable: bool,
}