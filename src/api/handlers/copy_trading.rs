use anyhow::Result;
use axum::{
    extract::{State, Query, Json as ExtractJson},
    response::Json,
    http::StatusCode,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{info, warn, debug};

use crate::api::server::AppState;

/// **GET TRADERS ENDPOINT**
/// Get list of available traders to copy with gaming profiles
pub async fn get_traders(
    State(app_state): State<Arc<AppState>>,
    Query(params): Query<GetTradersParams>,
) -> Result<Json<GetTradersResponse>, StatusCode> {
    info!("🤝 Copy trading traders list request");
    
    let user_session = if let Some(token) = &params.session_token {
        app_state.auth_manager.verify_session(token).await
            .map_err(|_| StatusCode::UNAUTHORIZED)?
    } else {
        return Err(StatusCode::UNAUTHORIZED);
    };
    
    // Get available traders
    let traders = get_available_traders(&app_state, &params).await;
    let leaderboard = get_traders_leaderboard(&app_state).await;
    let featured_traders = get_featured_traders(&app_state).await;
    
    // Generate gaming-themed messages
    let gaming_messages = generate_traders_gaming_messages(
        &app_state,
        &params.gaming_theme,
        &traders
    ).await;
    
    let response = GetTradersResponse {
        traders,
        leaderboard,
        featured_traders,
        gaming_messages,
        filters_applied: TraderFilters {
            min_win_rate: params.min_win_rate,
            min_followers: params.min_followers,
            combat_class: params.combat_class.clone(),
            risk_level: params.risk_level.clone(),
        },
        total_traders: 156, // Placeholder
        timestamp: chrono::Utc::now(),
    };
    
    debug!("✅ Traders data compiled successfully");
    Ok(Json(response))
}

/// **FOLLOW TRADER ENDPOINT**
/// Follow a trader and start copy trading
pub async fn follow_trader(
    State(app_state): State<Arc<AppState>>,
    ExtractJson(payload): ExtractJson<FollowTraderRequest>,
) -> Result<Json<FollowTraderResponse>, StatusCode> {
    info!("👥 Follow trader request: {} following {}", 
          payload.follower_id, payload.trader_id);
    
    let user_session = app_state.auth_manager.verify_session(&payload.session_token).await
        .map_err(|_| StatusCode::UNAUTHORIZED)?;
    
    // Validate follow request
    let validation_result = validate_follow_request(&app_state, &user_session, &payload).await;
    if let Err(error_msg) = validation_result {
        warn!("🚫 Follow validation failed: {}", error_msg);
        return Ok(Json(FollowTraderResponse {
            success: false,
            copy_trading_id: None,
            error_message: Some(error_msg),
            gaming_message: get_follow_rejection_message(&payload.gaming_theme, &error_msg),
            trader_profile: None,
            estimated_costs: None,
            timestamp: chrono::Utc::now(),
        }));
    }
    
    // Execute follow action
    match execute_follow_trader(&app_state, &user_session, &payload).await {
        Ok(copy_trading_result) => {
            info!("🎉 Successfully started following trader: {}", payload.trader_id);
            
            // Get trader profile for response
            let trader_profile = get_trader_profile(&app_state, &payload.trader_id).await.unwrap_or_default();
            
            // Generate success gaming message
            let gaming_message = generate_follow_success_message(
                &payload.gaming_theme,
                &trader_profile.display_name,
                &trader_profile.combat_class
            );
            
            // Send notifications
            let _ = send_follow_notifications(&app_state, &user_session, &trader_profile, &gaming_message).await;
            
            Ok(Json(FollowTraderResponse {
                success: true,
                copy_trading_id: Some(copy_trading_result.copy_trading_id),
                error_message: None,
                gaming_message,
                trader_profile: Some(trader_profile),
                estimated_costs: Some(copy_trading_result.estimated_monthly_cost),
                timestamp: chrono::Utc::now(),
            }))
        },
        Err(e) => {
            warn!("💥 Failed to follow trader: {}", e);
            
            let gaming_message = match payload.gaming_theme.as_deref() {
                Some("runescape") => "⚔️ Failed to join the party! Try again when the server is less busy!",
                Some("cod") => "🪖 Squad join failed! Regroup and try again, soldier!",
                Some("crypto") => "💎 Copy trading activation failed! Diamond hands patience needed!",
                Some("fortnite") => "🏆 Team join failed! Drop again and try to squad up!",
                _ => "🤝 Failed to start copy trading - please try again!"
            }.to_string();
            
            Ok(Json(FollowTraderResponse {
                success: false,
                copy_trading_id: None,
                error_message: Some(e.to_string()),
                gaming_message,
                trader_profile: None,
                estimated_costs: None,
                timestamp: chrono::Utc::now(),
            }))
        }
    }
}

/// **UNFOLLOW TRADER ENDPOINT**
/// Stop following a trader and end copy trading
pub async fn unfollow_trader(
    State(app_state): State<Arc<AppState>>,
    ExtractJson(payload): ExtractJson<UnfollowTraderRequest>,
) -> Result<Json<UnfollowTraderResponse>, StatusCode> {
    info!("👋 Unfollow trader request: {} unfollowing {}", 
          payload.follower_id, payload.trader_id);
    
    let user_session = app_state.auth_manager.verify_session(&payload.session_token).await
        .map_err(|_| StatusCode::UNAUTHORIZED)?;
    
    // Execute unfollow action
    match execute_unfollow_trader(&app_state, &user_session, &payload).await {
        Ok(final_stats) => {
            info!("✅ Successfully unfollowed trader: {}", payload.trader_id);
            
            // Generate farewell gaming message
            let gaming_message = generate_unfollow_message(
                &payload.gaming_theme,
                &final_stats.total_pnl,
                final_stats.trades_copied
            );
            
            Ok(Json(UnfollowTraderResponse {
                success: true,
                gaming_message,
                final_stats: Some(final_stats),
                timestamp: chrono::Utc::now(),
            }))
        },
        Err(e) => {
            warn!("💥 Failed to unfollow trader: {}", e);
            
            Ok(Json(UnfollowTraderResponse {
                success: false,
                gaming_message: "Failed to end copy trading relationship".to_string(),
                final_stats: None,
                timestamp: chrono::Utc::now(),
            }))
        }
    }
}

/// **GET COPY TRADING PERFORMANCE ENDPOINT**
/// Get performance data for copy trading activities
pub async fn get_performance(
    State(app_state): State<Arc<AppState>>,
    Query(params): Query<CopyTradingPerformanceParams>,
) -> Result<Json<CopyTradingPerformanceResponse>, StatusCode> {
    info!("📊 Copy trading performance request");
    
    let user_session = if let Some(token) = &params.session_token {
        app_state.auth_manager.verify_session(token).await
            .map_err(|_| StatusCode::UNAUTHORIZED)?
    } else {
        return Err(StatusCode::UNAUTHORIZED);
    };
    
    // Get copy trading performance data
    let following = get_following_traders(&app_state, &user_session).await;
    let performance = calculate_copy_trading_performance(&app_state, &user_session, &following).await;
    let analytics = calculate_copy_trading_analytics(&following, &performance).await;
    
    // Generate insights with gaming themes
    let insights = generate_copy_trading_insights(
        &app_state,
        &params.gaming_theme,
        &performance,
        &analytics
    ).await;
    
    let response = CopyTradingPerformanceResponse {
        following,
        performance,
        analytics,
        insights,
        total_invested: calculate_total_invested(&following),
        monthly_fees: calculate_monthly_fees(&following),
        profit_sharing_paid: calculate_profit_sharing(&performance),
        timestamp: chrono::Utc::now(),
    };
    
    Ok(Json(response))
}

/// **SUPPORTING FUNCTIONS**

async fn get_available_traders(_app_state: &Arc<AppState>, params: &GetTradersParams) -> Vec<TraderProfile> {
    // Placeholder implementation - would query database
    let mut traders = vec![
        TraderProfile {
            trader_id: "trader_001".to_string(),
            display_name: "SolanaSamurai".to_string(),
            combat_class: "MemelordGeneral".to_string(),
            win_rate: 78.5,
            total_followers: 1247,
            total_pnl: 156789.45,
            total_pnl_percent: 234.67,
            risk_score: 6.8,
            max_drawdown: -12.3,
            average_trade_size: 2500.0,
            total_trades: 456,
            gaming_title: "🏆 Legendary Meme Commander".to_string(),
            gaming_description: "⚔️ Master of degen plays! Leads from the front lines with diamond hands and steel nerves!".to_string(),
            ranking: "Diamond I".to_string(),
            verified: true,
            subscription_fee: 0.015, // 1.5% monthly
            profit_share: 0.20, // 20% of profits
            minimum_copy_amount: 100.0,
            last_active: chrono::Utc::now() - chrono::Duration::minutes(5),
        },
        TraderProfile {
            trader_id: "trader_002".to_string(),
            display_name: "WhaleHunterPro".to_string(),
            combat_class: "WhaleHunter".to_string(),
            win_rate: 65.2,
            total_followers: 892,
            total_pnl: 89456.78,
            total_pnl_percent: 145.23,
            risk_score: 8.2,
            max_drawdown: -18.7,
            average_trade_size: 5000.0,
            total_trades: 234,
            gaming_title: "🐋 Elite Whale Tracker".to_string(),
            gaming_description: "🎯 Hunts the biggest moves in crypto! High risk, high reward specialist!".to_string(),
            ranking: "Platinum III".to_string(),
            verified: true,
            subscription_fee: 0.025, // 2.5% monthly
            profit_share: 0.25, // 25% of profits
            minimum_copy_amount: 500.0,
            last_active: chrono::Utc::now() - chrono::Duration::minutes(2),
        },
        TraderProfile {
            trader_id: "trader_003".to_string(),
            display_name: "ScalperNinja".to_string(),
            combat_class: "ScalperAssassin".to_string(),
            win_rate: 82.1,
            total_followers: 567,
            total_pnl: 34567.89,
            total_pnl_percent: 78.45,
            risk_score: 4.5,
            max_drawdown: -8.9,
            average_trade_size: 800.0,
            total_trades: 1234,
            gaming_title: "⚡ Lightning Fast Executor".to_string(),
            gaming_description: "🥷 In and out before you blink! Perfect for quick profits and minimal risk!".to_string(),
            ranking: "Gold II".to_string(),
            verified: false,
            subscription_fee: 0.01, // 1% monthly
            profit_share: 0.15, // 15% of profits
            minimum_copy_amount: 50.0,
            last_active: chrono::Utc::now() - chrono::Duration::minutes(1),
        }
    ];
    
    // Apply filters
    if let Some(min_win_rate) = params.min_win_rate {
        traders.retain(|t| t.win_rate >= min_win_rate);
    }
    
    if let Some(min_followers) = params.min_followers {
        traders.retain(|t| t.total_followers >= min_followers);
    }
    
    if let Some(combat_class) = &params.combat_class {
        traders.retain(|t| t.combat_class == *combat_class);
    }
    
    // Sort by performance
    traders.sort_by(|a, b| b.total_pnl_percent.partial_cmp(&a.total_pnl_percent).unwrap());
    
    traders
}

async fn get_traders_leaderboard(_app_state: &Arc<AppState>) -> Vec<LeaderboardEntry> {
    vec![
        LeaderboardEntry {
            rank: 1,
            trader_id: "trader_001".to_string(),
            display_name: "SolanaSamurai".to_string(),
            combat_class: "MemelordGeneral".to_string(),
            total_pnl_percent: 234.67,
            followers: 1247,
            gaming_title: "🏆 Supreme Meme Lord".to_string(),
        },
        LeaderboardEntry {
            rank: 2,
            trader_id: "trader_002".to_string(),
            display_name: "WhaleHunterPro".to_string(),
            combat_class: "WhaleHunter".to_string(),
            total_pnl_percent: 145.23,
            followers: 892,
            gaming_title: "🐋 Apex Predator".to_string(),
        },
        LeaderboardEntry {
            rank: 3,
            trader_id: "trader_003".to_string(),
            display_name: "ScalperNinja".to_string(),
            combat_class: "ScalperAssassin".to_string(),
            total_pnl_percent: 78.45,
            followers: 567,
            gaming_title: "⚡ Speed Demon".to_string(),
        }
    ]
}

async fn get_featured_traders(_app_state: &Arc<AppState>) -> Vec<FeaturedTrader> {
    vec![
        FeaturedTrader {
            trader_id: "trader_001".to_string(),
            display_name: "SolanaSamurai".to_string(),
            feature_reason: "TOP_PERFORMER".to_string(),
            feature_title: "🏆 This Week's Champion".to_string(),
            feature_description: "Dominating the leaderboards with consistent gains and legendary meme trades!".to_string(),
            special_badge: Some("👑 Featured Trader".to_string()),
        }
    ]
}

async fn generate_traders_gaming_messages(
    app_state: &Arc<AppState>,
    gaming_theme: &Option<String>,
    traders: &[TraderProfile],
) -> TradersGamingMessages {
    let context = crate::integrations::trenchware_ui::MessageContext::default();
    
    TradersGamingMessages {
        welcome_message: app_state.trenchware_ui.get_gaming_message("copy_trading_welcome", context),
        selection_tip: match gaming_theme.as_deref() {
            Some("runescape") => "💡 Choose your party members wisely! Different combat classes excel in different situations! ⚔️".to_string(),
            Some("cod") => "💡 Pick your squad carefully! Each specialist brings unique tactical advantages! 🪖".to_string(),
            Some("crypto") => "💡 Diamond hands recognize diamond hands! Follow traders with proven HODL strength! 💎".to_string(),
            _ => "💡 Research trader performance and risk levels before following! 📊".to_string(),
        },
        risk_warning: "⚠️ Copy trading involves risk! Only invest what you can afford to lose!".to_string(),
        top_performer_highlight: if !traders.is_empty() {
            format!("🌟 {} is currently leading with {:.1}% returns!", 
                   traders[0].display_name, traders[0].total_pnl_percent)
        } else {
            "🔍 No traders available with current filters".to_string()
        },
    }
}

async fn validate_follow_request(
    _app_state: &Arc<AppState>,
    user_session: &crate::api::server::UserSession,
    payload: &FollowTraderRequest,
) -> Result<(), String> {
    // Check if user has copy trading permissions
    if !user_session.permissions.iter().any(|p| matches!(p, crate::api::server::Permission::CopyTrade)) {
        return Err("Insufficient permissions for copy trading".to_string());
    }
    
    // Check minimum investment amount
    if payload.copy_amount < 10.0 {
        return Err("Minimum copy amount is $10".to_string());
    }
    
    // Check if user is trying to follow themselves (if they were a trader)
    if payload.follower_id == payload.trader_id {
        return Err("Cannot copy trade your own strategies".to_string());
    }
    
    // Add more validation logic as needed
    Ok(())
}

async fn execute_follow_trader(
    _app_state: &Arc<AppState>,
    _user_session: &crate::api::server::UserSession,
    payload: &FollowTraderRequest,
) -> Result<CopyTradingResult> {
    // Simulate copy trading setup
    let copy_trading_id = format!("copy_{}", uuid::Uuid::new_v4().to_string()[0..8].to_string());
    
    // Calculate estimated monthly cost
    let monthly_fee = payload.copy_amount * 0.015; // 1.5% default
    
    Ok(CopyTradingResult {
        copy_trading_id,
        estimated_monthly_cost: monthly_fee,
    })
}

async fn execute_unfollow_trader(
    _app_state: &Arc<AppState>,
    _user_session: &crate::api::server::UserSession,
    _payload: &UnfollowTraderRequest,
) -> Result<CopyTradingFinalStats> {
    // Simulate final stats calculation
    Ok(CopyTradingFinalStats {
        total_duration_days: 45,
        trades_copied: 23,
        total_pnl: 156.78,
        total_pnl_percent: 12.34,
        total_fees_paid: 67.89,
        best_trade_pnl: 45.67,
        worst_trade_pnl: -12.34,
    })
}

fn get_follow_rejection_message(gaming_theme: &Option<String>, error_msg: &str) -> String {
    match gaming_theme.as_deref() {
        Some("runescape") => format!("⚔️ Party join failed: {}! Check your inventory and combat level!", error_msg),
        Some("cod") => format!("🪖 Squad recruitment denied: {}! Verify your credentials, soldier!", error_msg),
        Some("crypto") => format!("💎 Copy trading rejected: {}! Check your wallet and try again!", error_msg),
        Some("fortnite") => format!("🏆 Team join blocked: {}! Make sure you meet the requirements!", error_msg),
        _ => format!("🚫 Copy trading failed: {}!", error_msg),
    }
}

fn generate_follow_success_message(
    gaming_theme: &Option<String>,
    trader_name: &str,
    combat_class: &str,
) -> String {
    match gaming_theme.as_deref() {
        Some("runescape") => format!("⚔️ PARTY JOINED! You're now following {} the {}! Adventure awaits!", trader_name, combat_class),
        Some("cod") => format!("🪖 SQUAD UP! {} ({}) is now your commanding officer! Ready for combat!", trader_name, combat_class),
        Some("crypto") => format!("💎 DIAMOND TEAM ASSEMBLED! Following {} for maximum gains potential! 🚀", trader_name),
        Some("fortnite") => format!("🏆 SQUAD FORMED! {} is your team leader! Let's get that Victory Royale!", trader_name),
        _ => format!("🤝 Successfully started following {}! Copy trading activated!", trader_name),
    }
}

fn generate_unfollow_message(
    gaming_theme: &Option<String>,
    total_pnl: &f64,
    trades_copied: u32,
) -> String {
    let performance = if *total_pnl > 0.0 { "profitable" } else { "learning experience" };
    
    match gaming_theme.as_deref() {
        Some("runescape") => format!("⚔️ QUEST COMPLETE! {} trades copied with ${:.2} result! Time for a new adventure!", trades_copied, total_pnl),
        Some("cod") => format!("🪖 MISSION CONCLUDED! {} operations completed with ${:.2} outcome! Well fought, soldier!", trades_copied, total_pnl),
        Some("crypto") => format!("💎 TRADING PARTNERSHIP ENDED! {} trades, ${:.2} {} - keep those diamond hands strong!", trades_copied, total_pnl, performance),
        Some("fortnite") => format!("🏆 SQUAD DISBANDED! {} plays completed with ${:.2} result! GG to your former teammate!", trades_copied, total_pnl),
        _ => format!("👋 Copy trading ended! {} trades copied with ${:.2} total result.", trades_copied, total_pnl),
    }
}

async fn send_follow_notifications(
    _app_state: &Arc<AppState>,
    user_session: &crate::api::server::UserSession,
    trader_profile: &TraderProfile,
    gaming_message: &str,
) -> Result<()> {
    info!("📢 Sending follow notification to user {}: Started following {} - {}", 
          user_session.user_id, trader_profile.display_name, gaming_message);
    Ok(())
}

async fn get_trader_profile(_app_state: &Arc<AppState>, trader_id: &str) -> Option<TraderProfile> {
    // Placeholder - would query database
    Some(TraderProfile {
        trader_id: trader_id.to_string(),
        display_name: "SolanaSamurai".to_string(),
        combat_class: "MemelordGeneral".to_string(),
        win_rate: 78.5,
        total_followers: 1247,
        total_pnl: 156789.45,
        total_pnl_percent: 234.67,
        risk_score: 6.8,
        max_drawdown: -12.3,
        average_trade_size: 2500.0,
        total_trades: 456,
        gaming_title: "🏆 Legendary Meme Commander".to_string(),
        gaming_description: "⚔️ Master of degen plays!".to_string(),
        ranking: "Diamond I".to_string(),
        verified: true,
        subscription_fee: 0.015,
        profit_share: 0.20,
        minimum_copy_amount: 100.0,
        last_active: chrono::Utc::now(),
    })
}

async fn get_following_traders(
    _app_state: &Arc<AppState>,
    _user_session: &crate::api::server::UserSession,
) -> Vec<FollowingTrader> {
    vec![
        FollowingTrader {
            copy_trading_id: "copy_001".to_string(),
            trader_profile: TraderProfile {
                trader_id: "trader_001".to_string(),
                display_name: "SolanaSamurai".to_string(),
                combat_class: "MemelordGeneral".to_string(),
                win_rate: 78.5,
                total_followers: 1247,
                total_pnl: 156789.45,
                total_pnl_percent: 234.67,
                risk_score: 6.8,
                max_drawdown: -12.3,
                average_trade_size: 2500.0,
                total_trades: 456,
                gaming_title: "🏆 Legendary Meme Commander".to_string(),
                gaming_description: "⚔️ Master of degen plays!".to_string(),
                ranking: "Diamond I".to_string(),
                verified: true,
                subscription_fee: 0.015,
                profit_share: 0.20,
                minimum_copy_amount: 100.0,
                last_active: chrono::Utc::now(),
            },
            copy_amount: 1000.0,
            trades_copied: 15,
            copy_pnl: 125.67,
            copy_pnl_percent: 12.57,
            started_at: chrono::Utc::now() - chrono::Duration::days(30),
            status: "ACTIVE".to_string(),
        }
    ]
}

async fn calculate_copy_trading_performance(
    _app_state: &Arc<AppState>,
    _user_session: &crate::api::server::UserSession,
    following: &[FollowingTrader],
) -> CopyTradingPerformance {
    let total_invested: f64 = following.iter().map(|f| f.copy_amount).sum();
    let total_pnl: f64 = following.iter().map(|f| f.copy_pnl).sum();
    
    CopyTradingPerformance {
        total_invested,
        total_pnl,
        total_pnl_percent: if total_invested > 0.0 { (total_pnl / total_invested) * 100.0 } else { 0.0 },
        active_copies: following.iter().filter(|f| f.status == "ACTIVE").count() as u32,
        total_trades_copied: following.iter().map(|f| f.trades_copied).sum(),
        best_performing_trader: following.iter().max_by(|a, b| a.copy_pnl_percent.partial_cmp(&b.copy_pnl_percent).unwrap()).map(|f| f.trader_profile.display_name.clone()).unwrap_or_default(),
        worst_performing_trader: following.iter().min_by(|a, b| a.copy_pnl_percent.partial_cmp(&b.copy_pnl_percent).unwrap()).map(|f| f.trader_profile.display_name.clone()).unwrap_or_default(),
    }
}

async fn calculate_copy_trading_analytics(
    following: &[FollowingTrader],
    _performance: &CopyTradingPerformance,
) -> CopyTradingAnalytics {
    CopyTradingAnalytics {
        diversification_score: calculate_copy_diversification_score(following),
        risk_distribution: calculate_risk_distribution(following),
        average_trader_win_rate: following.iter().map(|f| f.trader_profile.win_rate).sum::<f64>() / following.len() as f64,
        correlation_to_market: 0.65, // Placeholder
        sharpe_ratio: 1.85, // Placeholder
    }
}

async fn generate_copy_trading_insights(
    _app_state: &Arc<AppState>,
    gaming_theme: &Option<String>,
    performance: &CopyTradingPerformance,
    _analytics: &CopyTradingAnalytics,
) -> Vec<CopyTradingInsight> {
    let mut insights = vec![];
    
    if performance.total_pnl > 0.0 {
        insights.push(CopyTradingInsight {
            insight_type: "PERFORMANCE".to_string(),
            title: "Copy Trading Success".to_string(),
            description: format!("Your copy trading is up {:.2}% overall - great trader selection!", performance.total_pnl_percent),
            gaming_message: match gaming_theme.as_deref() {
                Some("runescape") => "⚔️ Your party is performing excellently! Great leadership choices!".to_string(),
                Some("cod") => "🎖️ Squad performing above expectations! Your tactical decisions are paying off!".to_string(),
                _ => "🏆 Excellent copy trading performance! Keep following those winners!".to_string(),
            },
            actionable: false,
        });
    }
    
    if performance.active_copies == 1 {
        insights.push(CopyTradingInsight {
            insight_type: "DIVERSIFICATION".to_string(),
            title: "Consider Diversification".to_string(),
            description: "You're only following one trader. Consider diversifying across multiple strategies.".to_string(),
            gaming_message: "🎯 Don't put all your eggs in one basket! Consider building a balanced team!".to_string(),
            actionable: true,
        });
    }
    
    insights
}

fn calculate_total_invested(following: &[FollowingTrader]) -> f64 {
    following.iter().map(|f| f.copy_amount).sum()
}

fn calculate_monthly_fees(following: &[FollowingTrader]) -> f64 {
    following.iter().map(|f| f.copy_amount * f.trader_profile.subscription_fee).sum()
}

fn calculate_profit_sharing(performance: &CopyTradingPerformance) -> f64 {
    if performance.total_pnl > 0.0 {
        performance.total_pnl * 0.20 // Assume 20% average profit share
    } else {
        0.0
    }
}

fn calculate_copy_diversification_score(following: &[FollowingTrader]) -> f64 {
    // Simple diversification score based on number of traders and strategy diversity
    let trader_count = following.len() as f64;
    let unique_classes = following.iter().map(|f| &f.trader_profile.combat_class).collect::<std::collections::HashSet<_>>().len() as f64;
    
    (trader_count.min(5.0) / 5.0) * (unique_classes / trader_count.max(1.0)) * 100.0
}

fn calculate_risk_distribution(following: &[FollowingTrader]) -> Vec<RiskDistribution> {
    let mut risk_buckets = std::collections::HashMap::new();
    
    for trader in following {
        let risk_level = if trader.trader_profile.risk_score < 5.0 {
            "LOW"
        } else if trader.trader_profile.risk_score < 7.0 {
            "MEDIUM"
        } else {
            "HIGH"
        };
        
        *risk_buckets.entry(risk_level.to_string()).or_insert(0.0) += trader.copy_amount;
    }
    
    risk_buckets.into_iter().map(|(level, amount)| {
        RiskDistribution { risk_level: level, allocation_amount: amount }
    }).collect()
}

/// **REQUEST/RESPONSE TYPES**

#[derive(Debug, Deserialize)]
pub struct GetTradersParams {
    pub session_token: Option<String>,
    pub gaming_theme: Option<String>,
    pub min_win_rate: Option<f64>,
    pub min_followers: Option<u32>,
    pub combat_class: Option<String>,
    pub risk_level: Option<String>,
    pub sort_by: Option<String>, // "performance", "followers", "win_rate"
}

#[derive(Debug, Deserialize)]
pub struct FollowTraderRequest {
    pub session_token: String,
    pub gaming_theme: Option<String>,
    pub follower_id: String,
    pub trader_id: String,
    pub copy_amount: f64,
    pub copy_settings: CopySettings,
}

#[derive(Debug, Deserialize)]
pub struct CopySettings {
    pub copy_percentage: f64, // What percentage of their trades to copy
    pub max_trade_size: Option<f64>,
    pub stop_loss: Option<f64>,
    pub take_profit: Option<f64>,
}

#[derive(Debug, Deserialize)]
pub struct UnfollowTraderRequest {
    pub session_token: String,
    pub gaming_theme: Option<String>,
    pub follower_id: String,
    pub trader_id: String,
    pub copy_trading_id: String,
}

#[derive(Debug, Deserialize)]
pub struct CopyTradingPerformanceParams {
    pub session_token: Option<String>,
    pub gaming_theme: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct GetTradersResponse {
    pub traders: Vec<TraderProfile>,
    pub leaderboard: Vec<LeaderboardEntry>,
    pub featured_traders: Vec<FeaturedTrader>,
    pub gaming_messages: TradersGamingMessages,
    pub filters_applied: TraderFilters,
    pub total_traders: u32,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Serialize)]
pub struct FollowTraderResponse {
    pub success: bool,
    pub copy_trading_id: Option<String>,
    pub error_message: Option<String>,
    pub gaming_message: String,
    pub trader_profile: Option<TraderProfile>,
    pub estimated_costs: Option<f64>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Serialize)]
pub struct UnfollowTraderResponse {
    pub success: bool,
    pub gaming_message: String,
    pub final_stats: Option<CopyTradingFinalStats>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Serialize)]
pub struct CopyTradingPerformanceResponse {
    pub following: Vec<FollowingTrader>,
    pub performance: CopyTradingPerformance,
    pub analytics: CopyTradingAnalytics,
    pub insights: Vec<CopyTradingInsight>,
    pub total_invested: f64,
    pub monthly_fees: f64,
    pub profit_sharing_paid: f64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Serialize, Default)]
pub struct TraderProfile {
    pub trader_id: String,
    pub display_name: String,
    pub combat_class: String,
    pub win_rate: f64,
    pub total_followers: u32,
    pub total_pnl: f64,
    pub total_pnl_percent: f64,
    pub risk_score: f64,
    pub max_drawdown: f64,
    pub average_trade_size: f64,
    pub total_trades: u32,
    pub gaming_title: String,
    pub gaming_description: String,
    pub ranking: String,
    pub verified: bool,
    pub subscription_fee: f64,
    pub profit_share: f64,
    pub minimum_copy_amount: f64,
    pub last_active: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Serialize)]
pub struct LeaderboardEntry {
    pub rank: u32,
    pub trader_id: String,
    pub display_name: String,
    pub combat_class: String,
    pub total_pnl_percent: f64,
    pub followers: u32,
    pub gaming_title: String,
}

#[derive(Debug, Serialize)]
pub struct FeaturedTrader {
    pub trader_id: String,
    pub display_name: String,
    pub feature_reason: String,
    pub feature_title: String,
    pub feature_description: String,
    pub special_badge: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct TradersGamingMessages {
    pub welcome_message: String,
    pub selection_tip: String,
    pub risk_warning: String,
    pub top_performer_highlight: String,
}

#[derive(Debug, Serialize)]
pub struct TraderFilters {
    pub min_win_rate: Option<f64>,
    pub min_followers: Option<u32>,
    pub combat_class: Option<String>,
    pub risk_level: Option<String>,
}

#[derive(Debug)]
pub struct CopyTradingResult {
    pub copy_trading_id: String,
    pub estimated_monthly_cost: f64,
}

#[derive(Debug, Serialize)]
pub struct CopyTradingFinalStats {
    pub total_duration_days: u32,
    pub trades_copied: u32,
    pub total_pnl: f64,
    pub total_pnl_percent: f64,
    pub total_fees_paid: f64,
    pub best_trade_pnl: f64,
    pub worst_trade_pnl: f64,
}

#[derive(Debug, Serialize)]
pub struct FollowingTrader {
    pub copy_trading_id: String,
    pub trader_profile: TraderProfile,
    pub copy_amount: f64,
    pub trades_copied: u32,
    pub copy_pnl: f64,
    pub copy_pnl_percent: f64,
    pub started_at: chrono::DateTime<chrono::Utc>,
    pub status: String,
}

#[derive(Debug, Serialize)]
pub struct CopyTradingPerformance {
    pub total_invested: f64,
    pub total_pnl: f64,
    pub total_pnl_percent: f64,
    pub active_copies: u32,
    pub total_trades_copied: u32,
    pub best_performing_trader: String,
    pub worst_performing_trader: String,
}

#[derive(Debug, Serialize)]
pub struct CopyTradingAnalytics {
    pub diversification_score: f64,
    pub risk_distribution: Vec<RiskDistribution>,
    pub average_trader_win_rate: f64,
    pub correlation_to_market: f64,
    pub sharpe_ratio: f64,
}

#[derive(Debug, Serialize)]
pub struct RiskDistribution {
    pub risk_level: String,
    pub allocation_amount: f64,
}

#[derive(Debug, Serialize)]
pub struct CopyTradingInsight {
    pub insight_type: String,
    pub title: String,
    pub description: String,
    pub gaming_message: String,
    pub actionable: bool,
}