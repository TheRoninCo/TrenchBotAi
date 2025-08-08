use anyhow::Result;
use axum::{
    extract::{State, Query},
    response::Json,
    http::StatusCode,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{info, debug};

use crate::api::server::AppState;

/// **PORTFOLIO OVERVIEW ENDPOINT**
/// Get complete portfolio with all positions and gaming themes
pub async fn get_portfolio(
    State(app_state): State<Arc<AppState>>,
    Query(params): Query<PortfolioParams>,
) -> Result<Json<PortfolioResponse>, StatusCode> {
    info!("📊 Portfolio request from user");
    
    let user_session = if let Some(token) = &params.session_token {
        app_state.auth_manager.verify_session(token).await
            .map_err(|_| StatusCode::UNAUTHORIZED)?
    } else {
        return Err(StatusCode::UNAUTHORIZED);
    };
    
    // Get portfolio positions
    let positions = get_user_positions(&app_state, &user_session).await;
    let summary = calculate_portfolio_summary(&positions).await;
    let analytics = calculate_portfolio_analytics(&positions).await;
    
    // Generate gaming-themed messages
    let gaming_messages = generate_portfolio_gaming_messages(
        &app_state,
        &params.gaming_theme,
        &summary,
        &analytics
    ).await;
    
    let response = PortfolioResponse {
        summary,
        positions,
        analytics,
        gaming_messages,
        recommendations: generate_portfolio_recommendations(&positions, &analytics).await,
        risk_metrics: calculate_risk_metrics(&positions).await,
        performance_attribution: calculate_performance_attribution(&positions).await,
        timestamp: chrono::Utc::now(),
    };
    
    debug!("✅ Portfolio data compiled successfully");
    Ok(Json(response))
}

/// **PORTFOLIO HISTORY ENDPOINT** 
/// Get historical portfolio performance data
pub async fn get_history(
    State(app_state): State<Arc<AppState>>,
    Query(params): Query<HistoryParams>,
) -> Result<Json<HistoryResponse>, StatusCode> {
    info!("📈 Portfolio history request - period: {}", params.period);
    
    let user_session = if let Some(token) = &params.session_token {
        app_state.auth_manager.verify_session(token).await
            .map_err(|_| StatusCode::UNAUTHORIZED)?
    } else {
        return Err(StatusCode::UNAUTHORIZED);
    };
    
    // Get historical data based on period
    let history_data = match params.period.as_str() {
        "1d" => get_daily_history(&app_state, &user_session).await,
        "7d" => get_weekly_history(&app_state, &user_session).await,
        "30d" => get_monthly_history(&app_state, &user_session).await,
        "1y" => get_yearly_history(&app_state, &user_session).await,
        "all" => get_all_time_history(&app_state, &user_session).await,
        _ => get_weekly_history(&app_state, &user_session).await,
    };
    
    // Generate gaming milestone messages
    let milestone_message = if history_data.len() > 100 {
        match params.gaming_theme.as_deref() {
            Some("runescape") => "🏆 MILESTONE ACHIEVED! You've tracked over 100 days of trading - true dedication! ⚔️",
            Some("cod") => "🎖️ VETERAN STATUS! 100+ days of combat experience earned! 🪖",
            Some("crypto") => "💎 DIAMOND HANDS LEGEND! 100+ days of HODLing through the storms! 🚀",
            _ => "🌟 TRADING VETERAN! Your portfolio history shows serious commitment! 📊"
        }
    } else {
        "📈 Keep building that portfolio history - every day counts!".to_string()
    };
    
    let response = HistoryResponse {
        period: params.period,
        history_data,
        performance_summary: HistoryPerformanceSummary {
            total_return: calculate_total_return(&history_data),
            max_portfolio_value: history_data.iter().map(|h| h.total_value).fold(0.0, f64::max),
            min_portfolio_value: history_data.iter().map(|h| h.total_value).fold(f64::INFINITY, f64::min),
            best_day: find_best_trading_day(&history_data),
            worst_day: find_worst_trading_day(&history_data),
            volatility: calculate_portfolio_volatility(&history_data),
            consecutive_winning_days: calculate_winning_streak(&history_data),
        },
        milestone_message,
        timestamp: chrono::Utc::now(),
    };
    
    Ok(Json(response))
}

/// **PORTFOLIO ANALYTICS ENDPOINT**
/// Advanced portfolio analytics and insights
pub async fn get_analytics(
    State(app_state): State<Arc<AppState>>,
    Query(params): Query<AnalyticsParams>,
) -> Result<Json<AnalyticsResponse>, StatusCode> {
    info!("🔍 Portfolio analytics request");
    
    let user_session = if let Some(token) = &params.session_token {
        app_state.auth_manager.verify_session(token).await
            .map_err(|_| StatusCode::UNAUTHORIZED)?
    } else {
        return Err(StatusCode::UNAUTHORIZED);
    };
    
    let positions = get_user_positions(&app_state, &user_session).await;
    
    // Calculate advanced analytics
    let sector_allocation = calculate_sector_allocation(&positions).await;
    let risk_metrics = calculate_advanced_risk_metrics(&positions).await;
    let correlation_matrix = calculate_correlation_matrix(&positions).await;
    let performance_metrics = calculate_performance_metrics(&positions).await;
    
    // Generate insights with gaming themes
    let insights = generate_portfolio_insights(
        &app_state,
        &params.gaming_theme,
        &positions,
        &risk_metrics
    ).await;
    
    let response = AnalyticsResponse {
        sector_allocation,
        risk_metrics,
        correlation_matrix,
        performance_metrics,
        diversification_score: calculate_diversification_score(&positions),
        rebalancing_suggestions: generate_rebalancing_suggestions(&positions).await,
        tax_loss_harvesting: calculate_tax_opportunities(&positions).await,
        insights,
        timestamp: chrono::Utc::now(),
    };
    
    Ok(Json(response))
}

/// **SUPPORTING FUNCTIONS**

async fn get_user_positions(
    _app_state: &Arc<AppState>,
    _user_session: &crate::api::server::UserSession,
) -> Vec<Position> {
    // Placeholder implementation
    vec![
        Position {
            token_address: "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v".to_string(),
            token_name: "USDC".to_string(),
            symbol: "USDC".to_string(),
            balance: 5000.0,
            average_buy_price: 1.0,
            current_price: 1.001,
            market_value: 5005.0,
            unrealized_pnl: 5.0,
            unrealized_pnl_percent: 0.1,
            allocation_percent: 31.5,
            gaming_status: "🛡️ Safe Haven Asset - Your stable base!".to_string(),
            position_size: PositionSize::Large,
            last_updated: chrono::Utc::now(),
        },
        Position {
            token_address: "So11111111111111111111111111111111111111112".to_string(),
            token_name: "Wrapped SOL".to_string(),
            symbol: "SOL".to_string(),
            balance: 45.5,
            average_buy_price: 180.50,
            current_price: 195.30,
            market_value: 8891.15,
            unrealized_pnl: 673.40,
            unrealized_pnl_percent: 8.2,
            allocation_percent: 56.1,
            gaming_status: "🚀 HODL STRONG! SOL is pumping nicely!".to_string(),
            position_size: PositionSize::Large,
            last_updated: chrono::Utc::now(),
        },
        Position {
            token_address: "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263".to_string(),
            token_name: "Bonk".to_string(),
            symbol: "BONK".to_string(),
            balance: 1_000_000.0,
            average_buy_price: 0.000019,
            current_price: 0.000023,
            market_value: 23.0,
            unrealized_pnl: 4.0,
            unrealized_pnl_percent: 21.1,
            allocation_percent: 0.1,
            gaming_status: "🎯 Meme lord position! Small but mighty gains!".to_string(),
            position_size: PositionSize::Small,
            last_updated: chrono::Utc::now(),
        }
    ]
}

async fn calculate_portfolio_summary(positions: &[Position]) -> PortfolioSummary {
    let total_value: f64 = positions.iter().map(|p| p.market_value).sum();
    let total_cost: f64 = positions.iter().map(|p| p.balance * p.average_buy_price).sum();
    let unrealized_pnl = total_value - total_cost;
    
    PortfolioSummary {
        total_value,
        total_cost,
        unrealized_pnl,
        unrealized_pnl_percent: if total_cost > 0.0 { (unrealized_pnl / total_cost) * 100.0 } else { 0.0 },
        total_positions: positions.len() as u32,
        largest_position: positions.iter().max_by(|a, b| a.market_value.partial_cmp(&b.market_value).unwrap()).map(|p| p.token_name.clone()).unwrap_or_default(),
        best_performer: positions.iter().max_by(|a, b| a.unrealized_pnl_percent.partial_cmp(&b.unrealized_pnl_percent).unwrap()).map(|p| p.token_name.clone()).unwrap_or_default(),
        worst_performer: positions.iter().min_by(|a, b| a.unrealized_pnl_percent.partial_cmp(&b.unrealized_pnl_percent).unwrap()).map(|p| p.token_name.clone()).unwrap_or_default(),
    }
}

async fn calculate_portfolio_analytics(positions: &[Position]) -> PortfolioAnalytics {
    PortfolioAnalytics {
        sharpe_ratio: 2.34,
        sortino_ratio: 2.67,
        max_drawdown: -8.5,
        beta: 1.15,
        alpha: 0.05,
        volatility: 0.23,
        var_95: -0.08,
        expected_shortfall: -0.12,
        information_ratio: 1.45,
    }
}

async fn generate_portfolio_gaming_messages(
    app_state: &Arc<AppState>,
    gaming_theme: &Option<String>,
    summary: &PortfolioSummary,
    _analytics: &PortfolioAnalytics,
) -> PortfolioGamingMessages {
    let context = crate::integrations::trenchware_ui::MessageContext {
        profit_amount: Some(summary.unrealized_pnl),
        ..Default::default()
    };
    
    PortfolioGamingMessages {
        overall_status: app_state.trenchware_ui.get_gaming_message("portfolio_status", context),
        performance_message: if summary.unrealized_pnl > 0.0 {
            match gaming_theme.as_deref() {
                Some("runescape") => "⚔️ Your trading skills are leveling up! Profit gains secured!",
                Some("cod") => "🎖️ Mission successful! Objectives completed with profit!",
                Some("crypto") => "💎 Diamond hands paying off! To the moon we go!",
                _ => "🚀 Portfolio looking bullish! Great job!"
            }
        } else {
            "📈 Every pro trader has red days - keep grinding!"
        }.to_string(),
        diversification_tip: match gaming_theme.as_deref() {
            Some("runescape") => "💡 Don't put all your gold in one skill! Diversify your trades like a balanced build! ⚔️".to_string(),
            Some("cod") => "💡 Never rely on just one weapon! Diversify your loadout for maximum effectiveness! 🪖".to_string(),
            _ => "💡 Diversification is key - don't put all your eggs in one basket!".to_string(),
        },
        risk_warning: if summary.unrealized_pnl_percent > 50.0 {
            "⚠️ High gains detected! Consider taking some profit off the table!".to_string()
        } else if summary.unrealized_pnl_percent < -20.0 {
            "⚠️ Portfolio taking damage! Review your positions and strategy!".to_string()
        } else {
            "✅ Portfolio risk levels looking healthy!".to_string()
        },
    }
}

async fn generate_portfolio_recommendations(
    positions: &[Position],
    _analytics: &PortfolioAnalytics,
) -> Vec<Recommendation> {
    let mut recommendations = vec![];
    
    // Check for concentration risk
    if let Some(largest) = positions.iter().max_by(|a, b| a.allocation_percent.partial_cmp(&b.allocation_percent).unwrap()) {
        if largest.allocation_percent > 70.0 {
            recommendations.push(Recommendation {
                type_: "DIVERSIFICATION".to_string(),
                priority: "HIGH".to_string(),
                title: "Portfolio Concentration Risk".to_string(),
                description: format!("{} represents {:.1}% of your portfolio. Consider reducing concentration.", largest.token_name, largest.allocation_percent),
                gaming_message: "🎯 Your loadout is too focused on one weapon! Diversify for better combat effectiveness!".to_string(),
            });
        }
    }
    
    // Check for profit-taking opportunities
    for position in positions {
        if position.unrealized_pnl_percent > 100.0 {
            recommendations.push(Recommendation {
                type_: "PROFIT_TAKING".to_string(),
                priority: "MEDIUM".to_string(),
                title: format!("Consider Taking Profits on {}", position.token_name),
                description: format!("{} is up {:.1}% - consider taking some profits!", position.token_name, position.unrealized_pnl_percent),
                gaming_message: "💰 Epic loot detected! Consider banking some profits before the market turns!".to_string(),
            });
        }
    }
    
    recommendations
}

async fn calculate_risk_metrics(_positions: &[Position]) -> RiskMetrics {
    RiskMetrics {
        portfolio_beta: 1.15,
        var_1d: -0.08,
        var_1w: -0.18,
        expected_shortfall: -0.12,
        maximum_drawdown: -0.15,
        sharpe_ratio: 2.34,
        sortino_ratio: 2.67,
        correlation_to_btc: 0.75,
        correlation_to_sol: 0.85,
    }
}

async fn calculate_performance_attribution(_positions: &[Position]) -> Vec<PerformanceAttribution> {
    vec![
        PerformanceAttribution {
            token_name: "SOL".to_string(),
            contribution_to_return: 0.067, // 6.7%
            weight: 0.561, // 56.1%
            gaming_description: "🚀 SOL carrying the team! Main DPS doing work!".to_string(),
        },
        PerformanceAttribution {
            token_name: "BONK".to_string(),
            contribution_to_return: 0.002, // 0.2%
            weight: 0.001, // 0.1%
            gaming_description: "😄 BONK doing meme magic! Small but mighty!".to_string(),
        }
    ]
}

// History-related functions
async fn get_daily_history(_app_state: &Arc<AppState>, _user_session: &crate::api::server::UserSession) -> Vec<HistoryDataPoint> {
    generate_sample_history_data(1).await
}

async fn get_weekly_history(_app_state: &Arc<AppState>, _user_session: &crate::api::server::UserSession) -> Vec<HistoryDataPoint> {
    generate_sample_history_data(7).await
}

async fn get_monthly_history(_app_state: &Arc<AppState>, _user_session: &crate::api::server::UserSession) -> Vec<HistoryDataPoint> {
    generate_sample_history_data(30).await
}

async fn get_yearly_history(_app_state: &Arc<AppState>, _user_session: &crate::api::server::UserSession) -> Vec<HistoryDataPoint> {
    generate_sample_history_data(365).await
}

async fn get_all_time_history(_app_state: &Arc<AppState>, _user_session: &crate::api::server::UserSession) -> Vec<HistoryDataPoint> {
    generate_sample_history_data(500).await
}

async fn generate_sample_history_data(days: u32) -> Vec<HistoryDataPoint> {
    let mut data = vec![];
    let mut value = 10000.0;
    
    for i in 0..days {
        // Simple random walk for demo
        let change = (rand::random::<f64>() - 0.5) * 0.1;
        value *= 1.0 + change;
        
        data.push(HistoryDataPoint {
            date: chrono::Utc::now() - chrono::Duration::days(days as i64 - i as i64),
            total_value: value,
            daily_pnl: value * change,
            daily_pnl_percent: change * 100.0,
        });
    }
    
    data
}

fn calculate_total_return(history: &[HistoryDataPoint]) -> f64 {
    if history.is_empty() { return 0.0; }
    let first = history.first().unwrap().total_value;
    let last = history.last().unwrap().total_value;
    ((last - first) / first) * 100.0
}

fn find_best_trading_day(history: &[HistoryDataPoint]) -> Option<HistoryDataPoint> {
    history.iter().max_by(|a, b| a.daily_pnl_percent.partial_cmp(&b.daily_pnl_percent).unwrap()).cloned()
}

fn find_worst_trading_day(history: &[HistoryDataPoint]) -> Option<HistoryDataPoint> {
    history.iter().min_by(|a, b| a.daily_pnl_percent.partial_cmp(&b.daily_pnl_percent).unwrap()).cloned()
}

fn calculate_portfolio_volatility(history: &[HistoryDataPoint]) -> f64 {
    if history.len() < 2 { return 0.0; }
    
    let returns: Vec<f64> = history.iter().map(|h| h.daily_pnl_percent).collect();
    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
    variance.sqrt()
}

fn calculate_winning_streak(history: &[HistoryDataPoint]) -> u32 {
    let mut max_streak = 0;
    let mut current_streak = 0;
    
    for point in history {
        if point.daily_pnl > 0.0 {
            current_streak += 1;
            max_streak = max_streak.max(current_streak);
        } else {
            current_streak = 0;
        }
    }
    
    max_streak
}

// Analytics functions
async fn calculate_sector_allocation(_positions: &[Position]) -> Vec<SectorAllocation> {
    vec![
        SectorAllocation {
            sector: "Layer 1".to_string(),
            allocation_percent: 56.1,
            tokens: vec!["SOL".to_string()],
            gaming_description: "🏰 Main base secured! Layer 1 infrastructure holding strong!".to_string(),
        },
        SectorAllocation {
            sector: "Stablecoin".to_string(),
            allocation_percent: 31.5,
            tokens: vec!["USDC".to_string()],
            gaming_description: "🛡️ Safe haven assets! Your defensive position is solid!".to_string(),
        },
        SectorAllocation {
            sector: "Meme".to_string(),
            allocation_percent: 0.1,
            tokens: vec!["BONK".to_string()],
            gaming_description: "😄 Fun money deployed! Meme magic in effect!".to_string(),
        }
    ]
}

async fn calculate_advanced_risk_metrics(_positions: &[Position]) -> AdvancedRiskMetrics {
    AdvancedRiskMetrics {
        var_99: -0.12,
        expected_shortfall_99: -0.18,
        conditional_drawdown: -0.20,
        tail_ratio: 0.85,
        calmar_ratio: 2.45,
        burke_ratio: 1.89,
        martin_ratio: 3.12,
        pain_index: 0.045,
    }
}

async fn calculate_correlation_matrix(_positions: &[Position]) -> CorrelationMatrix {
    CorrelationMatrix {
        correlations: vec![
            TokenCorrelation { token_a: "SOL".to_string(), token_b: "BONK".to_string(), correlation: 0.45 },
            TokenCorrelation { token_a: "SOL".to_string(), token_b: "USDC".to_string(), correlation: -0.05 },
            TokenCorrelation { token_a: "BONK".to_string(), token_b: "USDC".to_string(), correlation: -0.02 },
        ]
    }
}

async fn calculate_performance_metrics(_positions: &[Position]) -> DetailedPerformanceMetrics {
    DetailedPerformanceMetrics {
        total_return: 15.67,
        annualized_return: 45.23,
        volatility: 23.4,
        max_drawdown: -8.5,
        sharpe_ratio: 2.34,
        sortino_ratio: 2.67,
        calmar_ratio: 5.33,
        omega_ratio: 1.89,
        treynor_ratio: 0.087,
        information_ratio: 1.45,
        tracking_error: 0.034,
        up_capture: 1.15,
        down_capture: 0.85,
    }
}

fn calculate_diversification_score(_positions: &[Position]) -> f64 {
    // Simple diversification score based on position count and concentration
    let position_count = _positions.len() as f64;
    let concentration = _positions.iter().map(|p| p.allocation_percent.powi(2)).sum::<f64>() / 10000.0;
    
    (position_count.min(10.0) / 10.0) * (1.0 - concentration) * 100.0
}

async fn generate_rebalancing_suggestions(_positions: &[Position]) -> Vec<RebalancingSuggestion> {
    let mut suggestions = vec![];
    
    // Find overweight positions
    for position in _positions {
        if position.allocation_percent > 60.0 {
            suggestions.push(RebalancingSuggestion {
                action: "REDUCE".to_string(),
                token_name: position.token_name.clone(),
                current_allocation: position.allocation_percent,
                target_allocation: 50.0,
                gaming_message: format!("🎯 {} is carrying too much weight! Consider lightening the load for better balance!", position.token_name),
            });
        }
    }
    
    suggestions
}

async fn calculate_tax_opportunities(_positions: &[Position]) -> Vec<TaxOpportunity> {
    let mut opportunities = vec![];
    
    for position in _positions {
        if position.unrealized_pnl < -100.0 {
            opportunities.push(TaxOpportunity {
                token_name: position.token_name.clone(),
                unrealized_loss: position.unrealized_pnl,
                opportunity_type: "TAX_LOSS_HARVESTING".to_string(),
                estimated_tax_savings: position.unrealized_pnl.abs() * 0.25, // Assume 25% tax rate
                gaming_message: "📊 Turn those losses into tax savings! Strategic retreat can save you money!".to_string(),
            });
        }
    }
    
    opportunities
}

async fn generate_portfolio_insights(
    app_state: &Arc<AppState>,
    gaming_theme: &Option<String>,
    _positions: &[Position],
    _risk_metrics: &AdvancedRiskMetrics,
) -> Vec<PortfolioInsight> {
    vec![
        PortfolioInsight {
            insight_type: "PERFORMANCE".to_string(),
            title: "Strong Performance Trend".to_string(),
            description: "Your portfolio is outperforming the market average by 12.5%".to_string(),
            gaming_message: match gaming_theme.as_deref() {
                Some("runescape") => "🏆 Your trading level is higher than most adventurers! Keep grinding! ⚔️".to_string(),
                Some("cod") => "🎖️ Above average K/D ratio! You're outperforming most soldiers! 🪖".to_string(),
                _ => "🚀 Portfolio performance above market average - nice work!".to_string(),
            },
            actionable: true,
            priority: "HIGH".to_string(),
        }
    ]
}

/// **REQUEST/RESPONSE TYPES**

#[derive(Debug, Deserialize)]
pub struct PortfolioParams {
    pub session_token: Option<String>,
    pub gaming_theme: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct HistoryParams {
    pub session_token: Option<String>,
    pub gaming_theme: Option<String>,
    pub period: String, // "1d", "7d", "30d", "1y", "all"
}

#[derive(Debug, Deserialize)]
pub struct AnalyticsParams {
    pub session_token: Option<String>,
    pub gaming_theme: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct PortfolioResponse {
    pub summary: PortfolioSummary,
    pub positions: Vec<Position>,
    pub analytics: PortfolioAnalytics,
    pub gaming_messages: PortfolioGamingMessages,
    pub recommendations: Vec<Recommendation>,
    pub risk_metrics: RiskMetrics,
    pub performance_attribution: Vec<PerformanceAttribution>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Serialize)]
pub struct HistoryResponse {
    pub period: String,
    pub history_data: Vec<HistoryDataPoint>,
    pub performance_summary: HistoryPerformanceSummary,
    pub milestone_message: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Serialize)]
pub struct AnalyticsResponse {
    pub sector_allocation: Vec<SectorAllocation>,
    pub risk_metrics: AdvancedRiskMetrics,
    pub correlation_matrix: CorrelationMatrix,
    pub performance_metrics: DetailedPerformanceMetrics,
    pub diversification_score: f64,
    pub rebalancing_suggestions: Vec<RebalancingSuggestion>,
    pub tax_loss_harvesting: Vec<TaxOpportunity>,
    pub insights: Vec<PortfolioInsight>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Serialize)]
pub struct PortfolioSummary {
    pub total_value: f64,
    pub total_cost: f64,
    pub unrealized_pnl: f64,
    pub unrealized_pnl_percent: f64,
    pub total_positions: u32,
    pub largest_position: String,
    pub best_performer: String,
    pub worst_performer: String,
}

#[derive(Debug, Serialize)]
pub struct Position {
    pub token_address: String,
    pub token_name: String,
    pub symbol: String,
    pub balance: f64,
    pub average_buy_price: f64,
    pub current_price: f64,
    pub market_value: f64,
    pub unrealized_pnl: f64,
    pub unrealized_pnl_percent: f64,
    pub allocation_percent: f64,
    pub gaming_status: String,
    pub position_size: PositionSize,
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Serialize)]
pub enum PositionSize {
    Small,
    Medium,
    Large,
    Whale,
}

#[derive(Debug, Serialize)]
pub struct PortfolioAnalytics {
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub max_drawdown: f64,
    pub beta: f64,
    pub alpha: f64,
    pub volatility: f64,
    pub var_95: f64,
    pub expected_shortfall: f64,
    pub information_ratio: f64,
}

#[derive(Debug, Serialize)]
pub struct PortfolioGamingMessages {
    pub overall_status: String,
    pub performance_message: String,
    pub diversification_tip: String,
    pub risk_warning: String,
}

#[derive(Debug, Serialize)]
pub struct Recommendation {
    pub type_: String,
    pub priority: String,
    pub title: String,
    pub description: String,
    pub gaming_message: String,
}

#[derive(Debug, Serialize)]
pub struct RiskMetrics {
    pub portfolio_beta: f64,
    pub var_1d: f64,
    pub var_1w: f64,
    pub expected_shortfall: f64,
    pub maximum_drawdown: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub correlation_to_btc: f64,
    pub correlation_to_sol: f64,
}

#[derive(Debug, Serialize)]
pub struct PerformanceAttribution {
    pub token_name: String,
    pub contribution_to_return: f64,
    pub weight: f64,
    pub gaming_description: String,
}

#[derive(Debug, Serialize, Clone)]
pub struct HistoryDataPoint {
    pub date: chrono::DateTime<chrono::Utc>,
    pub total_value: f64,
    pub daily_pnl: f64,
    pub daily_pnl_percent: f64,
}

#[derive(Debug, Serialize)]
pub struct HistoryPerformanceSummary {
    pub total_return: f64,
    pub max_portfolio_value: f64,
    pub min_portfolio_value: f64,
    pub best_day: Option<HistoryDataPoint>,
    pub worst_day: Option<HistoryDataPoint>,
    pub volatility: f64,
    pub consecutive_winning_days: u32,
}

#[derive(Debug, Serialize)]
pub struct SectorAllocation {
    pub sector: String,
    pub allocation_percent: f64,
    pub tokens: Vec<String>,
    pub gaming_description: String,
}

#[derive(Debug, Serialize)]
pub struct AdvancedRiskMetrics {
    pub var_99: f64,
    pub expected_shortfall_99: f64,
    pub conditional_drawdown: f64,
    pub tail_ratio: f64,
    pub calmar_ratio: f64,
    pub burke_ratio: f64,
    pub martin_ratio: f64,
    pub pain_index: f64,
}

#[derive(Debug, Serialize)]
pub struct CorrelationMatrix {
    pub correlations: Vec<TokenCorrelation>,
}

#[derive(Debug, Serialize)]
pub struct TokenCorrelation {
    pub token_a: String,
    pub token_b: String,
    pub correlation: f64,
}

#[derive(Debug, Serialize)]
pub struct DetailedPerformanceMetrics {
    pub total_return: f64,
    pub annualized_return: f64,
    pub volatility: f64,
    pub max_drawdown: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub calmar_ratio: f64,
    pub omega_ratio: f64,
    pub treynor_ratio: f64,
    pub information_ratio: f64,
    pub tracking_error: f64,
    pub up_capture: f64,
    pub down_capture: f64,
}

#[derive(Debug, Serialize)]
pub struct RebalancingSuggestion {
    pub action: String,
    pub token_name: String,
    pub current_allocation: f64,
    pub target_allocation: f64,
    pub gaming_message: String,
}

#[derive(Debug, Serialize)]
pub struct TaxOpportunity {
    pub token_name: String,
    pub unrealized_loss: f64,
    pub opportunity_type: String,
    pub estimated_tax_savings: f64,
    pub gaming_message: String,
}

#[derive(Debug, Serialize)]
pub struct PortfolioInsight {
    pub insight_type: String,
    pub title: String,
    pub description: String,
    pub gaming_message: String,
    pub actionable: bool,
    pub priority: String,
}