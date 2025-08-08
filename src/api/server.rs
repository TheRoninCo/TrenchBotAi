use anyhow::Result;
use axum::{
    Router,
    routing::{get, post},
    middleware::{self, Next},
    extract::{Request, State, WebSocketUpgrade},
    response::Response,
    http::{StatusCode, HeaderValue, Method},
};
use tower::ServiceBuilder;
use tower_http::{
    cors::{Any, CorsLayer},
    trace::TraceLayer,
    compression::CompressionLayer,
    limit::RequestBodyLimitLayer,
};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, error};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

use crate::api::{routes, middleware as api_middleware, websocket, auth, handlers};
use crate::integrations::{
    phantom_wallet::PhantomWalletIntegration,
    telegram_integration::TelegramIntegration,
    notification_system::NotificationSystem,
    ranking_system::RankingSystem,
    trenchware_ui::TrenchwareUI,
    dynamic_response_engine::DynamicResponseEngine,
};
use crate::social::copy_trading::CopyTradingSystem;

/// **TRENCHBOT WEB SERVER**  
/// Complete REST API + WebSocket server with gaming themes and real-time features
#[derive(Debug)]
pub struct TrenchBotServer {
    // **SERVER CONFIG**
    pub config: ServerConfig,
    pub app_state: Arc<AppState>,
    
    // **INTEGRATIONS**
    pub phantom_wallet: Arc<PhantomWalletIntegration>,
    pub telegram: Arc<TelegramIntegration>,
    pub notifications: Arc<NotificationSystem>,
    pub ranking_system: Arc<RankingSystem>,
    pub copy_trading: Arc<CopyTradingSystem>,
    
    // **UI & RESPONSE**
    pub trenchware_ui: Arc<TrenchwareUI>,
    pub response_engine: Arc<DynamicResponseEngine>,
    
    // **REAL-TIME**
    pub websocket_manager: Arc<websocket::WebSocketManager>,
    pub active_sessions: Arc<RwLock<HashMap<String, UserSession>>>,
    
    // **METRICS & MONITORING**  
    pub metrics_collector: Arc<MetricsCollector>,
    pub performance_tracker: Arc<PerformanceTracker>,
}

/// **SERVER CONFIGURATION**
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub cors_origins: Vec<String>,
    pub max_connections: usize,
    pub request_timeout: Duration,
    pub max_request_size: usize,
    pub rate_limits: RateLimitConfig,
    pub ssl_config: Option<SSLConfig>,
    pub environment: Environment,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    pub requests_per_minute: u32,
    pub burst_size: u32,
    pub websocket_messages_per_minute: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SSLConfig {
    pub cert_path: String,
    pub key_path: String,
    pub redirect_http: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Environment {
    Development,
    Staging,
    Production,
}

/// **APPLICATION STATE**
/// Shared state across all routes and WebSocket connections
#[derive(Debug)]
pub struct AppState {
    // **CORE SYSTEMS**
    pub phantom_wallet: Arc<PhantomWalletIntegration>,
    pub telegram: Arc<TelegramIntegration>,
    pub notifications: Arc<NotificationSystem>,
    pub ranking_system: Arc<RankingSystem>,
    pub copy_trading: Arc<CopyTradingSystem>,
    pub trenchware_ui: Arc<TrenchwareUI>,
    pub response_engine: Arc<DynamicResponseEngine>,
    
    // **SESSION & AUTH**
    pub auth_manager: Arc<auth::AuthManager>,
    pub active_sessions: Arc<RwLock<HashMap<String, UserSession>>>,
    pub user_connections: Arc<RwLock<HashMap<String, Vec<String>>>>, // user_id -> websocket_ids
    
    // **REAL-TIME DATA**
    pub live_prices: Arc<RwLock<HashMap<String, TokenPrice>>>,
    pub trading_signals: Arc<RwLock<Vec<TradingSignal>>>,
    pub whale_alerts: Arc<RwLock<Vec<WhaleAlert>>>,
    
    // **METRICS**
    pub metrics: Arc<MetricsCollector>,
    pub performance: Arc<PerformanceTracker>,
    
    // **CONFIGURATION**
    pub config: ServerConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserSession {
    pub user_id: String,
    pub wallet_address: Option<String>,
    pub session_token: String,
    pub rank_theme: Option<crate::integrations::ranking_system::RankTheme>,
    pub permissions: Vec<Permission>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub last_active: chrono::DateTime<chrono::Utc>,
    pub websocket_connections: Vec<String>,
    pub subscription_tier: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Permission {
    ViewPortfolio,
    ExecuteTrades,
    ManageSettings,
    AccessPremium,
    CopyTrade,
    AdminPanel,
}

impl TrenchBotServer {
    pub async fn new(config: ServerConfig) -> Result<Self> {
        info!("🚀 INITIALIZING TRENCHBOT WEB SERVER");
        info!("🌐 Host: {}:{}", config.host, config.port);
        info!("🎮 Gaming-themed REST API + WebSocket server");
        info!("⚡ Real-time trading data and notifications");
        info!("🔐 JWT authentication with Phantom wallet integration");
        
        // Initialize all integrations
        let phantom_wallet = Arc::new(PhantomWalletIntegration::new().await?);
        let mut telegram = TelegramIntegration::new("dummy_token".to_string()).await?;
        
        // Link integrations
        telegram.link_phantom_integration(phantom_wallet.clone()).await?;
        let telegram = Arc::new(telegram);
        
        let notifications = Arc::new(NotificationSystem::new().await?);
        let ranking_system = Arc::new(RankingSystem::new());
        let copy_trading = Arc::new(CopyTradingSystem::new().await?);
        let trenchware_ui = Arc::new(TrenchwareUI::new());
        let response_engine = Arc::new(DynamicResponseEngine::new().await?);
        
        // Initialize server components
        let websocket_manager = Arc::new(websocket::WebSocketManager::new().await?);
        let active_sessions = Arc::new(RwLock::new(HashMap::new()));
        let metrics_collector = Arc::new(MetricsCollector::new());
        let performance_tracker = Arc::new(PerformanceTracker::new());
        
        // Create app state
        let app_state = Arc::new(AppState {
            phantom_wallet: phantom_wallet.clone(),
            telegram: telegram.clone(),
            notifications: notifications.clone(),
            ranking_system: ranking_system.clone(),
            copy_trading: copy_trading.clone(),
            trenchware_ui: trenchware_ui.clone(),
            response_engine: response_engine.clone(),
            auth_manager: Arc::new(auth::AuthManager::new().await?),
            active_sessions: active_sessions.clone(),
            user_connections: Arc::new(RwLock::new(HashMap::new())),
            live_prices: Arc::new(RwLock::new(HashMap::new())),
            trading_signals: Arc::new(RwLock::new(Vec::new())),
            whale_alerts: Arc::new(RwLock::new(Vec::new())),
            metrics: metrics_collector.clone(),
            performance: performance_tracker.clone(),
            config: config.clone(),
        });
        
        let server = Self {
            config,
            app_state,
            phantom_wallet,
            telegram,
            notifications,
            ranking_system,
            copy_trading,
            trenchware_ui,
            response_engine,
            websocket_manager,
            active_sessions,
            metrics_collector,
            performance_tracker,
        };
        
        info!("✅ TrenchBot Web Server initialized!");
        Ok(server)
    }
    
    /// **BUILD ROUTER**
    /// Creates the complete axum router with all routes and middleware
    pub fn build_router(&self) -> Router {
        info!("🔧 Building TrenchBot API router");
        
        // Create CORS layer
        let cors = CorsLayer::new()
            .allow_origin(Any) // In production, use specific origins
            .allow_methods([Method::GET, Method::POST, Method::PUT, Method::DELETE, Method::OPTIONS])
            .allow_headers(Any)
            .max_age(Duration::from_secs(3600));
        
        // Build the main router
        let app = Router::new()
            // **HEALTH & STATUS**
            .route("/health", get(handlers::health_check))
            .route("/status", get(handlers::system_status))
            
            // **AUTHENTICATION ROUTES**
            .route("/api/v1/auth/connect", post(handlers::auth::connect_wallet))
            .route("/api/v1/auth/verify", post(handlers::auth::verify_session))
            .route("/api/v1/auth/refresh", post(handlers::auth::refresh_token))
            .route("/api/v1/auth/disconnect", post(handlers::auth::disconnect_wallet))
            
            // **DASHBOARD ROUTES**  
            .route("/api/v1/dashboard", get(handlers::dashboard::get_dashboard))
            .route("/api/v1/dashboard/overview", get(handlers::dashboard::get_overview))
            .route("/api/v1/dashboard/performance", get(handlers::dashboard::get_performance))
            
            // **PORTFOLIO ROUTES**
            .route("/api/v1/portfolio", get(handlers::portfolio::get_portfolio))
            .route("/api/v1/portfolio/history", get(handlers::portfolio::get_history))
            .route("/api/v1/portfolio/analytics", get(handlers::portfolio::get_analytics))
            
            // **TRADING ROUTES**
            .route("/api/v1/trades", get(handlers::trading::get_trades))
            .route("/api/v1/trades/execute", post(handlers::trading::execute_trade))
            .route("/api/v1/trades/history", get(handlers::trading::get_trade_history))
            
            // **COPY TRADING ROUTES**
            .route("/api/v1/copy-trading/traders", get(handlers::copy_trading::get_traders))
            .route("/api/v1/copy-trading/follow", post(handlers::copy_trading::follow_trader))
            .route("/api/v1/copy-trading/unfollow", post(handlers::copy_trading::unfollow_trader))
            .route("/api/v1/copy-trading/performance", get(handlers::copy_trading::get_performance))
            
            // **SOCIAL ROUTES**
            .route("/api/v1/social/feed", get(handlers::social::get_social_feed))
            .route("/api/v1/social/leaderboard", get(handlers::social::get_leaderboard))
            .route("/api/v1/social/profile", get(handlers::social::get_profile))
            
            // **RANKING & ACHIEVEMENTS**
            .route("/api/v1/rankings/my-rank", get(handlers::rankings::get_my_rank))
            .route("/api/v1/rankings/themes", get(handlers::rankings::get_themes))
            .route("/api/v1/rankings/change-theme", post(handlers::rankings::change_theme))
            .route("/api/v1/achievements", get(handlers::achievements::get_achievements))
            .route("/api/v1/achievements/claim", post(handlers::achievements::claim_achievement))
            
            // **NOTIFICATIONS**
            .route("/api/v1/notifications", get(handlers::notifications::get_notifications))
            .route("/api/v1/notifications/settings", get(handlers::notifications::get_settings))
            .route("/api/v1/notifications/settings", post(handlers::notifications::update_settings))
            
            // **WEBSOCKET ROUTES**
            .route("/ws", get(websocket::websocket_handler))
            .route("/ws/portfolio", get(websocket::portfolio_websocket))
            .route("/ws/trading", get(websocket::trading_websocket))
            .route("/ws/social", get(websocket::social_websocket))
            
            // **METRICS & MONITORING**
            .route("/metrics", get(handlers::metrics::prometheus_metrics))
            .route("/api/v1/metrics/performance", get(handlers::metrics::get_performance_metrics))
            .route("/api/v1/metrics/usage", get(handlers::metrics::get_usage_metrics))
            
            // **ADMIN ROUTES** (protected)
            .route("/api/v1/admin/users", get(handlers::admin::get_users))
            .route("/api/v1/admin/system", get(handlers::admin::get_system_info))
            .route("/api/v1/admin/metrics", get(handlers::admin::get_admin_metrics))
            
            // **STATIC FILES** (for frontend)
            .route("/", get(handlers::static_files::index))
            .nest_service("/static", handlers::static_files::static_handler())
            
            // **STATE & MIDDLEWARE**
            .with_state(self.app_state.clone())
            .layer(
                ServiceBuilder::new()
                    .layer(TraceLayer::new_for_http())
                    .layer(cors)
                    .layer(CompressionLayer::new())
                    .layer(RequestBodyLimitLayer::new(self.config.max_request_size))
                    .layer(middleware::from_fn_with_state(
                        self.app_state.clone(),
                        api_middleware::auth_middleware
                    ))
                    .layer(middleware::from_fn_with_state(
                        self.app_state.clone(),
                        api_middleware::rate_limit_middleware
                    ))
                    .layer(middleware::from_fn_with_state(
                        self.app_state.clone(),
                        api_middleware::metrics_middleware
                    ))
                    .layer(middleware::from_fn_with_state(
                        self.app_state.clone(),
                        api_middleware::gaming_theme_middleware
                    ))
            );
        
        info!("✅ TrenchBot API router built with {} routes", 30); // Rough count
        app
    }
    
    /// **START SERVER**
    /// Starts the axum server with graceful shutdown
    pub async fn start(&self) -> Result<()> {
        let addr = format!("{}:{}", self.config.host, self.config.port);
        info!("🚀 Starting TrenchBot Server on {}", addr);
        info!("🎮 Gaming-themed API ready for battle!");
        info!("⚡ WebSocket connections: /ws");
        info!("📊 Metrics endpoint: /metrics");
        info!("🔐 Auth endpoints: /api/v1/auth/*");
        info!("📈 Dashboard API: /api/v1/dashboard/*");
        info!("🤝 Copy trading: /api/v1/copy-trading/*");
        
        // Start background tasks
        self.start_background_tasks().await?;
        
        // Build and start the server
        let app = self.build_router();
        let listener = tokio::net::TcpListener::bind(&addr).await?;
        
        info!("🎯 TrenchBot is LOCKED AND LOADED on {}", addr);
        info!("💀 Ready to eliminate some profits!");
        
        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal())
            .await?;
        
        Ok(())
    }
    
    /// **START BACKGROUND TASKS**
    /// Initialize background processes for real-time features
    async fn start_background_tasks(&self) -> Result<()> {
        info!("🔄 Starting background tasks");
        
        // Start WebSocket manager
        let websocket_manager = self.websocket_manager.clone();
        tokio::spawn(async move {
            websocket_manager.start_connection_manager().await;
        });
        
        // Start price feed updater
        let app_state = self.app_state.clone();
        tokio::spawn(async move {
            loop {
                if let Err(e) = update_live_prices(&app_state).await {
                    warn!("Failed to update live prices: {}", e);
                }
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
        });
        
        // Start metrics collector
        let metrics = self.metrics_collector.clone();
        tokio::spawn(async move {
            metrics.start_collection_loop().await;
        });
        
        // Start session cleanup
        let active_sessions = self.active_sessions.clone();
        tokio::spawn(async move {
            loop {
                cleanup_expired_sessions(&active_sessions).await;
                tokio::time::sleep(Duration::from_minutes(5)).await;
            }
        });
        
        info!("✅ All background tasks started");
        Ok(())
    }
}

/// **SUPPORTING TYPES**
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenPrice {
    pub token_address: String,
    pub price_usd: f64,
    pub price_change_24h: f64,
    pub volume_24h: f64,
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingSignal {
    pub signal_id: String,
    pub token_address: String,
    pub signal_type: SignalType,
    pub confidence: f64,
    pub expected_profit: f64,
    pub risk_level: RiskLevel,
    pub expires_at: chrono::DateTime<chrono::Utc>,
    pub gaming_description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SignalType {
    Buy,
    Sell,
    Hold,
    Avoid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Extreme,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhaleAlert {
    pub alert_id: String,
    pub wallet_address: String,
    pub amount: f64,
    pub token_address: String,
    pub action: String, // "buy", "sell", "transfer"
    pub gaming_message: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

// **IMPLEMENTATION STUBS**
#[derive(Debug)] pub struct MetricsCollector;
#[derive(Debug)] pub struct PerformanceTracker;

impl MetricsCollector {
    fn new() -> Self { Self }
    async fn start_collection_loop(&self) {
        info!("📊 Metrics collection loop started");
    }
}

impl PerformanceTracker {
    fn new() -> Self { Self }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 3000,
            cors_origins: vec!["http://localhost:3000".to_string()],
            max_connections: 10000,
            request_timeout: Duration::from_secs(30),
            max_request_size: 16 * 1024 * 1024, // 16MB
            rate_limits: RateLimitConfig {
                requests_per_minute: 100,
                burst_size: 10,
                websocket_messages_per_minute: 1000,
            },
            ssl_config: None,
            environment: Environment::Development,
        }
    }
}

// **HELPER FUNCTIONS**
async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            info!("🛑 Received Ctrl+C signal, shutting down gracefully");
        },
        _ = terminate => {
            info!("🛑 Received terminate signal, shutting down gracefully");
        },
    }
}

async fn update_live_prices(app_state: &AppState) -> Result<()> {
    // Update live price feeds
    // In real implementation, this would fetch from exchanges
    Ok(())
}

async fn cleanup_expired_sessions(sessions: &Arc<RwLock<HashMap<String, UserSession>>>) {
    let mut sessions = sessions.write().await;
    let now = chrono::Utc::now();
    
    sessions.retain(|_, session| {
        now.signed_duration_since(session.last_active) < chrono::Duration::hours(24)
    });
}