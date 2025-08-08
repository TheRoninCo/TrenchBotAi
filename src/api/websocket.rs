use anyhow::Result;
use axum::{
    extract::{WebSocketUpgrade, State, Query},
    response::Response,
    http::StatusCode,
};
use axum::extract::ws::{WebSocket, Message};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc, broadcast};
use tracing::{info, warn, error, debug};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use futures::{sink::SinkExt, stream::StreamExt};

use crate::api::server::AppState;
use crate::integrations::trenchware_ui::TrenchwareUI;

/// **WEBSOCKET MANAGER**
/// Handles all real-time WebSocket connections and message broadcasting
#[derive(Debug)]
pub struct WebSocketManager {
    // **CONNECTION MANAGEMENT**
    pub connections: Arc<RwLock<HashMap<String, ActiveConnection>>>,
    pub user_connections: Arc<RwLock<HashMap<String, Vec<String>>>>, // user_id -> connection_ids
    pub room_connections: Arc<RwLock<HashMap<String, Vec<String>>>>, // room_name -> connection_ids
    
    // **BROADCAST CHANNELS**
    pub price_broadcaster: broadcast::Sender<PriceUpdate>,
    pub trade_broadcaster: broadcast::Sender<TradeUpdate>,
    pub notification_broadcaster: broadcast::Sender<NotificationUpdate>,
    pub social_broadcaster: broadcast::Sender<SocialUpdate>,
    pub whale_broadcaster: broadcast::Sender<WhaleUpdate>,
    
    // **MESSAGE PROCESSING**
    pub message_processor: Arc<MessageProcessor>,
    pub rate_limiter: Arc<WebSocketRateLimiter>,
    
    // **GAMING FEATURES**
    pub trenchware_ui: Arc<TrenchwareUI>,
    pub live_leaderboard: Arc<RwLock<LiveLeaderboard>>,
}

#[derive(Debug, Clone)]
pub struct ActiveConnection {
    pub connection_id: String,
    pub user_id: Option<String>,
    pub wallet_address: Option<String>,
    pub subscriptions: Vec<SubscriptionType>,
    pub last_ping: std::time::Instant,
    pub message_count: u32,
    pub gaming_theme: Option<String>,
    pub rank_theme: Option<crate::integrations::ranking_system::RankTheme>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SubscriptionType {
    PriceUpdates(Vec<String>), // token addresses
    PortfolioUpdates,
    TradingSignals,
    Notifications,
    SocialFeed,
    WhaleAlerts,
    Leaderboard,
    CopyTrading,
}

impl WebSocketManager {
    pub async fn new() -> Result<Self> {
        info!("🔄 Initializing WebSocket Manager");
        
        let (price_tx, _) = broadcast::channel(1000);
        let (trade_tx, _) = broadcast::channel(1000);
        let (notification_tx, _) = broadcast::channel(1000);
        let (social_tx, _) = broadcast::channel(1000);
        let (whale_tx, _) = broadcast::channel(1000);
        
        let manager = Self {
            connections: Arc::new(RwLock::new(HashMap::new())),
            user_connections: Arc::new(RwLock::new(HashMap::new())),
            room_connections: Arc::new(RwLock::new(HashMap::new())),
            price_broadcaster: price_tx,
            trade_broadcaster: trade_tx,
            notification_broadcaster: notification_tx,
            social_broadcaster: social_tx,
            whale_broadcaster: whale_tx,
            message_processor: Arc::new(MessageProcessor::new()),
            rate_limiter: Arc::new(WebSocketRateLimiter::new()),
            trenchware_ui: Arc::new(TrenchwareUI::new()),
            live_leaderboard: Arc::new(RwLock::new(LiveLeaderboard::new())),
        };
        
        info!("✅ WebSocket Manager initialized");
        Ok(manager)
    }
    
    /// **START CONNECTION MANAGER**
    /// Starts background tasks for connection management
    pub async fn start_connection_manager(&self) {
        info!("🔄 Starting WebSocket connection manager");
        
        // Start heartbeat checker
        let connections = self.connections.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(30));
            
            loop {
                interval.tick().await;
                Self::cleanup_stale_connections(&connections).await;
            }
        });
        
        // Start live leaderboard updater
        let live_leaderboard = self.live_leaderboard.clone();
        let social_broadcaster = self.social_broadcaster.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(10));
            
            loop {
                interval.tick().await;
                if let Ok(update) = live_leaderboard.read().await.generate_update() {
                    let _ = social_broadcaster.send(SocialUpdate::LeaderboardUpdate(update));
                }
            }
        });
        
        info!("✅ WebSocket connection manager started");
    }
    
    /// **ADD CONNECTION**
    /// Registers a new WebSocket connection
    pub async fn add_connection(&self, connection: ActiveConnection) -> String {
        let connection_id = connection.connection_id.clone();
        
        // Add to main connections
        self.connections.write().await.insert(connection_id.clone(), connection.clone());
        
        // Add to user connections if authenticated
        if let Some(user_id) = &connection.user_id {
            self.user_connections.write().await
                .entry(user_id.clone())
                .or_insert_with(Vec::new)
                .push(connection_id.clone());
        }
        
        info!("🔗 New WebSocket connection: {} (user: {:?})", 
              connection_id, connection.user_id);
        
        connection_id
    }
    
    /// **REMOVE CONNECTION**
    /// Removes a WebSocket connection and cleans up
    pub async fn remove_connection(&self, connection_id: &str) {
        let connection = self.connections.write().await.remove(connection_id);
        
        if let Some(conn) = connection {
            // Remove from user connections
            if let Some(user_id) = &conn.user_id {
                if let Some(user_conns) = self.user_connections.write().await.get_mut(user_id) {
                    user_conns.retain(|id| id != connection_id);
                    if user_conns.is_empty() {
                        self.user_connections.write().await.remove(user_id);
                    }
                }
            }
            
            info!("🔌 WebSocket disconnected: {} (user: {:?})", 
                  connection_id, conn.user_id);
        }
    }
    
    /// **BROADCAST TO USER**
    /// Send message to all connections for a specific user
    pub async fn broadcast_to_user(&self, user_id: &str, message: WebSocketMessage) -> Result<u32> {
        let user_connections = self.user_connections.read().await;
        
        if let Some(connection_ids) = user_connections.get(user_id) {
            let connections = self.connections.read().await;
            let mut sent_count = 0;
            
            for connection_id in connection_ids {
                if connections.contains_key(connection_id) {
                    // In a real implementation, we'd send via the WebSocket
                    // For now, we'll use the broadcast channels
                    sent_count += 1;
                }
            }
            
            debug!("📡 Broadcast to user {}: {} connections", user_id, sent_count);
            return Ok(sent_count);
        }
        
        Ok(0)
    }
    
    /// **BROADCAST PRICE UPDATE**
    /// Broadcast price updates to all subscribed connections
    pub async fn broadcast_price_update(&self, update: PriceUpdate) -> Result<()> {
        debug!("📈 Broadcasting price update for {}", update.token_address);
        
        let gaming_message = self.trenchware_ui.get_gaming_message(
            "price_update",
            crate::integrations::trenchware_ui::MessageContext {
                token_name: Some(update.token_name.clone()),
                pump_percent: Some(update.price_change_percent),
                ..Default::default()
            }
        );
        
        let enhanced_update = PriceUpdate {
            gaming_message: Some(gaming_message),
            ..update
        };
        
        self.price_broadcaster.send(enhanced_update)?;
        Ok(())
    }
    
    /// **BROADCAST WHALE ALERT**  
    /// Send whale alerts with gaming themes
    pub async fn broadcast_whale_alert(&self, alert: WhaleAlert) -> Result<()> {
        info!("🐋 Broadcasting whale alert: ${} {}", alert.amount, alert.action);
        
        let gaming_message = self.trenchware_ui.get_gaming_message(
            "whale_alert",
            crate::integrations::trenchware_ui::MessageContext {
                amount: Some(alert.amount),
                token_name: Some(alert.token_name.clone()),
                ..Default::default()
            }
        );
        
        let enhanced_alert = WhaleAlert {
            gaming_message: Some(gaming_message),
            ..alert
        };
        
        self.whale_broadcaster.send(WhaleUpdate::NewWhaleAlert(enhanced_alert))?;
        Ok(())
    }
    
    /// **BROADCAST RANK UPDATE**
    /// Send rank progression updates with themed messages
    pub async fn broadcast_rank_update(&self, user_id: &str, rank_update: RankUpdate) -> Result<()> {
        info!("🏆 Broadcasting rank update for user {}: {:?}", user_id, rank_update.new_rank);
        
        let gaming_message = match rank_update.rank_theme {
            Some(crate::integrations::ranking_system::RankTheme::RuneScape) => {
                format!("🎉 DING! LEVEL UP! Your Trading level is now {}! 🎽", rank_update.new_rank)
            },
            Some(crate::integrations::ranking_system::RankTheme::CallOfDuty) => {
                format!("🪖 PROMOTION EARNED! You've been promoted to {}! 🎖️", rank_update.new_rank)
            },
            Some(crate::integrations::ranking_system::RankTheme::Crypto) => {
                format!("💎 HANDS GETTING STRONGER! You've reached {} status! 🚀", rank_update.new_rank)
            },
            _ => {
                format!("🌟 Congratulations! You've reached {}! 🏆", rank_update.new_rank)
            }
        };
        
        let social_update = SocialUpdate::RankUpdate {
            user_id: user_id.to_string(),
            old_rank: rank_update.old_rank,
            new_rank: rank_update.new_rank.clone(),
            gaming_message,
            achievements_unlocked: rank_update.achievements_unlocked,
        };
        
        // Broadcast to all social feed subscribers
        self.social_broadcaster.send(social_update)?;
        
        // Send personal notification to the user
        let notification = NotificationUpdate {
            user_id: Some(user_id.to_string()),
            notification_type: NotificationType::RankUp,
            title: "🏆 RANK UP!".to_string(),
            message: gaming_message,
            timestamp: chrono::Utc::now(),
            action_url: Some("/rankings".to_string()),
        };
        
        self.notification_broadcaster.send(notification)?;
        Ok(())
    }
    
    // Helper method for cleanup
    async fn cleanup_stale_connections(connections: &Arc<RwLock<HashMap<String, ActiveConnection>>>) {
        let mut connections = connections.write().await;
        let now = std::time::Instant::now();
        
        connections.retain(|_, conn| {
            now.duration_since(conn.last_ping) < std::time::Duration::from_secs(60)
        });
    }
}

/// **WEBSOCKET HANDLERS**
/// Axum route handlers for WebSocket upgrades

pub async fn websocket_handler(
    ws: WebSocketUpgrade,
    Query(params): Query<WebSocketParams>,
    State(app_state): State<Arc<AppState>>,
) -> Response {
    info!("🔄 WebSocket upgrade request with params: {:?}", params);
    
    ws.on_upgrade(move |socket| {
        handle_websocket_connection(socket, params, app_state)
    })
}

pub async fn portfolio_websocket(
    ws: WebSocketUpgrade,
    State(app_state): State<Arc<AppState>>,
) -> Response {
    ws.on_upgrade(move |socket| {
        handle_portfolio_websocket(socket, app_state)
    })
}

pub async fn trading_websocket(
    ws: WebSocketUpgrade,
    State(app_state): State<Arc<AppState>>,
) -> Response {
    ws.on_upgrade(move |socket| {
        handle_trading_websocket(socket, app_state)
    })
}

pub async fn social_websocket(
    ws: WebSocketUpgrade,
    State(app_state): State<Arc<AppState>>,
) -> Response {
    ws.on_upgrade(move |socket| {
        handle_social_websocket(socket, app_state)
    })
}

/// **WEBSOCKET CONNECTION HANDLERS**

async fn handle_websocket_connection(
    socket: WebSocket,
    params: WebSocketParams,
    app_state: Arc<AppState>,
) {
    let connection_id = Uuid::new_v4().to_string();
    info!("🔗 Handling new WebSocket connection: {}", connection_id);
    
    let connection = ActiveConnection {
        connection_id: connection_id.clone(),
        user_id: params.user_id,
        wallet_address: params.wallet_address,
        subscriptions: params.subscriptions.unwrap_or_default(),
        last_ping: std::time::Instant::now(),
        message_count: 0,
        gaming_theme: params.gaming_theme,
        rank_theme: params.rank_theme,
    };
    
    // Add connection to manager (placeholder - in real implementation)
    // app_state.websocket_manager.add_connection(connection).await;
    
    // Handle the WebSocket connection
    let (mut sender, mut receiver) = socket.split();
    
    // Send welcome message with gaming theme
    let welcome_message = match connection.gaming_theme.as_deref() {
        Some("runescape") => "🏰 Welcome to TrenchScape! Your trading adventure begins now! ⚔️",
        Some("cod") => "🪖 Soldier reporting for duty! Ready for combat! 🎖️",
        Some("crypto") => "💎 Diamond hands activated! Let's get this bread! 🚀",
        Some("fortnite") => "🏆 Dropping into the trading zone! Victory Royale incoming! 👑",
        _ => "🤖 TrenchBot systems online! Ready to secure the bag! 💰"
    };
    
    let welcome = WebSocketMessage {
        message_type: MessageType::Welcome,
        data: serde_json::json!({
            "message": welcome_message,
            "connection_id": connection_id,
            "features": ["real-time-prices", "trading-signals", "whale-alerts", "social-feed"]
        }),
        timestamp: chrono::Utc::now(),
    };
    
    if let Ok(welcome_json) = serde_json::to_string(&welcome) {
        let _ = sender.send(Message::Text(welcome_json)).await;
    }
    
    // Start message processing loop
    while let Some(msg) = receiver.next().await {
        match msg {
            Ok(Message::Text(text)) => {
                if let Err(e) = handle_websocket_message(&text, &connection_id, &app_state).await {
                    warn!("Failed to handle WebSocket message: {}", e);
                }
            },
            Ok(Message::Ping(ping)) => {
                let _ = sender.send(Message::Pong(ping)).await;
            },
            Ok(Message::Close(_)) => {
                info!("🔌 WebSocket connection closed: {}", connection_id);
                break;
            },
            Err(e) => {
                warn!("WebSocket error for {}: {}", connection_id, e);
                break;
            },
            _ => {} // Handle other message types
        }
    }
    
    // Clean up connection
    // app_state.websocket_manager.remove_connection(&connection_id).await;
}

async fn handle_portfolio_websocket(socket: WebSocket, app_state: Arc<AppState>) {
    info!("📊 Portfolio WebSocket connection established");
    // Implementation for portfolio-specific WebSocket
}

async fn handle_trading_websocket(socket: WebSocket, app_state: Arc<AppState>) {
    info!("📈 Trading WebSocket connection established");
    // Implementation for trading-specific WebSocket
}

async fn handle_social_websocket(socket: WebSocket, app_state: Arc<AppState>) {
    info!("🤝 Social WebSocket connection established");
    // Implementation for social-specific WebSocket
}

async fn handle_websocket_message(
    message: &str,
    connection_id: &str,
    app_state: &Arc<AppState>,
) -> Result<()> {
    let parsed: WebSocketMessage = serde_json::from_str(message)?;
    
    debug!("📨 WebSocket message from {}: {:?}", connection_id, parsed.message_type);
    
    match parsed.message_type {
        MessageType::Subscribe => {
            // Handle subscription requests
            info!("📡 Subscription request from {}", connection_id);
        },
        MessageType::Unsubscribe => {
            // Handle unsubscription requests
            info!("📡 Unsubscription request from {}", connection_id);
        },
        MessageType::Ping => {
            // Handle ping/heartbeat
            debug!("💓 Heartbeat from {}", connection_id);
        },
        MessageType::TradingAction => {
            // Handle trading action requests
            info!("📈 Trading action from {}", connection_id);
        },
        _ => {
            debug!("🤷 Unhandled message type: {:?}", parsed.message_type);
        }
    }
    
    Ok(())
}

/// **SUPPORTING TYPES**

#[derive(Debug, Deserialize)]
pub struct WebSocketParams {
    pub user_id: Option<String>,
    pub wallet_address: Option<String>,
    pub subscriptions: Option<Vec<SubscriptionType>>,
    pub gaming_theme: Option<String>,
    pub rank_theme: Option<crate::integrations::ranking_system::RankTheme>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebSocketMessage {
    pub message_type: MessageType,
    pub data: serde_json::Value,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageType {
    Welcome,
    Subscribe,
    Unsubscribe,
    PriceUpdate,
    TradeUpdate,
    Notification,
    SocialUpdate,
    WhaleAlert,
    RankUpdate,
    TradingAction,
    Error,
    Ping,
    Pong,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceUpdate {
    pub token_address: String,
    pub token_name: String,
    pub price_usd: f64,
    pub price_change_percent: f64,
    pub volume_24h: f64,
    pub gaming_message: Option<String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeUpdate {
    pub trade_id: String,
    pub user_id: String,
    pub trade_type: String,
    pub token_address: String,
    pub amount: f64,
    pub price: f64,
    pub status: String,
    pub gaming_message: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationUpdate {
    pub user_id: Option<String>, // None for broadcast
    pub notification_type: NotificationType,
    pub title: String,
    pub message: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub action_url: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationType {
    TradeExecuted,
    ProfitAlert,
    LossAlert,
    WhaleAlert,
    RankUp,
    Achievement,
    SystemUpdate,
    CopyTradeExecuted,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SocialUpdate {
    NewPost { user_id: String, content: String },
    RankUpdate { 
        user_id: String, 
        old_rank: String, 
        new_rank: String, 
        gaming_message: String,
        achievements_unlocked: Vec<String>,
    },
    LeaderboardUpdate(LeaderboardUpdate),
    NewFollower { trader_id: String, follower_id: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WhaleUpdate {
    NewWhaleAlert(WhaleAlert),
    WhaleMovement { wallet: String, amount: f64, direction: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhaleAlert {
    pub whale_id: String,
    pub wallet_address: String,
    pub token_address: String,
    pub token_name: String,
    pub amount: f64,
    pub action: String, // "buy", "sell", "transfer"
    pub gaming_message: Option<String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankUpdate {
    pub old_rank: String,
    pub new_rank: String,
    pub rank_theme: Option<crate::integrations::ranking_system::RankTheme>,
    pub achievements_unlocked: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaderboardUpdate {
    pub top_traders: Vec<LeaderboardEntry>,
    pub my_rank: Option<u32>,
    pub rank_changes: Vec<RankChange>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaderboardEntry {
    pub rank: u32,
    pub user_id: String,
    pub display_name: String,
    pub profit: f64,
    pub win_rate: f64,
    pub gaming_title: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankChange {
    pub user_id: String,
    pub old_rank: u32,
    pub new_rank: u32,
    pub change: i32, // positive = rank up, negative = rank down
}

// **IMPLEMENTATION STUBS**
#[derive(Debug)] pub struct MessageProcessor;
#[derive(Debug)] pub struct WebSocketRateLimiter;
#[derive(Debug)] pub struct LiveLeaderboard;

impl MessageProcessor {
    fn new() -> Self { Self }
}

impl WebSocketRateLimiter {
    fn new() -> Self { Self }
}

impl LiveLeaderboard {
    fn new() -> Self { Self }
    
    fn generate_update(&self) -> Result<LeaderboardUpdate> {
        Ok(LeaderboardUpdate {
            top_traders: vec![],
            my_rank: None,
            rank_changes: vec![],
        })
    }
}

impl Default for crate::integrations::trenchware_ui::MessageContext {
    fn default() -> Self {
        Self {
            profit_amount: None,
            loss_percent: None,
            token_name: None,
            amount: None,
            entry_price: None,
            pump_percent: None,
            tokens_scanned: None,
        }
    }
}