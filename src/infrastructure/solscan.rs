use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use tokio::time::{sleep, Duration};
use tracing::{info, warn, error, debug};

/// **SOLSCAN API CLIENT**
/// Advanced Solscan integration for comprehensive blockchain analytics
#[derive(Debug, Clone)]
pub struct SolscanClient {
    api_key: String,
    base_url: String,
    client: reqwest::Client,
    rate_limiter: RateLimiter,
}

impl SolscanClient {
    pub fn new() -> Result<Self> {
        let api_key = env::var("SOLSCAN_API_KEY")
            .map_err(|_| anyhow::anyhow!("SOLSCAN_API_KEY not found in environment"))?;
        
        let base_url = "https://pro-api.solscan.io/v1.0".to_string();
        
        info!("🔍 Initializing Solscan client");
        info!("  🌐 Base URL: {}", base_url);
        
        Ok(Self {
            api_key,
            base_url,
            client: reqwest::Client::new(),
            rate_limiter: RateLimiter::new(20, Duration::from_secs(60)), // 20 requests per minute
        })
    }
    
    /// **GET ACCOUNT INFO**
    /// Comprehensive account analysis including balances and activity
    pub async fn get_account_info(&self, address: &str) -> Result<AccountInfo> {
        self.rate_limiter.wait().await;
        
        let url = format!("{}/account/{}", self.base_url, address);
        let response = self.client
            .get(&url)
            .header("token", &self.api_key)
            .send()
            .await?;
        
        if !response.status().is_success() {
            return Err(anyhow::anyhow!("Solscan API error: {}", response.status()));
        }
        
        let account_info: AccountInfo = response.json().await?;
        
        info!("👤 Account info for {}: {} SOL", address, account_info.lamports as f64 / 1e9);
        
        Ok(account_info)
    }
    
    /// **GET TOKEN BALANCES**
    /// Detailed token holdings for an account
    pub async fn get_token_balances(&self, address: &str) -> Result<Vec<TokenBalance>> {
        self.rate_limiter.wait().await;
        
        let url = format!("{}/account/tokens", self.base_url);
        let response = self.client
            .get(&url)
            .header("token", &self.api_key)
            .query(&[("account", address)])
            .send()
            .await?;
        
        if !response.status().is_success() {
            return Err(anyhow::anyhow!("Failed to get token balances: {}", response.status()));
        }
        
        let balances: Vec<TokenBalance> = response.json().await?;
        
        info!("💰 Found {} token balances for {}", balances.len(), address);
        
        Ok(balances)
    }
    
    /// **GET TRANSACTION HISTORY**
    /// Comprehensive transaction history with advanced filtering
    pub async fn get_transaction_history(&self, 
                                       address: &str,
                                       before: Option<&str>,
                                       limit: Option<u32>) -> Result<TransactionHistory> {
        self.rate_limiter.wait().await;
        
        let url = format!("{}/account/transactions", self.base_url);
        let mut params = vec![("account", address.to_string())];
        
        if let Some(before_sig) = before {
            params.push(("before", before_sig.to_string()));
        }
        
        if let Some(limit_val) = limit {
            params.push(("limit", limit_val.to_string()));
        }
        
        let response = self.client
            .get(&url)
            .header("token", &self.api_key)
            .query(&params)
            .send()
            .await?;
        
        if !response.status().is_success() {
            return Err(anyhow::anyhow!("Failed to get transaction history: {}", response.status()));
        }
        
        let history: TransactionHistory = response.json().await?;
        
        info!("📜 Retrieved {} transactions for {}", history.data.len(), address);
        
        Ok(history)
    }
    
    /// **GET TOKEN INFO**
    /// Detailed token metadata and market information
    pub async fn get_token_info(&self, token_address: &str) -> Result<TokenInfo> {
        self.rate_limiter.wait().await;
        
        let url = format!("{}/token/meta", self.base_url);
        let response = self.client
            .get(&url)
            .header("token", &self.api_key)
            .query(&[("tokenAddress", token_address)])
            .send()
            .await?;
        
        if !response.status().is_success() {
            return Err(anyhow::anyhow!("Failed to get token info: {}", response.status()));
        }
        
        let token_info: TokenInfo = response.json().await?;
        
        info!("🪙 Token info: {} ({}) - Supply: {}", 
              token_info.name, token_info.symbol, token_info.supply);
        
        Ok(token_info)
    }
    
    /// **GET TOKEN HOLDERS**
    /// Top holders analysis for token distribution insights
    pub async fn get_token_holders(&self, token_address: &str, limit: Option<u32>) -> Result<TokenHolders> {
        self.rate_limiter.wait().await;
        
        let url = format!("{}/token/holders", self.base_url);
        let mut params = vec![("tokenAddress", token_address.to_string())];
        
        if let Some(limit_val) = limit {
            params.push(("limit", limit_val.to_string()));
        }
        
        let response = self.client
            .get(&url)
            .header("token", &self.api_key)
            .query(&params)
            .send()
            .await?;
        
        if !response.status().is_success() {
            return Err(anyhow::anyhow!("Failed to get token holders: {}", response.status()));
        }
        
        let holders: TokenHolders = response.json().await?;
        
        info!("👥 Found {} token holders for {}", holders.data.len(), token_address);
        
        Ok(holders)
    }
    
    /// **GET TOKEN TRANSFERS**
    /// Recent transfer activity for a token
    pub async fn get_token_transfers(&self, 
                                   token_address: &str,
                                   limit: Option<u32>,
                                   offset: Option<u32>) -> Result<TokenTransfers> {
        self.rate_limiter.wait().await;
        
        let url = format!("{}/token/transfer", self.base_url);
        let mut params = vec![("tokenAddress", token_address.to_string())];
        
        if let Some(limit_val) = limit {
            params.push(("limit", limit_val.to_string()));
        }
        
        if let Some(offset_val) = offset {
            params.push(("offset", offset_val.to_string()));
        }
        
        let response = self.client
            .get(&url)
            .header("token", &self.api_key)
            .query(&params)
            .send()
            .await?;
        
        if !response.status().is_success() {
            return Err(anyhow::anyhow!("Failed to get token transfers: {}", response.status()));
        }
        
        let transfers: TokenTransfers = response.json().await?;
        
        info!("🔄 Retrieved {} token transfers", transfers.data.len());
        
        Ok(transfers)
    }
    
    /// **GET TRANSACTION DETAILS**
    /// Detailed transaction analysis including instructions and effects
    pub async fn get_transaction_details(&self, signature: &str) -> Result<TransactionDetail> {
        self.rate_limiter.wait().await;
        
        let url = format!("{}/transaction", self.base_url);
        let response = self.client
            .get(&url)
            .header("token", &self.api_key)
            .query(&[("tx", signature)])
            .send()
            .await?;
        
        if !response.status().is_success() {
            return Err(anyhow::anyhow!("Failed to get transaction details: {}", response.status()));
        }
        
        let transaction: TransactionDetail = response.json().await?;
        
        info!("📋 Transaction details: {} - Fee: {} lamports", 
              transaction.signature, transaction.fee);
        
        Ok(transaction)
    }
    
    /// **GET MARKET DATA**
    /// Token market information including price and volume
    pub async fn get_market_data(&self, token_address: &str) -> Result<MarketData> {
        self.rate_limiter.wait().await;
        
        let url = format!("{}/token/price", self.base_url);
        let response = self.client
            .get(&url)
            .header("token", &self.api_key)
            .query(&[("tokenAddress", token_address)])
            .send()
            .await?;
        
        if !response.status().is_success() {
            return Err(anyhow::anyhow!("Failed to get market data: {}", response.status()));
        }
        
        let market_data: MarketData = response.json().await?;
        
        info!("📈 Market data for {}: ${:.6} (24h: {:.2}%)", 
              token_address, market_data.price, market_data.price_change_24h);
        
        Ok(market_data)
    }
    
    /// **ANALYZE WHALE ACTIVITY**
    /// Detect large token movements and whale behavior
    pub async fn analyze_whale_activity(&self, 
                                      token_address: &str,
                                      min_amount: f64) -> Result<Vec<WhaleActivity>> {
        let transfers = self.get_token_transfers(token_address, Some(100), None).await?;
        let token_info = self.get_token_info(token_address).await?;
        
        let mut whale_activities = Vec::new();
        
        for transfer in transfers.data {
            let amount_normalized = transfer.amount as f64 / 10_f64.powi(token_info.decimals as i32);
            
            if amount_normalized >= min_amount {
                whale_activities.push(WhaleActivity {
                    signature: transfer.signature,
                    from: transfer.from_address,
                    to: transfer.to_address,
                    amount: amount_normalized,
                    amount_usd: amount_normalized * self.get_market_data(token_address).await?.price,
                    timestamp: transfer.block_time,
                    block_number: transfer.slot,
                });
            }
        }
        
        whale_activities.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        
        info!("🐋 Found {} whale activities for {} (min: {} tokens)", 
              whale_activities.len(), token_address, min_amount);
        
        Ok(whale_activities)
    }
    
    /// **GET DEFI POSITIONS**
    /// Analyze DeFi positions and liquidity across protocols
    pub async fn get_defi_positions(&self, address: &str) -> Result<Vec<DefiPosition>> {
        // This would require multiple API calls to get comprehensive DeFi data
        let account_info = self.get_account_info(address).await?;
        let token_balances = self.get_token_balances(address).await?;
        
        let mut defi_positions = Vec::new();
        
        // Analyze token balances for DeFi positions
        for balance in token_balances {
            if balance.amount > 0.0 {
                // Check if this is a known LP token or DeFi token
                if self.is_defi_token(&balance.token_address).await {
                    defi_positions.push(DefiPosition {
                        protocol: self.identify_protocol(&balance.token_address).await,
                        token_address: balance.token_address.clone(),
                        position_type: PositionType::LiquidityProvider,
                        amount: balance.amount,
                        value_usd: balance.amount_usd.unwrap_or(0.0),
                        apy: None, // Would need additional API calls
                    });
                }
            }
        }
        
        info!("🏦 Found {} DeFi positions for {}", defi_positions.len(), address);
        
        Ok(defi_positions)
    }
    
    // Helper methods
    async fn is_defi_token(&self, _token_address: &str) -> bool {
        // Simplified - in reality would check against known DeFi token lists
        true
    }
    
    async fn identify_protocol(&self, _token_address: &str) -> String {
        // Simplified - would analyze token metadata and known protocol addresses
        "Unknown".to_string()
    }
}

// Rate limiter (reused from helius.rs but with different limits)
#[derive(Debug)]
struct RateLimiter {
    max_requests: u32,
    window_duration: Duration,
    requests: std::sync::Arc<tokio::sync::Mutex<Vec<std::time::Instant>>>,
}

impl RateLimiter {
    fn new(max_requests: u32, window_duration: Duration) -> Self {
        Self {
            max_requests,
            window_duration,
            requests: std::sync::Arc::new(tokio::sync::Mutex::new(Vec::new())),
        }
    }
    
    async fn wait(&self) {
        let mut requests = self.requests.lock().await;
        let now = std::time::Instant::now();
        
        requests.retain(|&timestamp| {
            now.duration_since(timestamp) < self.window_duration
        });
        
        if requests.len() >= self.max_requests as usize {
            let oldest = requests[0];
            let wait_time = self.window_duration - now.duration_since(oldest);
            drop(requests);
            sleep(wait_time).await;
        } else {
            requests.push(now);
        }
    }
}

// Data structures for Solscan API responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccountInfo {
    pub account: String,
    pub lamports: u64,
    pub owner: String,
    pub executable: bool,
    pub rent_epoch: u64,
    pub type_: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenBalance {
    pub token_address: String,
    pub token_account: String,
    pub token_name: Option<String>,
    pub token_icon: Option<String>,
    pub token_symbol: Option<String>,
    pub amount: f64,
    pub decimals: u8,
    pub amount_usd: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionHistory {
    pub success: bool,
    pub data: Vec<TransactionSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionSummary {
    pub signature: String,
    pub slot: u64,
    pub block_time: i64,
    pub status: String,
    pub fee: u64,
    pub signer: Vec<String>,
    pub include_spl_transfer: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenInfo {
    pub token_address: String,
    pub name: String,
    pub symbol: String,
    pub icon: Option<String>,
    pub decimals: u8,
    pub supply: String,
    pub price: Option<f64>,
    pub volume_24h: Option<f64>,
    pub market_cap: Option<f64>,
    pub holder: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenHolders {
    pub success: bool,
    pub data: Vec<TokenHolder>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenHolder {
    pub address: String,
    pub amount: String,
    pub decimals: u8,
    pub owner: String,
    pub rank: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenTransfers {
    pub success: bool,
    pub data: Vec<TokenTransfer>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenTransfer {
    pub signature: String,
    pub slot: u64,
    pub block_time: i64,
    pub from_address: String,
    pub to_address: String,
    pub amount: u64,
    pub decimals: u8,
    pub token_address: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionDetail {
    pub signature: String,
    pub slot: u64,
    pub block_time: i64,
    pub status: String,
    pub fee: u64,
    pub compute_units_consumed: Option<u64>,
    pub recent_blockhash: String,
    pub instructions: Vec<serde_json::Value>,
    pub inner_instructions: Vec<serde_json::Value>,
    pub token_balances: Vec<serde_json::Value>,
    pub sol_balances: Vec<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketData {
    pub token_address: String,
    pub price: f64,
    pub price_change_24h: f64,
    pub price_change_7d: Option<f64>,
    pub volume_24h: f64,
    pub market_cap: Option<f64>,
    pub fully_diluted_valuation: Option<f64>,
    pub holders: Option<u64>,
    pub last_trade_unix_time: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhaleActivity {
    pub signature: String,
    pub from: String,
    pub to: String,
    pub amount: f64,
    pub amount_usd: f64,
    pub timestamp: i64,
    pub block_number: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DefiPosition {
    pub protocol: String,
    pub token_address: String,
    pub position_type: PositionType,
    pub amount: f64,
    pub value_usd: f64,
    pub apy: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PositionType {
    LiquidityProvider,
    Staking,
    Lending,
    Borrowing,
    Farming,
    Vault,
}