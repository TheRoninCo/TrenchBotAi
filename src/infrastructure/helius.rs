use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::env;
use tokio::time::{sleep, Duration};
use tracing::{info, warn, error};

/// **HELIUS RPC CLIENT**
/// Advanced Helius API integration with enhanced parsing and webhooks
#[derive(Debug, Clone)]
pub struct HeliusClient {
    api_key: String,
    rpc_url: String,
    webhook_url: String,
    client: reqwest::Client,
    rate_limiter: RateLimiter,
}

impl HeliusClient {
    pub fn new() -> Result<Self> {
        let api_key = env::var("HELIUS_API_KEY")
            .map_err(|_| anyhow::anyhow!("HELIUS_API_KEY not found in environment"))?;
        
        let rpc_url = env::var("HELIUS_RPC_URL")
            .unwrap_or_else(|_| "https://devnet.helius-rpc.com/".to_string());
        
        let webhook_url = env::var("HELIUS_WEBHOOK_URL")
            .unwrap_or_else(|_| "https://api.helius.xyz/v0/webhooks".to_string());
        
        info!("🚀 Initializing Helius client");
        info!("  🔗 RPC URL: {}", rpc_url);
        info!("  📡 Webhook URL: {}", webhook_url);
        
        Ok(Self {
            api_key,
            rpc_url,
            webhook_url,
            client: reqwest::Client::new(),
            rate_limiter: RateLimiter::new(100, Duration::from_secs(60)), // 100 requests per minute
        })
    }
    
    /// **GET PARSED TRANSACTIONS**
    /// Enhanced transaction parsing with detailed metadata
    pub async fn get_parsed_transactions(&self, signatures: Vec<String>) -> Result<Vec<ParsedTransaction>> {
        self.rate_limiter.wait().await;
        
        let request = HeliusRpcRequest {
            method: "getParsedTransactions".to_string(),
            params: vec![signatures],
            id: 1,
            jsonrpc: "2.0".to_string(),
        };
        
        let url = format!("{}?api-key={}", self.rpc_url, self.api_key);
        let response = self.client
            .post(&url)
            .json(&request)
            .send()
            .await?;
        
        if !response.status().is_success() {
            error!("Helius API error: {}", response.status());
            return Err(anyhow::anyhow!("Helius API request failed: {}", response.status()));
        }
        
        let rpc_response: HeliusRpcResponse<Vec<ParsedTransaction>> = response.json().await?;
        
        if let Some(error) = rpc_response.error {
            return Err(anyhow::anyhow!("Helius RPC error: {}", error.message));
        }
        
        Ok(rpc_response.result.unwrap_or_default())
    }
    
    /// **GET PARSED TRANSACTION HISTORY**
    /// Get comprehensive transaction history for an address
    pub async fn get_parsed_transaction_history(&self, 
                                              address: &str, 
                                              limit: Option<u32>) -> Result<Vec<ParsedTransaction>> {
        self.rate_limiter.wait().await;
        
        let mut params = vec![address.to_string()];
        if let Some(limit) = limit {
            params.push(limit.to_string());
        }
        
        let request = HeliusRpcRequest {
            method: "getParsedTransactionHistory".to_string(),
            params: vec![params],
            id: 1,
            jsonrpc: "2.0".to_string(),
        };
        
        let url = format!("{}?api-key={}", self.rpc_url, self.api_key);
        let response = self.client
            .post(&url)
            .json(&request)
            .send()
            .await?;
        
        let rpc_response: HeliusRpcResponse<Vec<ParsedTransaction>> = response.json().await?;
        
        if let Some(error) = rpc_response.error {
            warn!("Helius transaction history error: {}", error.message);
            return Ok(vec![]);
        }
        
        let transactions = rpc_response.result.unwrap_or_default();
        info!("📜 Retrieved {} parsed transactions for {}", transactions.len(), address);
        
        Ok(transactions)
    }
    
    /// **GET ASSET**
    /// Get detailed asset information including metadata
    pub async fn get_asset(&self, asset_id: &str) -> Result<AssetInfo> {
        self.rate_limiter.wait().await;
        
        let url = format!("https://api.helius.xyz/v0/token-metadata?api-key={}", self.api_key);
        let request = TokenMetadataRequest {
            mint_accounts: vec![asset_id.to_string()],
            include_off_chain: true,
            disable_cache: false,
        };
        
        let response = self.client
            .post(&url)
            .json(&request)
            .send()
            .await?;
        
        if !response.status().is_success() {
            return Err(anyhow::anyhow!("Failed to get asset info: {}", response.status()));
        }
        
        let metadata: Vec<TokenMetadata> = response.json().await?;
        
        if let Some(token_meta) = metadata.first() {
            Ok(AssetInfo {
                mint: token_meta.mint.clone(),
                name: token_meta.on_chain_metadata.metadata.name.clone(),
                symbol: token_meta.on_chain_metadata.metadata.symbol.clone(),
                description: token_meta.off_chain_metadata.as_ref()
                    .and_then(|m| m.description.clone()),
                image: token_meta.off_chain_metadata.as_ref()
                    .and_then(|m| m.image.clone()),
                decimals: token_meta.on_chain_metadata.token_standard.as_ref()
                    .map(|_| 9) // Default to 9 decimals for SPL tokens
                    .unwrap_or(0),
                supply: token_meta.on_chain_metadata.slot.unwrap_or(0),
                is_mutable: token_meta.on_chain_metadata.mint_authority.is_some(),
            })
        } else {
            Err(anyhow::anyhow!("Asset not found: {}", asset_id))
        }
    }
    
    /// **GET ASSETS BY OWNER**
    /// Get all assets owned by a specific address
    pub async fn get_assets_by_owner(&self, owner: &str) -> Result<Vec<AssetInfo>> {
        self.rate_limiter.wait().await;
        
        let url = format!("https://api.helius.xyz/v0/token-metadata?api-key={}", self.api_key);
        let request = AssetsByOwnerRequest {
            owner_address: owner.to_string(),
            page: 1,
            limit: Some(1000),
        };
        
        let response = self.client
            .post(&url)
            .json(&request)
            .send()
            .await?;
        
        let assets_response: AssetsByOwnerResponse = response.json().await?;
        
        info!("💰 Found {} assets for owner {}", assets_response.total, owner);
        
        Ok(assets_response.items.into_iter().map(|asset| AssetInfo {
            mint: asset.id,
            name: asset.content.metadata.name,
            symbol: asset.content.metadata.symbol.unwrap_or_default(),
            description: asset.content.metadata.description,
            image: asset.content.files.first().map(|f| f.uri.clone()),
            decimals: 9, // Default for SPL tokens
            supply: 0,   // Would need additional call to get supply
            is_mutable: true, // Conservative assumption
        }).collect())
    }
    
    /// **CREATE WEBHOOK**
    /// Set up webhook for real-time transaction monitoring
    pub async fn create_webhook(&self, 
                              webhook_url: &str,
                              transaction_types: Vec<String>,
                              addresses: Vec<String>) -> Result<WebhookInfo> {
        self.rate_limiter.wait().await;
        
        let request = CreateWebhookRequest {
            webhook_url: webhook_url.to_string(),
            transaction_types,
            account_addresses: addresses,
            webhook_type: "enhanced".to_string(),
        };
        
        let url = format!("{}?api-key={}", self.webhook_url, self.api_key);
        let response = self.client
            .post(&url)
            .json(&request)
            .send()
            .await?;
        
        if !response.status().is_success() {
            return Err(anyhow::anyhow!("Failed to create webhook: {}", response.status()));
        }
        
        let webhook: WebhookInfo = response.json().await?;
        
        info!("🎣 Created webhook: {}", webhook.webhook_id);
        info!("  📡 URL: {}", webhook.webhook_url);
        info!("  📊 Types: {:?}", webhook.transaction_types);
        info!("  🏠 Addresses: {}", webhook.account_addresses.len());
        
        Ok(webhook)
    }
    
    /// **GET TRANSACTION SIGNATURES**
    /// Get transaction signatures for an address with advanced filtering
    pub async fn get_signatures_for_address(&self, 
                                           address: &str,
                                           limit: Option<u32>,
                                           before: Option<&str>,
                                           until: Option<&str>) -> Result<Vec<TransactionSignature>> {
        self.rate_limiter.wait().await;
        
        let mut params = vec![address.to_string()];
        
        // Build options object
        let mut options = serde_json::Map::new();
        if let Some(limit) = limit {
            options.insert("limit".to_string(), serde_json::Value::Number(limit.into()));
        }
        if let Some(before) = before {
            options.insert("before".to_string(), serde_json::Value::String(before.to_string()));
        }
        if let Some(until) = until {
            options.insert("until".to_string(), serde_json::Value::String(until.to_string()));
        }
        
        if !options.is_empty() {
            params.push(serde_json::to_string(&options)?);
        }
        
        let request = HeliusRpcRequest {
            method: "getSignaturesForAddress".to_string(),
            params: vec![params],
            id: 1,
            jsonrpc: "2.0".to_string(),
        };
        
        let url = format!("{}?api-key={}", self.rpc_url, self.api_key);
        let response = self.client
            .post(&url)
            .json(&request)
            .send()
            .await?;
        
        let rpc_response: HeliusRpcResponse<Vec<TransactionSignature>> = response.json().await?;
        
        if let Some(error) = rpc_response.error {
            return Err(anyhow::anyhow!("Failed to get signatures: {}", error.message));
        }
        
        let signatures = rpc_response.result.unwrap_or_default();
        info!("✍️  Retrieved {} transaction signatures for {}", signatures.len(), address);
        
        Ok(signatures)
    }
    
    /// **PRIORITY FEE RECOMMENDATIONS**
    /// Get recommended priority fees for faster transaction processing
    pub async fn get_priority_fee_estimate(&self, accounts: Vec<String>) -> Result<PriorityFeeEstimate> {
        self.rate_limiter.wait().await;
        
        let url = format!("https://api.helius.xyz/v0/priority-fee?api-key={}", self.api_key);
        let request = PriorityFeeRequest {
            accounts: Some(accounts),
            options: Some(PriorityFeeOptions {
                include_all_priority_fee_levels: true,
                transaction_encoding: Some("base64".to_string()),
            }),
        };
        
        let response = self.client
            .post(&url)
            .json(&request)
            .send()
            .await?;
        
        if !response.status().is_success() {
            return Err(anyhow::anyhow!("Failed to get priority fee estimate: {}", response.status()));
        }
        
        let fee_estimate: PriorityFeeEstimate = response.json().await?;
        
        info!("⛽ Priority fee estimate:");
        info!("  🟢 Low: {} lamports", fee_estimate.per_compute_unit.min);
        info!("  🟡 Medium: {} lamports", fee_estimate.per_compute_unit.medium);
        info!("  🔴 High: {} lamports", fee_estimate.per_compute_unit.max);
        
        Ok(fee_estimate)
    }
}

// Rate limiter implementation
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
        
        // Remove old requests outside the window
        requests.retain(|&timestamp| {
            now.duration_since(timestamp) < self.window_duration
        });
        
        // If we're at the limit, wait
        if requests.len() >= self.max_requests as usize {
            let oldest = requests[0];
            let wait_time = self.window_duration - now.duration_since(oldest);
            drop(requests); // Release the lock before sleeping
            sleep(wait_time).await;
        } else {
            requests.push(now);
        }
    }
}

// Data structures for Helius API responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedTransaction {
    pub signature: String,
    pub slot: u64,
    pub block_time: Option<i64>,
    pub meta: Option<TransactionMeta>,
    pub transaction: Transaction,
    #[serde(rename = "type")]
    pub transaction_type: Option<String>,
    pub description: Option<String>,
    pub events: Option<serde_json::Value>,
    pub fee: Option<u64>,
    pub native_transfers: Option<Vec<NativeTransfer>>,
    pub token_transfers: Option<Vec<TokenTransfer>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionMeta {
    pub err: Option<serde_json::Value>,
    pub fee: u64,
    pub pre_balances: Vec<u64>,
    pub post_balances: Vec<u64>,
    pub compute_units_consumed: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transaction {
    pub signatures: Vec<String>,
    pub message: TransactionMessage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionMessage {
    pub account_keys: Vec<String>,
    pub instructions: Vec<serde_json::Value>,
    pub recent_blockhash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NativeTransfer {
    pub from_user_account: String,
    pub to_user_account: String,
    pub amount: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenTransfer {
    pub from_user_account: Option<String>,
    pub to_user_account: Option<String>,
    pub mint: String,
    pub token_amount: f64,
    pub token_standard: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetInfo {
    pub mint: String,
    pub name: String,
    pub symbol: String,
    pub description: Option<String>,
    pub image: Option<String>,
    pub decimals: u8,
    pub supply: u64,
    pub is_mutable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookInfo {
    pub webhook_id: String,
    pub webhook_url: String,
    pub transaction_types: Vec<String>,
    pub account_addresses: Vec<String>,
    pub webhook_type: String,
    pub auth_header: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionSignature {
    pub signature: String,
    pub slot: u64,
    pub err: Option<serde_json::Value>,
    pub memo: Option<String>,
    pub block_time: Option<i64>,
    pub confirmation_status: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriorityFeeEstimate {
    pub per_compute_unit: PriorityFeePerComputeUnit,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriorityFeePerComputeUnit {
    pub min: u64,
    pub low: u64,
    pub medium: u64,
    pub high: u64,
    pub very_high: u64,
    pub unsafe_max: u64,
    pub max: u64,
}

// Request/Response structures
#[derive(Debug, Serialize)]
struct HeliusRpcRequest {
    method: String,
    params: Vec<serde_json::Value>,
    id: u32,
    jsonrpc: String,
}

#[derive(Debug, Deserialize)]
struct HeliusRpcResponse<T> {
    result: Option<T>,
    error: Option<RpcError>,
    id: u32,
    jsonrpc: String,
}

#[derive(Debug, Deserialize)]
struct RpcError {
    code: i32,
    message: String,
}

#[derive(Debug, Serialize)]
struct TokenMetadataRequest {
    mint_accounts: Vec<String>,
    include_off_chain: bool,
    disable_cache: bool,
}

#[derive(Debug, Deserialize)]
struct TokenMetadata {
    mint: String,
    on_chain_metadata: OnChainMetadata,
    off_chain_metadata: Option<OffChainMetadata>,
}

#[derive(Debug, Deserialize)]
struct OnChainMetadata {
    metadata: OnChainMetadataData,
    mint_authority: Option<String>,
    token_standard: Option<String>,
    slot: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct OnChainMetadataData {
    name: String,
    symbol: String,
    uri: String,
}

#[derive(Debug, Deserialize)]
struct OffChainMetadata {
    name: Option<String>,
    symbol: Option<String>,
    description: Option<String>,
    image: Option<String>,
}

#[derive(Debug, Serialize)]
struct AssetsByOwnerRequest {
    owner_address: String,
    page: u32,
    limit: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct AssetsByOwnerResponse {
    total: u32,
    limit: u32,
    page: u32,
    items: Vec<AssetItem>,
}

#[derive(Debug, Deserialize)]
struct AssetItem {
    id: String,
    content: AssetContent,
    ownership: AssetOwnership,
}

#[derive(Debug, Deserialize)]
struct AssetContent {
    metadata: AssetMetadata,
    files: Vec<AssetFile>,
}

#[derive(Debug, Deserialize)]
struct AssetMetadata {
    name: String,
    symbol: Option<String>,
    description: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AssetFile {
    uri: String,
    mime: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AssetOwnership {
    frozen: bool,
    delegated: bool,
}

#[derive(Debug, Serialize)]
struct CreateWebhookRequest {
    webhook_url: String,
    transaction_types: Vec<String>,
    account_addresses: Vec<String>,
    webhook_type: String,
}

#[derive(Debug, Serialize)]
struct PriorityFeeRequest {
    accounts: Option<Vec<String>>,
    options: Option<PriorityFeeOptions>,
}

#[derive(Debug, Serialize)]
struct PriorityFeeOptions {
    include_all_priority_fee_levels: bool,
    transaction_encoding: Option<String>,
}