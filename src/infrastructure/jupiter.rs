use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use tokio::time::{sleep, Duration};
use tracing::{info, warn, error, debug};

/// **JUPITER API CLIENT**
/// Free Jupiter API integration for pricing, routing, and swap opportunities
#[derive(Debug, Clone)]
pub struct JupiterClient {
    quote_url: String,
    price_url: String,
    client: reqwest::Client,
    rate_limiter: RateLimiter,
}

impl JupiterClient {
    pub fn new() -> Result<Self> {
        let quote_url = env::var("JUPITER_API_URL")
            .unwrap_or_else(|_| "https://quote-api.jup.ag/v6".to_string());
        
        let price_url = env::var("JUPITER_PRICE_API_URL")
            .unwrap_or_else(|_| "https://price.jup.ag/v4/price".to_string());
        
        info!("🪐 Initializing Jupiter client (FREE tier)");
        info!("  🔗 Quote API: {}", quote_url);
        info!("  💰 Price API: {}", price_url);
        
        Ok(Self {
            quote_url,
            price_url,
            client: reqwest::Client::new(),
            rate_limiter: RateLimiter::new(60, Duration::from_secs(60)), // Conservative 60 per minute
        })
    }
    
    /// **GET QUOTE**
    /// Get the best swap quote across all DEXs
    pub async fn get_quote(&self, request: &QuoteRequest) -> Result<QuoteResponse> {
        self.rate_limiter.wait().await;
        
        let url = format!("{}/quote", self.quote_url);
        let mut params = vec![
            ("inputMint", request.input_mint.clone()),
            ("outputMint", request.output_mint.clone()),
            ("amount", request.amount.to_string()),
        ];
        
        if let Some(slippage) = request.slippage_bps {
            params.push(("slippageBps", slippage.to_string()));
        }
        
        if request.only_direct_routes {
            params.push(("onlyDirectRoutes", "true".to_string()));
        }
        
        if let Some(max_accounts) = request.max_accounts {
            params.push(("maxAccounts", max_accounts.to_string()));
        }
        
        let response = self.client
            .get(&url)
            .query(&params)
            .send()
            .await?;
        
        if !response.status().is_success() {
            return Err(anyhow::anyhow!("Jupiter quote API error: {}", response.status()));
        }
        
        let quote: QuoteResponse = response.json().await?;
        
        let input_amount = request.amount as f64 / 10_f64.powi(quote.input_mint_decimals as i32);
        let output_amount = quote.out_amount.parse::<u64>().unwrap_or(0) as f64 / 10_f64.powi(quote.output_mint_decimals as i32);
        let price_impact = quote.price_impact_pct.parse::<f64>().unwrap_or(0.0);
        
        info!("💱 Jupiter quote: {:.6} → {:.6} (impact: {:.3}%)", 
              input_amount, output_amount, price_impact);
        
        Ok(quote)
    }
    
    /// **GET PRICE**
    /// Get current token price in various currencies
    pub async fn get_price(&self, token_addresses: Vec<String>) -> Result<PriceResponse> {
        self.rate_limiter.wait().await;
        
        let ids = token_addresses.join(",");
        let url = format!("{}?ids={}&vsToken=USDC", self.price_url, ids);
        
        let response = self.client
            .get(&url)
            .send()
            .await?;
        
        if !response.status().is_success() {
            return Err(anyhow::anyhow!("Jupiter price API error: {}", response.status()));
        }
        
        let prices: PriceResponse = response.json().await?;
        
        info!("💲 Retrieved prices for {} tokens", prices.data.len());
        
        Ok(prices)
    }
    
    /// **GET PRICE HISTORY**
    /// Get historical price data for analysis
    pub async fn get_price_history(&self, 
                                 token_address: &str, 
                                 timeframe: PriceTimeframe) -> Result<Vec<PricePoint>> {
        self.rate_limiter.wait().await;
        
        let timeframe_str = match timeframe {
            PriceTimeframe::OneHour => "1H",
            PriceTimeframe::FourHour => "4H", 
            PriceTimeframe::OneDay => "1D",
            PriceTimeframe::OneWeek => "1W",
        };
        
        let url = format!("{}/history?id={}&timeframe={}", self.price_url, token_address, timeframe_str);
        
        let response = self.client
            .get(&url)
            .send()
            .await?;
        
        if !response.status().is_success() {
            warn!("Failed to get price history: {}", response.status());
            return Ok(vec![]);
        }
        
        let history: PriceHistoryResponse = response.json().await?;
        
        info!("📊 Retrieved {} price points for {}", history.data.len(), token_address);
        
        Ok(history.data)
    }
    
    /// **GET SWAP ROUTES**
    /// Get all possible routes for a swap
    pub async fn get_swap_routes(&self, 
                               input_mint: &str,
                               output_mint: &str,
                               amount: u64) -> Result<Vec<SwapRoute>> {
        let quote_request = QuoteRequest {
            input_mint: input_mint.to_string(),
            output_mint: output_mint.to_string(),
            amount,
            slippage_bps: Some(50), // 0.5% slippage
            only_direct_routes: false,
            max_accounts: None,
        };
        
        let quote = self.get_quote(&quote_request).await?;
        
        let mut routes = Vec::new();
        for (i, route_plan) in quote.route_plan.iter().enumerate() {
            routes.push(SwapRoute {
                route_id: i,
                input_mint: input_mint.to_string(),
                output_mint: output_mint.to_string(),
                amount_in: amount,
                amount_out: route_plan.swap_info.out_amount.parse().unwrap_or(0),
                price_impact_pct: quote.price_impact_pct.parse().unwrap_or(0.0),
                market_infos: route_plan.swap_info.amm_key.clone(),
                fees: vec![], // Would extract from route details
            });
        }
        
        info!("🛣️  Found {} swap routes", routes.len());
        
        Ok(routes)
    }
    
    /// **FIND ARBITRAGE OPPORTUNITIES**
    /// Scan for arbitrage opportunities across DEXs
    pub async fn find_arbitrage_opportunities(&self, 
                                            token_pairs: Vec<(String, String)>,
                                            min_profit_bps: u32) -> Result<Vec<ArbitrageOpportunity>> {
        let mut opportunities = Vec::new();
        
        for (token_a, token_b) in token_pairs {
            // Get quote A -> B
            let quote_ab = self.get_quote(&QuoteRequest {
                input_mint: token_a.clone(),
                output_mint: token_b.clone(),
                amount: 1_000_000, // 1 token with 6 decimals
                slippage_bps: Some(50),
                only_direct_routes: false,
                max_accounts: None,
            }).await;
            
            // Get quote B -> A
            let quote_ba = self.get_quote(&QuoteRequest {
                input_mint: token_b.clone(),
                output_mint: token_a.clone(),
                amount: 1_000_000,
                slippage_bps: Some(50),
                only_direct_routes: false,
                max_accounts: None,
            }).await;
            
            if let (Ok(ab_quote), Ok(ba_quote)) = (quote_ab, quote_ba) {
                let ab_output = ab_quote.out_amount.parse::<u64>().unwrap_or(0);
                let ba_output = ba_quote.out_amount.parse::<u64>().unwrap_or(0);
                
                // Calculate arbitrage profit potential
                let profit_ab = (ab_output as i64 - 1_000_000_i64) as f64 / 1_000_000_f64;
                let profit_ba = (ba_output as i64 - 1_000_000_i64) as f64 / 1_000_000_f64;
                
                let max_profit = profit_ab.max(profit_ba);
                let profit_bps = (max_profit * 10000.0) as u32;
                
                if profit_bps >= min_profit_bps {
                    opportunities.push(ArbitrageOpportunity {
                        token_a: token_a.clone(),
                        token_b: token_b.clone(),
                        profit_bps,
                        direction: if profit_ab > profit_ba { 
                            ArbitrageDirection::AtoB 
                        } else { 
                            ArbitrageDirection::BtoA 
                        },
                        estimated_gas_cost: 50000, // Estimate
                        max_input_amount: 1_000_000_000, // 1000 tokens
                        route_count: ab_quote.route_plan.len().max(ba_quote.route_plan.len()),
                    });
                }
            }
            
            // Rate limiting
            sleep(Duration::from_millis(100)).await;
        }
        
        opportunities.sort_by(|a, b| b.profit_bps.cmp(&a.profit_bps));
        
        info!("⚡ Found {} arbitrage opportunities", opportunities.len());
        
        Ok(opportunities)
    }
    
    /// **GET TOKEN LIST**
    /// Get list of all supported tokens
    pub async fn get_token_list(&self) -> Result<Vec<TokenInfo>> {
        self.rate_limiter.wait().await;
        
        let url = format!("{}/tokens", self.quote_url);
        
        let response = self.client
            .get(&url)
            .send()
            .await?;
        
        if !response.status().is_success() {
            return Err(anyhow::anyhow!("Failed to get token list: {}", response.status()));
        }
        
        let tokens: Vec<TokenInfo> = response.json().await?;
        
        info!("📋 Retrieved {} supported tokens", tokens.len());
        
        Ok(tokens)
    }
    
    /// **GET MARKET STATS**
    /// Get comprehensive market statistics
    pub async fn get_market_stats(&self, token_addresses: Vec<String>) -> Result<MarketStats> {
        let mut token_stats = HashMap::new();
        
        // Get prices for all tokens
        let prices = self.get_price(token_addresses.clone()).await?;
        
        // Get additional stats for each token
        for token_address in token_addresses {
            if let Some(price_data) = prices.data.get(&token_address) {
                // Get price history for volatility calculation
                let history = self.get_price_history(&token_address, PriceTimeframe::OneDay).await?;
                
                let volatility = if history.len() > 1 {
                    let prices: Vec<f64> = history.iter().map(|p| p.price).collect();
                    calculate_volatility(&prices)
                } else {
                    0.0
                };
                
                token_stats.insert(token_address.clone(), TokenStats {
                    address: token_address.clone(),
                    current_price: price_data.price,
                    price_change_24h: 0.0, // Would need to calculate from history
                    volume_24h: 0.0,       // Not available in price API
                    market_cap: 0.0,       // Would need supply data
                    volatility,
                    last_updated: std::time::SystemTime::now(),
                });
            }
        }
        
        Ok(MarketStats {
            tokens: token_stats,
            total_tokens: token_stats.len(),
            timestamp: std::time::SystemTime::now(),
        })
    }
    
    /// **ANALYZE LIQUIDITY**
    /// Analyze liquidity across different DEXs for a token pair
    pub async fn analyze_liquidity(&self, 
                                 input_mint: &str,
                                 output_mint: &str,
                                 test_amounts: Vec<u64>) -> Result<LiquidityAnalysis> {
        let mut liquidity_points = Vec::new();
        
        for amount in test_amounts {
            if let Ok(quote) = self.get_quote(&QuoteRequest {
                input_mint: input_mint.to_string(),
                output_mint: output_mint.to_string(),
                amount,
                slippage_bps: Some(50),
                only_direct_routes: false,
                max_accounts: None,
            }).await {
                let price_impact = quote.price_impact_pct.parse::<f64>().unwrap_or(0.0);
                let output_amount = quote.out_amount.parse::<u64>().unwrap_or(0);
                
                liquidity_points.push(LiquidityPoint {
                    input_amount: amount,
                    output_amount,
                    price_impact,
                    route_count: quote.route_plan.len(),
                });
            }
            
            // Small delay to avoid rate limiting
            sleep(Duration::from_millis(200)).await;
        }
        
        let analysis = LiquidityAnalysis {
            pair: format!("{}/{}", input_mint, output_mint),
            liquidity_points,
            deep_liquidity_threshold: calculate_deep_liquidity_threshold(&liquidity_points),
            max_efficient_size: calculate_max_efficient_size(&liquidity_points),
            average_price_impact: calculate_average_price_impact(&liquidity_points),
        };
        
        info!("💧 Liquidity analysis: {} points, max efficient: {}", 
              analysis.liquidity_points.len(), analysis.max_efficient_size);
        
        Ok(analysis)
    }
}

// Helper functions
fn calculate_volatility(prices: &[f64]) -> f64 {
    if prices.len() < 2 { return 0.0; }
    
    let mean = prices.iter().sum::<f64>() / prices.len() as f64;
    let variance = prices.iter()
        .map(|p| (p - mean).powi(2))
        .sum::<f64>() / prices.len() as f64;
    
    variance.sqrt() / mean // Coefficient of variation
}

fn calculate_deep_liquidity_threshold(points: &[LiquidityPoint]) -> u64 {
    // Find amount where price impact exceeds 1%
    points.iter()
        .find(|p| p.price_impact > 1.0)
        .map(|p| p.input_amount)
        .unwrap_or(0)
}

fn calculate_max_efficient_size(points: &[LiquidityPoint]) -> u64 {
    // Find amount where price impact exceeds 0.5%
    points.iter()
        .find(|p| p.price_impact > 0.5)
        .map(|p| p.input_amount)
        .unwrap_or(0)
}

fn calculate_average_price_impact(points: &[LiquidityPoint]) -> f64 {
    if points.is_empty() { return 0.0; }
    points.iter().map(|p| p.price_impact).sum::<f64>() / points.len() as f64
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

// Data structures for Jupiter API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuoteRequest {
    pub input_mint: String,
    pub output_mint: String,
    pub amount: u64,
    pub slippage_bps: Option<u32>,
    pub only_direct_routes: bool,
    pub max_accounts: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuoteResponse {
    #[serde(rename = "inputMint")]
    pub input_mint: String,
    #[serde(rename = "inAmount")]
    pub in_amount: String,
    #[serde(rename = "outputMint")]
    pub output_mint: String,
    #[serde(rename = "outAmount")]
    pub out_amount: String,
    #[serde(rename = "otherAmountThreshold")]
    pub other_amount_threshold: String,
    #[serde(rename = "swapMode")]
    pub swap_mode: String,
    #[serde(rename = "slippageBps")]
    pub slippage_bps: u32,
    #[serde(rename = "platformFee")]
    pub platform_fee: Option<PlatformFee>,
    #[serde(rename = "priceImpactPct")]
    pub price_impact_pct: String,
    #[serde(rename = "routePlan")]
    pub route_plan: Vec<RoutePlan>,
    #[serde(rename = "contextSlot")]
    pub context_slot: u64,
    #[serde(rename = "timeTaken")]
    pub time_taken: f64,
    #[serde(rename = "inputMintDecimals")]
    pub input_mint_decimals: u8,
    #[serde(rename = "outputMintDecimals")]
    pub output_mint_decimals: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformFee {
    pub amount: String,
    #[serde(rename = "feeBps")]
    pub fee_bps: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutePlan {
    #[serde(rename = "swapInfo")]
    pub swap_info: SwapInfo,
    pub percent: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwapInfo {
    #[serde(rename = "ammKey")]
    pub amm_key: String,
    pub label: String,
    #[serde(rename = "inputMint")]
    pub input_mint: String,
    #[serde(rename = "outputMint")]
    pub output_mint: String,
    #[serde(rename = "inAmount")]
    pub in_amount: String,
    #[serde(rename = "outAmount")]
    pub out_amount: String,
    #[serde(rename = "feeAmount")]
    pub fee_amount: String,
    #[serde(rename = "feeMint")]
    pub fee_mint: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceResponse {
    pub data: HashMap<String, TokenPrice>,
    #[serde(rename = "timeTaken")]
    pub time_taken: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenPrice {
    pub id: String,
    pub price: f64,
    #[serde(rename = "extraInfo")]
    pub extra_info: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceHistoryResponse {
    pub success: bool,
    pub data: Vec<PricePoint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PricePoint {
    pub timestamp: i64,
    pub price: f64,
}

#[derive(Debug, Clone)]
pub enum PriceTimeframe {
    OneHour,
    FourHour,
    OneDay,
    OneWeek,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwapRoute {
    pub route_id: usize,
    pub input_mint: String,
    pub output_mint: String,
    pub amount_in: u64,
    pub amount_out: u64,
    pub price_impact_pct: f64,
    pub market_infos: String,
    pub fees: Vec<SwapFee>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwapFee {
    pub amount: u64,
    pub mint: String,
    pub pct: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArbitrageOpportunity {
    pub token_a: String,
    pub token_b: String,
    pub profit_bps: u32,
    pub direction: ArbitrageDirection,
    pub estimated_gas_cost: u64,
    pub max_input_amount: u64,
    pub route_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArbitrageDirection {
    AtoB,
    BtoA,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenInfo {
    pub address: String,
    pub name: String,
    pub symbol: String,
    pub decimals: u8,
    pub logo_uri: Option<String>,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct MarketStats {
    pub tokens: HashMap<String, TokenStats>,
    pub total_tokens: usize,
    pub timestamp: std::time::SystemTime,
}

#[derive(Debug, Clone)]
pub struct TokenStats {
    pub address: String,
    pub current_price: f64,
    pub price_change_24h: f64,
    pub volume_24h: f64,
    pub market_cap: f64,
    pub volatility: f64,
    pub last_updated: std::time::SystemTime,
}

#[derive(Debug, Clone)]
pub struct LiquidityAnalysis {
    pub pair: String,
    pub liquidity_points: Vec<LiquidityPoint>,
    pub deep_liquidity_threshold: u64,
    pub max_efficient_size: u64,
    pub average_price_impact: f64,
}

#[derive(Debug, Clone)]
pub struct LiquidityPoint {
    pub input_amount: u64,
    pub output_amount: u64,
    pub price_impact: f64,
    pub route_count: usize,
}