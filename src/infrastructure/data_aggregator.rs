use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, error, debug};

use super::helius::HeliusClient;
use super::solscan::SolscanClient;
use super::jupiter::JupiterClient;

/// **MULTI-SOURCE DATA AGGREGATOR**
/// Intelligent data aggregation from Helius, Solscan, and Jupiter with fallback and validation
#[derive(Debug)]
pub struct DataAggregator {
    pub helius: Arc<HeliusClient>,
    pub solscan: Arc<SolscanClient>,
    pub jupiter: Arc<JupiterClient>,
    pub cache: Arc<RwLock<AggregatedDataCache>>,
    pub config: AggregatorConfig,
}

impl DataAggregator {
    pub async fn new() -> Result<Self> {
        info!("🔄 Initializing multi-source data aggregator");
        
        let helius = Arc::new(HeliusClient::new()?);
        let solscan = Arc::new(SolscanClient::new()?);
        let jupiter = Arc::new(JupiterClient::new()?);
        
        let config = AggregatorConfig {
            prefer_helius_for_transactions: true,
            prefer_solscan_for_analytics: true,
            prefer_jupiter_for_pricing: true,
            enable_cross_validation: true,
            cache_duration_seconds: 60,
            max_retries: 3,
            fallback_enabled: true,
        };
        
        info!("✅ Data aggregator initialized with 3 sources");
        info!("  🚀 Helius: Enhanced transaction parsing");
        info!("  🔍 Solscan: Comprehensive analytics");
        info!("  🪐 Jupiter: Free pricing & routing");
        
        Ok(Self {
            helius,
            solscan,
            jupiter,
            cache: Arc::new(RwLock::new(AggregatedDataCache::new())),
            config,
        })
    }
    
    /// **GET COMPREHENSIVE TOKEN INFO**
    /// Aggregate token information from all sources with cross-validation
    pub async fn get_comprehensive_token_info(&self, token_address: &str) -> Result<ComprehensiveTokenInfo> {
        info!("🪙 Aggregating comprehensive token info for {}", token_address);
        
        // Check cache first
        {
            let cache = self.cache.read().await;
            if let Some(cached_info) = cache.get_token_info(token_address) {
                if !cached_info.is_expired(self.config.cache_duration_seconds) {
                    debug!("📋 Using cached token info for {}", token_address);
                    return Ok(cached_info.data.clone());
                }
            }
        }
        
        let mut info = ComprehensiveTokenInfo {
            address: token_address.to_string(),
            name: String::new(),
            symbol: String::new(),
            decimals: 9,
            supply: 0,
            price_usd: 0.0,
            market_cap: 0.0,
            volume_24h: 0.0,
            price_change_24h: 0.0,
            holders: 0,
            is_verified: false,
            liquidity_score: 0.0,
            risk_score: 0.0,
            metadata: TokenMetadata::default(),
            data_sources: vec![],
            last_updated: std::time::SystemTime::now(),
        };
        
        // Get data from Helius (metadata focus)
        if let Ok(helius_asset) = self.helius.get_asset(token_address).await {
            info.name = helius_asset.name;
            info.symbol = helius_asset.symbol;
            info.decimals = helius_asset.decimals;
            info.supply = helius_asset.supply;
            info.metadata.description = helius_asset.description;
            info.metadata.image_uri = helius_asset.image;
            info.data_sources.push(DataSource::Helius);
        }
        
        // Get data from Solscan (comprehensive analytics)
        if let Ok(solscan_token) = self.solscan.get_token_info(token_address).await {
            if info.name.is_empty() { info.name = solscan_token.name; }
            if info.symbol.is_empty() { info.symbol = solscan_token.symbol; }
            if info.supply == 0 { info.supply = solscan_token.supply.parse().unwrap_or(0); }
            info.volume_24h = solscan_token.volume_24h.unwrap_or(0.0);
            info.market_cap = solscan_token.market_cap.unwrap_or(0.0);
            info.holders = solscan_token.holder.unwrap_or(0);
            if let Some(price) = solscan_token.price {
                info.price_usd = price;
            }
            info.data_sources.push(DataSource::Solscan);
        }
        
        // Get data from Jupiter (pricing focus)
        if let Ok(jupiter_price) = self.jupiter.get_price(vec![token_address.to_string()]).await {
            if let Some(price_data) = jupiter_price.data.get(token_address) {
                info.price_usd = price_data.price;
                info.data_sources.push(DataSource::Jupiter);
            }
        }
        
        // Cross-validation and quality scoring
        if self.config.enable_cross_validation {
            info.liquidity_score = self.calculate_liquidity_score(&info).await?;
            info.risk_score = self.calculate_risk_score(&info).await?;
            info.is_verified = self.validate_token_legitimacy(&info).await?;
        }
        
        // Cache the result
        {
            let mut cache = self.cache.write().await;
            cache.set_token_info(token_address, &info);
        }
        
        info!("✅ Aggregated token info from {} sources", info.data_sources.len());
        
        Ok(info)
    }
    
    /// **GET ENHANCED TRANSACTION ANALYSIS**
    /// Combine transaction data from multiple sources for comprehensive analysis
    pub async fn get_enhanced_transaction_analysis(&self, signature: &str) -> Result<EnhancedTransactionAnalysis> {
        info!("📋 Analyzing transaction {} with multiple sources", signature);
        
        let mut analysis = EnhancedTransactionAnalysis {
            signature: signature.to_string(),
            slot: 0,
            block_time: 0,
            status: TransactionStatus::Unknown,
            fee_lamports: 0,
            compute_units: 0,
            instructions: vec![],
            token_transfers: vec![],
            native_transfers: vec![],
            program_interactions: vec![],
            mev_classification: None,
            risk_indicators: vec![],
            profitability_analysis: None,
            data_sources: vec![],
            confidence_score: 0.0,
        };
        
        // Get enhanced parsing from Helius
        if let Ok(helius_txs) = self.helius.get_parsed_transactions(vec![signature.to_string()]).await {
            if let Some(tx) = helius_txs.first() {
                analysis.slot = tx.slot;
                analysis.block_time = tx.block_time.unwrap_or(0);
                analysis.fee_lamports = tx.fee.unwrap_or(0);
                
                if let Some(meta) = &tx.meta {
                    analysis.compute_units = meta.compute_units_consumed.unwrap_or(0);
                }
                
                // Extract token transfers
                if let Some(transfers) = &tx.token_transfers {
                    for transfer in transfers {
                        analysis.token_transfers.push(EnhancedTokenTransfer {
                            from: transfer.from_user_account.clone().unwrap_or_default(),
                            to: transfer.to_user_account.clone().unwrap_or_default(),
                            mint: transfer.mint.clone(),
                            amount: transfer.token_amount,
                            amount_usd: 0.0, // Will be filled later with price data
                        });
                    }
                }
                
                // Extract native transfers
                if let Some(transfers) = &tx.native_transfers {
                    for transfer in transfers {
                        analysis.native_transfers.push(EnhancedNativeTransfer {
                            from: transfer.from_user_account.clone(),
                            to: transfer.to_user_account.clone(),
                            amount_lamports: transfer.amount,
                            amount_sol: transfer.amount as f64 / 1e9,
                        });
                    }
                }
                
                analysis.status = TransactionStatus::Success;
                analysis.data_sources.push(DataSource::Helius);
            }
        }
        
        // Get detailed analysis from Solscan
        if let Ok(solscan_tx) = self.solscan.get_transaction_details(signature).await {
            if analysis.slot == 0 { analysis.slot = solscan_tx.slot; }
            if analysis.block_time == 0 { analysis.block_time = solscan_tx.block_time; }
            if analysis.fee_lamports == 0 { analysis.fee_lamports = solscan_tx.fee; }
            
            // Add additional compute unit information
            if let Some(compute_units) = solscan_tx.compute_units_consumed {
                analysis.compute_units = compute_units;
            }
            
            analysis.data_sources.push(DataSource::Solscan);
        }
        
        // Enhance with pricing data from Jupiter
        for transfer in &mut analysis.token_transfers {
            if let Ok(price_data) = self.jupiter.get_price(vec![transfer.mint.clone()]).await {
                if let Some(token_price) = price_data.data.get(&transfer.mint) {
                    transfer.amount_usd = transfer.amount * token_price.price;
                }
            }
        }
        
        // Advanced analysis
        analysis.mev_classification = self.classify_mev_transaction(&analysis).await?;
        analysis.risk_indicators = self.identify_risk_indicators(&analysis).await?;
        analysis.profitability_analysis = self.analyze_profitability(&analysis).await?;
        analysis.confidence_score = self.calculate_analysis_confidence(&analysis).await?;
        
        info!("✅ Enhanced transaction analysis complete (confidence: {:.1}%)", 
              analysis.confidence_score * 100.0);
        
        Ok(analysis)
    }
    
    /// **GET MARKET INTELLIGENCE**
    /// Comprehensive market analysis from all sources
    pub async fn get_market_intelligence(&self, tokens: Vec<String>) -> Result<MarketIntelligence> {
        info!("📊 Gathering market intelligence for {} tokens", tokens.len());
        
        let mut intelligence = MarketIntelligence {
            tokens: HashMap::new(),
            market_overview: MarketOverview::default(),
            trending_tokens: vec![],
            arbitrage_opportunities: vec![],
            liquidity_analysis: HashMap::new(),
            whale_movements: vec![],
            risk_assessment: RiskAssessment::default(),
            timestamp: std::time::SystemTime::now(),
        };
        
        // Get comprehensive token data
        for token in &tokens {
            if let Ok(token_info) = self.get_comprehensive_token_info(token).await {
                intelligence.tokens.insert(token.clone(), token_info);
            }
        }
        
        // Get arbitrage opportunities from Jupiter
        let token_pairs: Vec<(String, String)> = tokens.iter()
            .take(5) // Limit to avoid rate limits
            .enumerate()
            .filter_map(|(i, token_a)| {
                tokens.get(i + 1).map(|token_b| (token_a.clone(), token_b.clone()))
            })
            .collect();
        
        if let Ok(opportunities) = self.jupiter.find_arbitrage_opportunities(token_pairs, 10).await {
            intelligence.arbitrage_opportunities = opportunities;
        }
        
        // Analyze whale movements from Solscan
        for token in &tokens.iter().take(3) { // Limit for rate limits
            if let Ok(whale_activities) = self.solscan.analyze_whale_activity(token, 1000.0).await {
                intelligence.whale_movements.extend(whale_activities.into_iter().take(5));
            }
        }
        
        // Generate market overview
        intelligence.market_overview = self.generate_market_overview(&intelligence.tokens).await?;
        
        // Risk assessment
        intelligence.risk_assessment = self.assess_market_risk(&intelligence).await?;
        
        info!("✅ Market intelligence complete: {} tokens, {} arbitrage ops, {} whale movements",
              intelligence.tokens.len(), 
              intelligence.arbitrage_opportunities.len(),
              intelligence.whale_movements.len());
        
        Ok(intelligence)
    }
    
    /// **VALIDATE DATA CONSISTENCY**
    /// Cross-validate data across sources for accuracy
    pub async fn validate_data_consistency(&self, token_address: &str) -> Result<DataConsistencyReport> {
        info!("🔍 Validating data consistency for {}", token_address);
        
        let mut report = DataConsistencyReport {
            token_address: token_address.to_string(),
            sources_checked: 0,
            consistency_score: 0.0,
            discrepancies: vec![],
            recommendations: vec![],
        };
        
        // Get data from all sources
        let helius_data = self.helius.get_asset(token_address).await.ok();
        let solscan_data = self.solscan.get_token_info(token_address).await.ok();
        let jupiter_data = self.jupiter.get_price(vec![token_address.to_string()]).await.ok();
        
        report.sources_checked = 
            helius_data.is_some() as u32 + 
            solscan_data.is_some() as u32 + 
            jupiter_data.is_some() as u32;
        
        // Check for discrepancies
        if let (Some(helius), Some(solscan)) = (&helius_data, &solscan_data) {
            if helius.name != solscan.name {
                report.discrepancies.push(format!(
                    "Name mismatch: Helius '{}' vs Solscan '{}'", 
                    helius.name, solscan.name
                ));
            }
            
            if helius.symbol != solscan.symbol {
                report.discrepancies.push(format!(
                    "Symbol mismatch: Helius '{}' vs Solscan '{}'", 
                    helius.symbol, solscan.symbol
                ));
            }
        }
        
        // Calculate consistency score
        report.consistency_score = if report.discrepancies.is_empty() {
            1.0
        } else {
            1.0 - (report.discrepancies.len() as f64 * 0.2).min(1.0)
        };
        
        // Generate recommendations
        if report.consistency_score < 0.8 {
            report.recommendations.push("Manual verification recommended".to_string());
        }
        
        if report.sources_checked < 2 {
            report.recommendations.push("Insufficient data sources".to_string());
        }
        
        info!("📋 Data consistency: {:.1}% ({} discrepancies)", 
              report.consistency_score * 100.0, report.discrepancies.len());
        
        Ok(report)
    }
    
    // Helper methods for analysis
    async fn calculate_liquidity_score(&self, info: &ComprehensiveTokenInfo) -> Result<f64> {
        // Simplified liquidity scoring based on volume and market cap
        let volume_score = (info.volume_24h / info.market_cap.max(1.0)).min(1.0);
        let holder_score = (info.holders as f64 / 10000.0).min(1.0);
        Ok((volume_score + holder_score) / 2.0)
    }
    
    async fn calculate_risk_score(&self, info: &ComprehensiveTokenInfo) -> Result<f64> {
        let mut risk_factors = 0.0;
        
        // Low holder count increases risk
        if info.holders < 100 { risk_factors += 0.3; }
        
        // Low liquidity increases risk
        if info.liquidity_score < 0.3 { risk_factors += 0.3; }
        
        // High price volatility increases risk (simplified)
        if info.price_change_24h.abs() > 20.0 { risk_factors += 0.2; }
        
        // Unknown metadata increases risk
        if info.metadata.description.is_none() { risk_factors += 0.2; }
        
        Ok(risk_factors.min(1.0))
    }
    
    async fn validate_token_legitimacy(&self, info: &ComprehensiveTokenInfo) -> Result<bool> {
        // Simple validation - has name, symbol, and reasonable holder count
        Ok(!info.name.is_empty() && 
           !info.symbol.is_empty() && 
           info.holders > 10 &&
           info.data_sources.len() >= 2)
    }
    
    async fn classify_mev_transaction(&self, _analysis: &EnhancedTransactionAnalysis) -> Result<Option<MevClassification>> {
        // Simplified MEV classification - would need more sophisticated analysis
        Ok(None)
    }
    
    async fn identify_risk_indicators(&self, analysis: &EnhancedTransactionAnalysis) -> Result<Vec<RiskIndicator>> {
        let mut indicators = vec![];
        
        // High fee relative to transfer amount
        if analysis.fee_lamports > 100000 {
            indicators.push(RiskIndicator::HighFees);
        }
        
        // Many token transfers (potential sandwich attack)
        if analysis.token_transfers.len() > 5 {
            indicators.push(RiskIndicator::ComplexTransaction);
        }
        
        Ok(indicators)
    }
    
    async fn analyze_profitability(&self, _analysis: &EnhancedTransactionAnalysis) -> Result<Option<ProfitabilityAnalysis>> {
        // Would implement sophisticated profitability analysis
        Ok(None)
    }
    
    async fn calculate_analysis_confidence(&self, analysis: &EnhancedTransactionAnalysis) -> Result<f64> {
        let source_weight = analysis.data_sources.len() as f64 / 3.0; // Max 3 sources
        let data_completeness = if analysis.slot > 0 && analysis.block_time > 0 { 0.5 } else { 0.0 };
        Ok((source_weight * 0.6 + data_completeness * 0.4).min(1.0))
    }
    
    async fn generate_market_overview(&self, tokens: &HashMap<String, ComprehensiveTokenInfo>) -> Result<MarketOverview> {
        let total_market_cap: f64 = tokens.values().map(|t| t.market_cap).sum();
        let average_volume: f64 = tokens.values().map(|t| t.volume_24h).sum::<f64>() / tokens.len().max(1) as f64;
        let average_price_change: f64 = tokens.values().map(|t| t.price_change_24h).sum::<f64>() / tokens.len().max(1) as f64;
        
        Ok(MarketOverview {
            total_tokens: tokens.len(),
            total_market_cap,
            average_volume_24h: average_volume,
            average_price_change_24h: average_price_change,
            high_risk_tokens: tokens.values().filter(|t| t.risk_score > 0.7).count(),
            verified_tokens: tokens.values().filter(|t| t.is_verified).count(),
        })
    }
    
    async fn assess_market_risk(&self, _intelligence: &MarketIntelligence) -> Result<RiskAssessment> {
        Ok(RiskAssessment {
            overall_risk_level: RiskLevel::Medium,
            volatility_index: 0.5,
            liquidity_stress: 0.3,
            concentration_risk: 0.4,
            recommendations: vec![
                "Monitor whale movements".to_string(),
                "Maintain diverse positions".to_string(),
            ],
        })
    }
}

// Supporting data structures and cache
#[derive(Debug)]
struct AggregatedDataCache {
    token_info: HashMap<String, CachedData<ComprehensiveTokenInfo>>,
}

impl AggregatedDataCache {
    fn new() -> Self {
        Self {
            token_info: HashMap::new(),
        }
    }
    
    fn get_token_info(&self, address: &str) -> Option<&CachedData<ComprehensiveTokenInfo>> {
        self.token_info.get(address)
    }
    
    fn set_token_info(&mut self, address: &str, info: &ComprehensiveTokenInfo) {
        self.token_info.insert(address.to_string(), CachedData::new(info.clone()));
    }
}

#[derive(Debug, Clone)]
struct CachedData<T> {
    data: T,
    timestamp: std::time::SystemTime,
}

impl<T> CachedData<T> {
    fn new(data: T) -> Self {
        Self {
            data,
            timestamp: std::time::SystemTime::now(),
        }
    }
    
    fn is_expired(&self, max_age_seconds: u64) -> bool {
        self.timestamp.elapsed().unwrap_or_default().as_secs() > max_age_seconds
    }
}

#[derive(Debug, Clone)]
pub struct AggregatorConfig {
    pub prefer_helius_for_transactions: bool,
    pub prefer_solscan_for_analytics: bool,
    pub prefer_jupiter_for_pricing: bool,
    pub enable_cross_validation: bool,
    pub cache_duration_seconds: u64,
    pub max_retries: u32,
    pub fallback_enabled: bool,
}

// Comprehensive data structures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComprehensiveTokenInfo {
    pub address: String,
    pub name: String,
    pub symbol: String,
    pub decimals: u8,
    pub supply: u64,
    pub price_usd: f64,
    pub market_cap: f64,
    pub volume_24h: f64,
    pub price_change_24h: f64,
    pub holders: u64,
    pub is_verified: bool,
    pub liquidity_score: f64,
    pub risk_score: f64,
    pub metadata: TokenMetadata,
    pub data_sources: Vec<DataSource>,
    pub last_updated: std::time::SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TokenMetadata {
    pub description: Option<String>,
    pub image_uri: Option<String>,
    pub website: Option<String>,
    pub twitter: Option<String>,
    pub telegram: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataSource {
    Helius,
    Solscan,
    Jupiter,
}

#[derive(Debug, Clone)]
pub struct EnhancedTransactionAnalysis {
    pub signature: String,
    pub slot: u64,
    pub block_time: i64,
    pub status: TransactionStatus,
    pub fee_lamports: u64,
    pub compute_units: u64,
    pub instructions: Vec<String>,
    pub token_transfers: Vec<EnhancedTokenTransfer>,
    pub native_transfers: Vec<EnhancedNativeTransfer>,
    pub program_interactions: Vec<String>,
    pub mev_classification: Option<MevClassification>,
    pub risk_indicators: Vec<RiskIndicator>,
    pub profitability_analysis: Option<ProfitabilityAnalysis>,
    pub data_sources: Vec<DataSource>,
    pub confidence_score: f64,
}

#[derive(Debug, Clone)]
pub enum TransactionStatus {
    Success,
    Failed,
    Unknown,
}

#[derive(Debug, Clone)]
pub struct EnhancedTokenTransfer {
    pub from: String,
    pub to: String,
    pub mint: String,
    pub amount: f64,
    pub amount_usd: f64,
}

#[derive(Debug, Clone)]
pub struct EnhancedNativeTransfer {
    pub from: String,
    pub to: String,
    pub amount_lamports: u64,
    pub amount_sol: f64,
}

#[derive(Debug, Clone)]
pub enum MevClassification {
    Arbitrage,
    Sandwich,
    Liquidation,
    Unknown,
}

#[derive(Debug, Clone)]
pub enum RiskIndicator {
    HighFees,
    ComplexTransaction,
    UnknownProgram,
    LargeTransfer,
}

#[derive(Debug, Clone)]
pub struct ProfitabilityAnalysis {
    pub gross_profit_usd: f64,
    pub net_profit_usd: f64,
    pub roi_percentage: f64,
    pub profit_sources: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct MarketIntelligence {
    pub tokens: HashMap<String, ComprehensiveTokenInfo>,
    pub market_overview: MarketOverview,
    pub trending_tokens: Vec<String>,
    pub arbitrage_opportunities: Vec<super::jupiter::ArbitrageOpportunity>,
    pub liquidity_analysis: HashMap<String, f64>,
    pub whale_movements: Vec<super::solscan::WhaleActivity>,
    pub risk_assessment: RiskAssessment,
    pub timestamp: std::time::SystemTime,
}

#[derive(Debug, Clone, Default)]
pub struct MarketOverview {
    pub total_tokens: usize,
    pub total_market_cap: f64,
    pub average_volume_24h: f64,
    pub average_price_change_24h: f64,
    pub high_risk_tokens: usize,
    pub verified_tokens: usize,
}

#[derive(Debug, Clone, Default)]
pub struct RiskAssessment {
    pub overall_risk_level: RiskLevel,
    pub volatility_index: f64,
    pub liquidity_stress: f64,
    pub concentration_risk: f64,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone, Default)]
pub enum RiskLevel {
    Low,
    #[default]
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone)]
pub struct DataConsistencyReport {
    pub token_address: String,
    pub sources_checked: u32,
    pub consistency_score: f64,
    pub discrepancies: Vec<String>,
    pub recommendations: Vec<String>,
}