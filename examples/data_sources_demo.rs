use anyhow::Result;
use tracing::{info, warn};
use std::env;

use trench_bot_ai::infrastructure::{
    helius::HeliusClient,
    solscan::SolscanClient,
    jupiter::JupiterClient,
    data_aggregator::DataAggregator,
};

/// **COMPREHENSIVE DATA SOURCES DEMO**
/// Showcase integration with Helius, Solscan, and Jupiter APIs
#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::init();
    
    // Load environment variables
    dotenv::dotenv().ok();
    
    info!("🚀 TrenchBotAi Data Sources Integration Demo");
    info!("================================================");
    
    // Test popular Solana tokens
    let test_tokens = vec![
        "So11111111111111111111111111111111111111112".to_string(), // SOL (wrapped)
        "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v".to_string(), // USDC
        "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB".to_string(), // USDT
    ];
    
    let test_wallet = "9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM"; // Random whale wallet
    
    // Demo 1: Individual API clients
    demo_individual_clients(&test_tokens, test_wallet).await?;
    
    // Demo 2: Aggregated data analysis
    demo_data_aggregator(&test_tokens).await?;
    
    // Demo 3: Market intelligence
    demo_market_intelligence(&test_tokens).await?;
    
    info!("✅ All demos completed successfully!");
    
    Ok(())
}

/// **DEMO 1: Individual API Clients**
async fn demo_individual_clients(test_tokens: &[String], test_wallet: &str) -> Result<()> {
    info!("\n🔥 DEMO 1: Individual API Clients");
    info!("==================================");
    
    // Helius Demo
    info!("\n🚀 Testing Helius API...");
    match HeliusClient::new() {
        Ok(helius) => {
            // Get asset information
            if let Ok(asset) = helius.get_asset(&test_tokens[0]).await {
                info!("  📋 Asset: {} ({}) - {} decimals", asset.name, asset.symbol, asset.decimals);
            }
            
            // Get transaction signatures
            if let Ok(signatures) = helius.get_signatures_for_address(test_wallet, Some(5), None, None).await {
                info!("  ✍️  Found {} recent transaction signatures", signatures.len());
                
                // Analyze first transaction
                if let Some(sig) = signatures.first() {
                    if let Ok(parsed_txs) = helius.get_parsed_transactions(vec![sig.signature.clone()]).await {
                        if let Some(tx) = parsed_txs.first() {
                            info!("  🔍 Transaction analysis: {} lamports fee, {} transfers", 
                                  tx.fee.unwrap_or(0),
                                  tx.token_transfers.as_ref().map(|t| t.len()).unwrap_or(0));
                        }
                    }
                }
            }
            
            // Get priority fee estimate
            if let Ok(fees) = helius.get_priority_fee_estimate(vec![]).await {
                info!("  ⛽ Priority fees - Low: {} | Medium: {} | High: {} lamports", 
                      fees.per_compute_unit.low, fees.per_compute_unit.medium, fees.per_compute_unit.high);
            }
        },
        Err(e) => warn!("❌ Helius client failed: {} (check HELIUS_API_KEY)", e),
    }
    
    // Solscan Demo
    info!("\n🔍 Testing Solscan API...");
    match SolscanClient::new() {
        Ok(solscan) => {
            // Get account info
            if let Ok(account) = solscan.get_account_info(test_wallet).await {
                info!("  👤 Account: {:.3} SOL balance", account.lamports as f64 / 1e9);
            }
            
            // Get token balances
            if let Ok(balances) = solscan.get_token_balances(test_wallet).await {
                info!("  💰 Token balances: {} different tokens", balances.len());
                
                for balance in balances.iter().take(3) {
                    if balance.amount > 0.0 {
                        info!("    - {}: {:.2} {} (${:.2})", 
                              balance.token_symbol.as_ref().unwrap_or(&"Unknown".to_string()),
                              balance.amount,
                              balance.token_symbol.as_ref().unwrap_or(&"".to_string()),
                              balance.amount_usd.unwrap_or(0.0));
                    }
                }
            }
            
            // Get token info
            if let Ok(token_info) = solscan.get_token_info(&test_tokens[1]).await {
                info!("  🪙 Token: {} ({}) - {} holders, ${:.2}M market cap", 
                      token_info.name, token_info.symbol,
                      token_info.holder.unwrap_or(0),
                      token_info.market_cap.unwrap_or(0.0) / 1_000_000.0);
            }
            
            // Analyze whale activity
            if let Ok(whale_activities) = solscan.analyze_whale_activity(&test_tokens[0], 100.0).await {
                info!("  🐋 Whale activities: {} large movements detected", whale_activities.len());
                
                for activity in whale_activities.iter().take(2) {
                    info!("    - {:.1} tokens (${:.0}) from {} to {}", 
                          activity.amount, activity.amount_usd,
                          &activity.from[..8], &activity.to[..8]);
                }
            }
        },
        Err(e) => warn!("❌ Solscan client failed: {} (check SOLSCAN_API_KEY)", e),
    }
    
    // Jupiter Demo
    info!("\n🪐 Testing Jupiter API (FREE)...");
    if let Ok(jupiter) = JupiterClient::new() {
        // Get token prices
        if let Ok(prices) = jupiter.get_price(test_tokens.clone()).await {
            info!("  💲 Current prices:");
            for (token, price_data) in prices.data.iter().take(3) {
                info!("    - {}: ${:.6}", &token[..8], price_data.price);
            }
        }
        
        // Get swap quote
        if let Ok(quote) = jupiter.get_quote(&trench_bot_ai::infrastructure::jupiter::QuoteRequest {
            input_mint: test_tokens[0].clone(),  // SOL
            output_mint: test_tokens[1].clone(), // USDC
            amount: 1_000_000_000, // 1 SOL
            slippage_bps: Some(50), // 0.5%
            only_direct_routes: false,
            max_accounts: None,
        }).await {
            let output_amount = quote.out_amount.parse::<u64>().unwrap_or(0) as f64 / 1_000_000.0; // USDC has 6 decimals
            let price_impact = quote.price_impact_pct.parse::<f64>().unwrap_or(0.0);
            
            info!("  💱 Swap quote: 1 SOL → {:.2} USDC (impact: {:.3}%)", 
                  output_amount, price_impact);
            info!("    Routes available: {}", quote.route_plan.len());
        }
        
        // Find arbitrage opportunities
        let arb_pairs = vec![
            (test_tokens[0].clone(), test_tokens[1].clone()), // SOL/USDC
        ];
        
        if let Ok(opportunities) = jupiter.find_arbitrage_opportunities(arb_pairs, 10).await {
            info!("  ⚡ Arbitrage opportunities: {}", opportunities.len());
            
            for opp in opportunities.iter().take(2) {
                info!("    - {:.1} bps profit ({:?})", opp.profit_bps, opp.direction);
            }
        }
        
        // Get supported tokens
        if let Ok(token_list) = jupiter.get_token_list().await {
            info!("  📋 Supported tokens: {} total", token_list.len());
            
            let verified_count = token_list.iter().filter(|t| t.tags.contains(&"verified".to_string())).count();
            info!("    - Verified tokens: {}", verified_count);
        }
    }
    
    Ok(())
}

/// **DEMO 2: Data Aggregator**
async fn demo_data_aggregator(test_tokens: &[String]) -> Result<()> {
    info!("\n🔄 DEMO 2: Multi-Source Data Aggregation");
    info!("==========================================");
    
    if let Ok(aggregator) = DataAggregator::new().await {
        // Get comprehensive token information
        if let Ok(token_info) = aggregator.get_comprehensive_token_info(&test_tokens[1]).await {
            info!("  🪙 Comprehensive Token Analysis:");
            info!("    Name: {} ({})", token_info.name, token_info.symbol);
            info!("    Price: ${:.6}", token_info.price_usd);
            info!("    Market Cap: ${:.2}M", token_info.market_cap / 1_000_000.0);
            info!("    Holders: {}", token_info.holders);
            info!("    Liquidity Score: {:.2}", token_info.liquidity_score);
            info!("    Risk Score: {:.2}", token_info.risk_score);
            info!("    Verified: {}", token_info.is_verified);
            info!("    Data Sources: {:?}", token_info.data_sources);
        }
        
        // Validate data consistency
        if let Ok(consistency) = aggregator.validate_data_consistency(&test_tokens[0]).await {
            info!("  🔍 Data Consistency Report:");
            info!("    Sources checked: {}", consistency.sources_checked);
            info!("    Consistency score: {:.1}%", consistency.consistency_score * 100.0);
            info!("    Discrepancies: {}", consistency.discrepancies.len());
            
            if !consistency.recommendations.is_empty() {
                info!("    Recommendations: {:?}", consistency.recommendations);
            }
        }
    }
    
    Ok(())
}

/// **DEMO 3: Market Intelligence**
async fn demo_market_intelligence(test_tokens: &[String]) -> Result<()> {
    info!("\n📊 DEMO 3: Market Intelligence Analysis");
    info!("========================================");
    
    if let Ok(aggregator) = DataAggregator::new().await {
        if let Ok(intelligence) = aggregator.get_market_intelligence(test_tokens.clone()).await {
            info!("  📈 Market Overview:");
            info!("    Total tokens analyzed: {}", intelligence.market_overview.total_tokens);
            info!("    Total market cap: ${:.2}M", intelligence.market_overview.total_market_cap / 1_000_000.0);
            info!("    Average 24h volume: ${:.2}M", intelligence.market_overview.average_volume_24h / 1_000_000.0);
            info!("    Average price change: {:.2}%", intelligence.market_overview.average_price_change_24h);
            info!("    High-risk tokens: {}", intelligence.market_overview.high_risk_tokens);
            info!("    Verified tokens: {}", intelligence.market_overview.verified_tokens);
            
            info!("  ⚡ Arbitrage Opportunities:");
            for (i, opp) in intelligence.arbitrage_opportunities.iter().enumerate().take(3) {
                info!("    {}. {:.1} bps profit ({:?}) - {} routes", 
                      i + 1, opp.profit_bps, opp.direction, opp.route_count);
            }
            
            info!("  🐋 Recent Whale Movements:");
            for (i, movement) in intelligence.whale_movements.iter().enumerate().take(3) {
                info!("    {}. ${:.0} movement by {}...{}", 
                      i + 1, movement.amount_usd, &movement.from[..6], &movement.from[movement.from.len()-6..]);
            }
            
            info!("  🛡️ Risk Assessment:");
            info!("    Overall risk: {:?}", intelligence.risk_assessment.overall_risk_level);
            info!("    Volatility index: {:.2}", intelligence.risk_assessment.volatility_index);
            info!("    Liquidity stress: {:.2}", intelligence.risk_assessment.liquidity_stress);
            info!("    Recommendations: {:?}", intelligence.risk_assessment.recommendations);
        }
    }
    
    Ok(())
}

/// **ENVIRONMENT SETUP GUIDANCE**
#[allow(dead_code)]
fn print_setup_instructions() {
    println!("\n📋 SETUP INSTRUCTIONS:");
    println!("======================");
    println!("1. Copy .env.example to .env");
    println!("2. Add your API keys:");
    println!("   HELIUS_API_KEY=your_helius_dev_key");
    println!("   SOLSCAN_API_KEY=your_solscan_key");
    println!("3. Jupiter API is FREE (no key required)");
    println!("4. Run: cargo run --example data_sources_demo");
    println!("\n🔗 Get API Keys:");
    println!("  Helius: https://dev.helius.xyz/");
    println!("  Solscan: https://pro-api.solscan.io/");
    println!("  Jupiter: Free tier, no signup required");
}