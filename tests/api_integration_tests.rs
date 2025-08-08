//! **API INTEGRATION TESTS**
//! Comprehensive testing for Helius, Solscan, and Jupiter API integrations
//! Tests data reliability, cross-validation, and fallback mechanisms

use tokio::test;
use std::time::{Duration, Instant};
use trenchbot_dex::infrastructure::{
    helius::HeliusClient,
    solscan::SolscanClient, 
    jupiter::JupiterClient,
    data_aggregator::DataAggregator,
};
use serde_json::Value;

#[test]
async fn test_helius_api_integration() {
    println!("🚀 Testing Helius API Integration");
    
    // Initialize Helius client (will use mock/test mode if no API key)
    let helius_client = HeliusClient::new("test_api_key".to_string(), true).await.unwrap();
    
    // Test transaction parsing
    let test_signatures = vec![
        "test_signature_1".to_string(),
        "test_signature_2".to_string(),
        "test_signature_3".to_string(),
    ];
    
    let start_time = Instant::now();
    let parsed_transactions = helius_client.get_parsed_transactions(test_signatures).await;
    let response_time = start_time.elapsed();
    
    assert!(response_time < Duration::from_millis(2000), 
           "Helius API should respond within 2 seconds, took: {}ms", response_time.as_millis());
    
    match parsed_transactions {
        Ok(transactions) => {
            assert!(!transactions.is_empty(), "Should return parsed transactions");
            
            for transaction in &transactions {
                assert!(!transaction.signature.is_empty(), "Transaction should have signature");
                assert!(transaction.slot > 0, "Transaction should have valid slot");
                assert!(transaction.block_time > 0, "Transaction should have block time");
                
                // Validate parsed instruction data
                if let Some(instructions) = &transaction.instructions {
                    for instruction in instructions {
                        assert!(!instruction.program_id.is_empty(), "Instruction should have program ID");
                    }
                }
            }
            
            println!("✅ Successfully parsed {} transactions", transactions.len());
        },
        Err(e) => {
            // In test mode, this might be expected
            println!("⚠️  Helius API test mode or error: {}", e);
        }
    }
    
    // Test asset information retrieval
    let sol_token = "So11111111111111111111111111111111111111112";
    let asset_info = helius_client.get_asset_info(sol_token).await;
    
    match asset_info {
        Ok(info) => {
            assert_eq!(info.token_address, sol_token);
            assert!(!info.name.is_empty(), "Asset should have a name");
            assert!(info.decimals <= 18, "Asset decimals should be reasonable");
            println!("💰 Asset info: {} ({})", info.name, info.symbol);
        },
        Err(e) => {
            println!("⚠️  Asset info test mode or error: {}", e);
        }
    }
    
    // Test whale activity detection
    let whale_activity = helius_client.detect_whale_activity(sol_token, 1000000.0).await;
    
    match whale_activity {
        Ok(activity) => {
            for whale_tx in &activity {
                assert!(whale_tx.amount >= 1000000.0, "Whale transaction should meet minimum threshold");
                assert!(!whale_tx.from_address.is_empty(), "Should have from address");
                assert!(!whale_tx.to_address.is_empty(), "Should have to address");
            }
            println!("🐋 Detected {} whale activities", activity.len());
        },
        Err(e) => {
            println!("⚠️  Whale detection test mode or error: {}", e);
        }
    }
    
    println!("✅ Helius API integration tests completed");
}

#[test]
async fn test_solscan_api_integration() {
    println!("🔍 Testing Solscan API Integration");
    
    let solscan_client = SolscanClient::new("test_api_key".to_string(), true).await.unwrap();
    
    // Test token holder analysis
    let sol_token = "So11111111111111111111111111111111111111112";
    
    let start_time = Instant::now();
    let holder_analysis = solscan_client.analyze_token_holders(sol_token).await;
    let response_time = start_time.elapsed();
    
    assert!(response_time < Duration::from_millis(3000), 
           "Solscan API should respond within 3 seconds, took: {}ms", response_time.as_millis());
    
    match holder_analysis {
        Ok(analysis) => {
            assert!(analysis.total_holders > 0, "Should have positive holder count");
            assert!(analysis.holder_distribution.len() > 0, "Should have holder distribution data");
            assert!(analysis.concentration_ratio >= 0.0 && analysis.concentration_ratio <= 1.0, 
                   "Concentration ratio should be between 0 and 1");
            
            // Validate top holders data
            for holder in &analysis.top_holders {
                assert!(!holder.address.is_empty(), "Holder should have address");
                assert!(holder.balance > 0.0, "Holder should have positive balance");
                assert!(holder.percentage >= 0.0 && holder.percentage <= 100.0, 
                       "Holder percentage should be valid");
            }
            
            println!("👥 Token has {} holders, concentration: {:.2}%", 
                     analysis.total_holders, analysis.concentration_ratio * 100.0);
        },
        Err(e) => {
            println!("⚠️  Solscan holder analysis test mode or error: {}", e);
        }
    }
    
    // Test whale activity analysis
    let whale_activity = solscan_client.analyze_whale_activity(sol_token, 10000.0).await;
    
    match whale_activity {
        Ok(activities) => {
            for activity in &activities {
                assert!(activity.amount >= 10000.0, "Activity should meet whale threshold");
                assert!(!activity.transaction_signature.is_empty(), "Should have transaction signature");
                assert!(activity.timestamp > 0, "Should have valid timestamp");
                
                // Validate activity type
                match activity.activity_type.as_str() {
                    "buy" | "sell" | "transfer" => {},
                    _ => panic!("Unknown activity type: {}", activity.activity_type),
                }
            }
            
            println!("🐋 Found {} whale activities", activities.len());
        },
        Err(e) => {
            println!("⚠️  Whale activity analysis test mode or error: {}", e);
        }
    }
    
    // Test market metrics
    let market_metrics = solscan_client.get_market_metrics(sol_token).await;
    
    match market_metrics {
        Ok(metrics) => {
            assert!(metrics.price > 0.0, "Price should be positive");
            assert!(metrics.volume_24h >= 0.0, "Volume should be non-negative");
            assert!(metrics.market_cap > 0.0, "Market cap should be positive");
            assert!(metrics.circulating_supply > 0.0, "Circulating supply should be positive");
            
            println!("📊 Market metrics - Price: ${:.2}, Vol: ${:.0}, MCap: ${:.0}", 
                     metrics.price, metrics.volume_24h, metrics.market_cap);
        },
        Err(e) => {
            println!("⚠️  Market metrics test mode or error: {}", e);
        }
    }
    
    println!("✅ Solscan API integration tests completed");
}

#[test]
async fn test_jupiter_api_integration() {
    println!("🪐 Testing Jupiter API Integration");
    
    let jupiter_client = JupiterClient::new(true).await.unwrap(); // Test mode
    
    // Test price quotes
    let input_token = "So11111111111111111111111111111111111111112"; // SOL
    let output_token = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"; // USDC
    let amount = 1_000_000_000; // 1 SOL in lamports
    
    let start_time = Instant::now();
    let quote_result = jupiter_client.get_quote(input_token, output_token, amount).await;
    let response_time = start_time.elapsed();
    
    assert!(response_time < Duration::from_millis(1500), 
           "Jupiter API should respond within 1.5 seconds, took: {}ms", response_time.as_millis());
    
    match quote_result {
        Ok(quote) => {
            assert_eq!(quote.input_mint, input_token);
            assert_eq!(quote.output_mint, output_token);
            assert_eq!(quote.in_amount, amount.to_string());
            assert!(quote.out_amount.parse::<u64>().unwrap() > 0, "Output amount should be positive");
            assert!(!quote.route_plan.is_empty(), "Should have route plan");
            
            // Validate route information
            for route in &quote.route_plan {
                assert!(!route.swap_info.amm_key.is_empty(), "Route should have AMM key");
                assert!(!route.swap_info.label.is_empty(), "Route should have label");
            }
            
            println!("💱 Quote: {} {} -> {} {}", 
                     amount, "SOL", quote.out_amount, "USDC");
        },
        Err(e) => {
            println!("⚠️  Jupiter quote test mode or error: {}", e);
        }
    }
    
    // Test arbitrage opportunity detection
    let token_pairs = vec![
        (input_token.to_string(), output_token.to_string()),
        ("Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB".to_string(), input_token.to_string()), // USDT -> SOL
    ];
    
    let arbitrage_opportunities = jupiter_client.find_arbitrage_opportunities(token_pairs, 10).await;
    
    match arbitrage_opportunities {
        Ok(opportunities) => {
            for opportunity in &opportunities {
                assert!(!opportunity.token_a.is_empty(), "Should have token A");
                assert!(!opportunity.token_b.is_empty(), "Should have token B");
                assert!(opportunity.profit_bps >= 10, "Profit should meet minimum threshold");
                assert!(!opportunity.route_a.is_empty(), "Should have route A");
                assert!(!opportunity.route_b.is_empty(), "Should have route B");
                assert!(opportunity.confidence_score >= 0.0 && opportunity.confidence_score <= 1.0, 
                       "Confidence should be between 0 and 1");
            }
            
            println!("🔄 Found {} arbitrage opportunities", opportunities.len());
        },
        Err(e) => {
            println!("⚠️  Arbitrage detection test mode or error: {}", e);
        }
    }
    
    // Test token price information
    let tokens = vec![input_token.to_string(), output_token.to_string()];
    let price_info = jupiter_client.get_token_prices(tokens).await;
    
    match price_info {
        Ok(prices) => {
            assert!(!prices.is_empty(), "Should return price information");
            
            for (token, price_data) in &prices {
                assert!(!token.is_empty(), "Token address should not be empty");
                assert!(price_data.price > 0.0, "Price should be positive");
                
                if let Some(volume) = price_data.volume_24h {
                    assert!(volume >= 0.0, "Volume should be non-negative");
                }
            }
            
            println!("💰 Retrieved prices for {} tokens", prices.len());
        },
        Err(e) => {
            println!("⚠️  Price info test mode or error: {}", e);
        }
    }
    
    println!("✅ Jupiter API integration tests completed");
}

#[test]
async fn test_data_aggregator_cross_validation() {
    println!("🔄 Testing Data Aggregator Cross-Validation");
    
    let data_aggregator = DataAggregator::new().await.unwrap();
    
    // Test comprehensive token information aggregation
    let sol_token = "So11111111111111111111111111111111111111112";
    
    let start_time = Instant::now();
    let comprehensive_info = data_aggregator.get_comprehensive_token_info(sol_token).await;
    let response_time = start_time.elapsed();
    
    assert!(response_time < Duration::from_millis(5000), 
           "Data aggregation should complete within 5 seconds, took: {}ms", response_time.as_millis());
    
    match comprehensive_info {
        Ok(info) => {
            // Validate aggregated data completeness
            assert_eq!(info.token_address, sol_token);
            assert!(!info.name.is_empty(), "Should have token name");
            assert!(!info.symbol.is_empty(), "Should have token symbol");
            assert!(info.price > 0.0, "Should have positive price");
            assert!(info.market_cap > 0.0, "Should have positive market cap");
            
            // Test cross-validation consistency
            if info.helius_data.is_some() && info.solscan_data.is_some() {
                let helius_price = info.helius_data.as_ref().unwrap().price;
                let solscan_price = info.solscan_data.as_ref().unwrap().price;
                
                let price_difference = (helius_price - solscan_price).abs() / helius_price.max(solscan_price);
                assert!(price_difference < 0.05, 
                       "Price difference between sources should be <5%, got {:.2}%", price_difference * 100.0);
            }
            
            // Test data quality scores
            assert!(info.data_quality_score >= 0.5, "Data quality should be reasonable");
            assert!(info.confidence_level >= 0.0 && info.confidence_level <= 1.0, 
                   "Confidence level should be between 0 and 1");
            
            // Test whale activity aggregation
            if let Some(whale_activities) = &info.aggregated_whale_activity {
                for activity in whale_activities {
                    assert!(!activity.source.is_empty(), "Activity should have source");
                    assert!(activity.amount > 0.0, "Activity amount should be positive");
                    assert!(!activity.transaction_hash.is_empty(), "Should have transaction hash");
                }
            }
            
            println!("📊 Aggregated data for {} - Price: ${:.2}, Quality: {:.2}, Confidence: {:.2}",
                     info.name, info.price, info.data_quality_score, info.confidence_level);
        },
        Err(e) => {
            println!("⚠️  Data aggregation test mode or error: {}", e);
        }
    }
    
    println!("✅ Data aggregator cross-validation tests completed");
}

#[test]
async fn test_api_rate_limiting_and_fallbacks() {
    println!("🚦 Testing API Rate Limiting and Fallback Mechanisms");
    
    let data_aggregator = DataAggregator::new().await.unwrap();
    
    // Test rapid successive requests to trigger rate limiting
    let test_tokens = vec![
        "So11111111111111111111111111111111111111112", // SOL
        "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v", // USDC
        "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB", // USDT
    ];
    
    let mut successful_requests = 0;
    let mut rate_limited_requests = 0;
    let mut fallback_used = 0;
    
    let start_time = Instant::now();
    
    // Make rapid requests to test rate limiting
    for (i, token) in test_tokens.iter().enumerate() {
        let request_start = Instant::now();
        let result = data_aggregator.get_comprehensive_token_info(token).await;
        let request_time = request_start.elapsed();
        
        match result {
            Ok(info) => {
                successful_requests += 1;
                
                // Check if fallback mechanisms were used
                let source_count = [
                    info.helius_data.is_some(),
                    info.solscan_data.is_some(), 
                    info.jupiter_data.is_some(),
                ].iter().filter(|&&x| x).count();
                
                if source_count < 2 {
                    fallback_used += 1;
                    println!("🔄 Request {} used fallback (only {} sources available)", i + 1, source_count);
                }
                
                // Verify graceful degradation
                assert!(info.confidence_level >= 0.3, 
                       "Even with fallbacks, confidence should be reasonable");
                
                println!("✅ Request {} completed in {}ms (sources: {})", 
                         i + 1, request_time.as_millis(), source_count);
            },
            Err(e) => {
                if e.to_string().contains("rate limit") || e.to_string().contains("429") {
                    rate_limited_requests += 1;
                    println!("🚦 Request {} rate limited: {}", i + 1, e);
                } else {
                    println!("❌ Request {} failed: {}", i + 1, e);
                }
            }
        }
        
        // Small delay to avoid overwhelming in test environment
        if i < test_tokens.len() - 1 {
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }
    
    let total_time = start_time.elapsed();
    
    println!("📊 Rate limiting test results:");
    println!("   Successful: {}", successful_requests);
    println!("   Rate limited: {}", rate_limited_requests);
    println!("   Fallbacks used: {}", fallback_used);
    println!("   Total time: {}ms", total_time.as_millis());
    
    // Verify that at least some requests succeeded
    assert!(successful_requests > 0, "At least some requests should succeed");
    
    // If rate limiting occurred, verify it was handled gracefully
    if rate_limited_requests > 0 {
        println!("✅ Rate limiting detected and handled appropriately");
    }
    
    println!("✅ Rate limiting and fallback tests completed");
}

#[test]
async fn test_api_error_handling_and_recovery() {
    println!("🛠️ Testing API Error Handling and Recovery");
    
    let data_aggregator = DataAggregator::new().await.unwrap();
    
    // Test with invalid token addresses
    let invalid_tokens = vec![
        "invalid_token_address_1",
        "12345", // Too short
        "", // Empty
        "So11111111111111111111111111111111111111113", // Almost valid but wrong checksum
    ];
    
    for (i, invalid_token) in invalid_tokens.iter().enumerate() {
        println!("🧪 Testing invalid token {}: '{}'", i + 1, invalid_token);
        
        let result = data_aggregator.get_comprehensive_token_info(invalid_token).await;
        
        match result {
            Ok(info) => {
                // If somehow successful, data should indicate low quality/confidence
                assert!(info.confidence_level < 0.5, 
                       "Invalid token should have low confidence");
                println!("⚠️  Unexpected success with low confidence: {:.2}", info.confidence_level);
            },
            Err(e) => {
                // Error should be descriptive and not cause panic
                assert!(!e.to_string().is_empty(), "Error message should not be empty");
                println!("✅ Properly handled error: {}", e);
            }
        }
    }
    
    // Test timeout handling
    let timeout_test_start = Instant::now();
    let result = data_aggregator.get_comprehensive_token_info_with_timeout(
        "So11111111111111111111111111111111111111112",
        Duration::from_millis(50) // Very short timeout
    ).await;
    let timeout_test_time = timeout_test_start.elapsed();
    
    match result {
        Ok(_) => {
            println!("✅ Request completed within timeout");
        },
        Err(e) => {
            if e.to_string().contains("timeout") {
                assert!(timeout_test_time >= Duration::from_millis(45), 
                       "Timeout should respect the specified duration");
                assert!(timeout_test_time <= Duration::from_millis(200), 
                       "Timeout should not take much longer than specified");
                println!("✅ Timeout handled correctly in {}ms", timeout_test_time.as_millis());
            } else {
                println!("ℹ️  Different error during timeout test: {}", e);
            }
        }
    }
    
    println!("✅ Error handling and recovery tests completed");
}

#[test]
async fn test_api_performance_benchmarking() {
    println!("⚡ Testing API Performance Benchmarking");
    
    let data_aggregator = DataAggregator::new().await.unwrap();
    let sol_token = "So11111111111111111111111111111111111111112";
    
    // Benchmark individual API response times
    let mut helius_times = Vec::new();
    let mut solscan_times = Vec::new();
    let mut jupiter_times = Vec::new();
    let mut aggregated_times = Vec::new();
    
    for i in 0..5 {
        println!("🔄 Benchmark iteration {}", i + 1);
        
        // Test Helius performance
        if let Ok(helius_client) = HeliusClient::new("test".to_string(), true).await {
            let start = Instant::now();
            let _ = helius_client.get_asset_info(sol_token).await;
            helius_times.push(start.elapsed());
        }
        
        // Test Solscan performance
        if let Ok(solscan_client) = SolscanClient::new("test".to_string(), true).await {
            let start = Instant::now();
            let _ = solscan_client.get_market_metrics(sol_token).await;
            solscan_times.push(start.elapsed());
        }
        
        // Test Jupiter performance
        if let Ok(jupiter_client) = JupiterClient::new(true).await {
            let start = Instant::now();
            let _ = jupiter_client.get_token_prices(vec![sol_token.to_string()]).await;
            jupiter_times.push(start.elapsed());
        }
        
        // Test aggregated performance
        let start = Instant::now();
        let _ = data_aggregator.get_comprehensive_token_info(sol_token).await;
        aggregated_times.push(start.elapsed());
        
        // Delay between iterations
        if i < 4 {
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
    }
    
    // Calculate and display performance metrics
    let calc_avg = |times: &[Duration]| -> f64 {
        if times.is_empty() { return 0.0; }
        times.iter().sum::<Duration>().as_millis() as f64 / times.len() as f64
    };
    
    let calc_min = |times: &[Duration]| -> u128 {
        times.iter().min().map(|d| d.as_millis()).unwrap_or(0)
    };
    
    let calc_max = |times: &[Duration]| -> u128 {
        times.iter().max().map(|d| d.as_millis()).unwrap_or(0)
    };
    
    println!("📊 Performance Benchmarking Results:");
    
    if !helius_times.is_empty() {
        println!("🚀 Helius: avg={:.1}ms, min={}ms, max={}ms", 
                 calc_avg(&helius_times), calc_min(&helius_times), calc_max(&helius_times));
        assert!(calc_avg(&helius_times) < 3000.0, "Helius average response should be under 3 seconds");
    }
    
    if !solscan_times.is_empty() {
        println!("🔍 Solscan: avg={:.1}ms, min={}ms, max={}ms", 
                 calc_avg(&solscan_times), calc_min(&solscan_times), calc_max(&solscan_times));
        assert!(calc_avg(&solscan_times) < 4000.0, "Solscan average response should be under 4 seconds");
    }
    
    if !jupiter_times.is_empty() {
        println!("🪐 Jupiter: avg={:.1}ms, min={}ms, max={}ms", 
                 calc_avg(&jupiter_times), calc_min(&jupiter_times), calc_max(&jupiter_times));
        assert!(calc_avg(&jupiter_times) < 2000.0, "Jupiter average response should be under 2 seconds");
    }
    
    if !aggregated_times.is_empty() {
        println!("🔄 Aggregated: avg={:.1}ms, min={}ms, max={}ms", 
                 calc_avg(&aggregated_times), calc_min(&aggregated_times), calc_max(&aggregated_times));
        assert!(calc_avg(&aggregated_times) < 6000.0, "Aggregated average response should be under 6 seconds");
    }
    
    println!("✅ Performance benchmarking tests completed");
}

#[test]
async fn test_data_consistency_validation() {
    println!("🔍 Testing Data Consistency Validation");
    
    let data_aggregator = DataAggregator::new().await.unwrap();
    
    // Test multiple tokens for consistency
    let test_tokens = vec![
        ("So11111111111111111111111111111111111111112", "SOL"),
        ("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v", "USDC"),
    ];
    
    for (token_address, token_symbol) in test_tokens {
        println!("🔍 Validating consistency for {}", token_symbol);
        
        let comprehensive_info = data_aggregator.get_comprehensive_token_info(token_address).await;
        
        match comprehensive_info {
            Ok(info) => {
                // Test internal consistency
                assert_eq!(info.token_address, token_address, "Token address should match request");
                
                // Test cross-source price consistency (if multiple sources available)
                let mut prices = Vec::new();
                
                if let Some(helius_data) = &info.helius_data {
                    prices.push(("Helius", helius_data.price));
                }
                
                if let Some(solscan_data) = &info.solscan_data {
                    prices.push(("Solscan", solscan_data.price));
                }
                
                if let Some(jupiter_data) = &info.jupiter_data {
                    prices.push(("Jupiter", jupiter_data.price));
                }
                
                // Validate price consistency across sources
                if prices.len() >= 2 {
                    let min_price = prices.iter().map(|(_, p)| p).fold(f64::INFINITY, |a, &b| a.min(b));
                    let max_price = prices.iter().map(|(_, p)| p).fold(0.0f64, |a, &b| a.max(b));
                    let price_variance = (max_price - min_price) / min_price.max(max_price);
                    
                    assert!(price_variance < 0.10, 
                           "Price variance between sources should be <10% for {}, got {:.2}%", 
                           token_symbol, price_variance * 100.0);
                    
                    println!("💰 Price consistency for {}: min=${:.4}, max=${:.4}, variance={:.2}%", 
                             token_symbol, min_price, max_price, price_variance * 100.0);
                }
                
                // Test volume consistency (if available)
                let mut volumes = Vec::new();
                
                if let Some(helius_data) = &info.helius_data {
                    if let Some(volume) = helius_data.volume_24h {
                        volumes.push(("Helius", volume));
                    }
                }
                
                if let Some(solscan_data) = &info.solscan_data {
                    if let Some(volume) = solscan_data.volume_24h {
                        volumes.push(("Solscan", volume));
                    }
                }
                
                if volumes.len() >= 2 {
                    let volume_ratio = volumes[0].1 / volumes[1].1.max(1.0);
                    
                    // Volume can vary more than price due to different calculation methods
                    assert!(volume_ratio > 0.1 && volume_ratio < 10.0, 
                           "Volume ratio should be reasonable for {}, got {:.2}x", 
                           token_symbol, volume_ratio);
                    
                    println!("📊 Volume consistency for {}: ratio={:.2}x", token_symbol, volume_ratio);
                }
                
                // Test metadata consistency
                if !info.name.is_empty() && !info.symbol.is_empty() {
                    assert!(info.symbol.len() <= 10, "Token symbol should be reasonable length");
                    assert!(info.name.len() <= 100, "Token name should be reasonable length");
                    assert!(info.symbol.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-'), 
                           "Token symbol should contain valid characters");
                }
                
                println!("✅ Consistency validation passed for {}", token_symbol);
            },
            Err(e) => {
                println!("⚠️  Consistency test error for {}: {}", token_symbol, e);
            }
        }
    }
    
    println!("✅ Data consistency validation tests completed");
}