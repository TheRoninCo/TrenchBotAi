use std::sync::Arc;
use std::time::{Duration, Instant};
use trenchbot_dex::infrastructure::{
    solana_rpc::*,
    performance_monitor::{TrenchBotPerformanceMonitor, MonitorConfig},
};

/// 🎯 BLOCKCHAIN PERFORMANCE DEMONSTRATION
/// 
/// This example showcases the ultra-fast blockchain infrastructure
/// components in action, demonstrating microsecond-level performance
/// capabilities for high-frequency trading operations.

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();
    
    println!("🔥 TRENCHBOT BLOCKCHAIN PERFORMANCE DEMO");
    println!("=========================================");
    
    // === DEMO 1: High-Frequency Transaction Queue ===
    demo_high_frequency_queue().await?;
    
    // === DEMO 2: Memory-Mapped Buffer Performance ===
    demo_memory_mapped_buffer().await?;
    
    // === DEMO 3: SIMD Signature Verification ===
    demo_simd_signature_verification().await?;
    
    // === DEMO 4: Connection Pool with Failover ===
    demo_connection_pool_failover().await?;
    
    // === DEMO 5: End-to-End Performance Monitoring ===
    demo_performance_monitoring().await?;
    
    println!("\n✅ ALL DEMOS COMPLETED SUCCESSFULLY!");
    println!("🚀 TRENCHBOT BLOCKCHAIN INFRASTRUCTURE IS BATTLE-READY!");
    
    Ok(())
}

async fn demo_high_frequency_queue() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🎯 DEMO 1: High-Frequency Transaction Queue");
    println!("--------------------------------------------");
    
    let mut queue = HighFrequencyTransactionQueue::new(1024);
    let start_time = Instant::now();
    
    // Fill the queue with transactions
    let mut successful_pushes = 0;
    for i in 0..1000 {
        let tx = StreamedTransaction {
            signature: format!("demo_tx_{}", i),
            slot: i,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_micros() as u64,
            accounts: vec![format!("account_{}", i)],
            instruction_data: vec![i as u8; 64],
            compute_units: Some(200_000),
            fee: 5000,
        };
        
        if queue.try_push(tx).is_ok() {
            successful_pushes += 1;
        }
    }
    
    // Drain the queue
    let mut successful_pops = 0;
    while queue.try_pop().is_some() {
        successful_pops += 1;
    }
    
    let total_time = start_time.elapsed();
    let metrics = queue.get_metrics();
    
    println!("  ✅ Queue operations completed in {:?}", total_time);
    println!("  📊 Successful pushes: {}", successful_pushes);
    println!("  📊 Successful pops: {}", successful_pops);
    println!("  📊 Total processed: {}", metrics.total_processed);
    println!("  📊 Average latency: {}μs", metrics.avg_latency_us);
    println!("  📊 Max latency: {}μs", metrics.max_latency_us);
    
    Ok(())
}

async fn demo_memory_mapped_buffer() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n⚡ DEMO 2: Memory-Mapped Buffer Performance");
    println!("-------------------------------------------");
    
    let mut buffer = TransactionBuffer::new(1024 * 1024)?; // 1MB buffer
    let start_time = Instant::now();
    let write_size = 256;
    let mut total_written = 0;
    let mut successful_writes = 0;
    
    // Perform rapid writes
    for i in 0..1000 {
        unsafe {
            if let Some(_slice) = buffer.get_write_slice(write_size) {
                // Simulate writing transaction data
                buffer.commit_write(write_size);
                total_written += write_size;
                successful_writes += 1;
            } else {
                break; // Buffer full
            }
        }
    }
    
    let total_time = start_time.elapsed();
    let throughput_mb_s = (total_written as f64 / 1024.0 / 1024.0) / total_time.as_secs_f64();
    
    println!("  ✅ Buffer operations completed in {:?}", total_time);
    println!("  📊 Successful writes: {}", successful_writes);
    println!("  📊 Total bytes written: {} KB", total_written / 1024);
    println!("  📊 Throughput: {:.2} MB/s", throughput_mb_s);
    println!("  📊 Available space: {} KB", buffer.available_space() / 1024);
    
    Ok(())
}

async fn demo_simd_signature_verification() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n⚔️  DEMO 3: SIMD Signature Verification");
    println!("---------------------------------------");
    
    let verifier = SIMDSignatureVerifier::new(32);
    let start_time = Instant::now();
    
    // Create mock signatures for testing
    let mut signatures = Vec::new();
    for i in 0..100 {
        let public_key = vec![i as u8; 32];
        let message = format!("transaction_message_{}", i).into_bytes();
        let signature = vec![(i % 256) as u8; 64];
        signatures.push((public_key, message, signature));
    }
    
    // Perform batch verification
    let verification_start = Instant::now();
    let results = verifier.verify_signatures_batch(&signatures);
    let verification_time = verification_start.elapsed();
    
    let valid_signatures = results.iter().filter(|&&valid| valid).count();
    let stats = verifier.get_verification_stats();
    
    println!("  ✅ Signature verification completed in {:?}", verification_time);
    println!("  📊 Signatures verified: {}", signatures.len());
    println!("  📊 Valid signatures: {}", valid_signatures);
    println!("  📊 Verification rate: {:.0} signatures/sec", stats.signatures_per_second);
    println!("  📊 Average verification time: {}μs", stats.avg_signature_verification_us);
    println!("  📊 SIMD operations: {}", stats.simd_ops_count);
    
    Ok(())
}

async fn demo_connection_pool_failover() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🛡️  DEMO 4: Connection Pool with Failover");
    println!("------------------------------------------");
    
    // Create pool with mix of valid and invalid URLs for testing
    let primary_urls = vec![
        "http://invalid-test-url-1.com".to_string(),
        "https://api.devnet.solana.com".to_string(), // This should work
        "http://invalid-test-url-2.com".to_string(),
    ];
    let backup_urls = vec![
        "http://invalid-backup.com".to_string(),
        "https://api.testnet.solana.com".to_string(), // This might work
    ];
    
    let pool = Arc::new(SolanaConnectionPool::new(primary_urls, backup_urls));
    let _health_handle = pool.start_health_monitoring();
    
    let start_time = Instant::now();
    let mut successful_connections = 0;
    let mut failed_connections = 0;
    
    // Test rapid connection requests
    for i in 0..10 {
        match tokio::time::timeout(Duration::from_secs(2), pool.get_client()).await {
            Ok(Ok(_client)) => {
                successful_connections += 1;
                println!("  🔗 Connection {} successful", i + 1);
            }
            Ok(Err(_)) | Err(_) => {
                failed_connections += 1;
                println!("  ❌ Connection {} failed", i + 1);
            }
        }
        
        // Small delay between connection attempts
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    
    let total_time = start_time.elapsed();
    let pool_stats = pool.get_pool_stats();
    
    println!("  ✅ Connection pool test completed in {:?}", total_time);
    println!("  📊 Successful connections: {}", successful_connections);
    println!("  📊 Failed connections: {}", failed_connections);
    println!("  📊 Total requests: {}", pool_stats.total_requests);
    println!("  📊 Failover events: {}", pool_stats.failover_events);
    println!("  📊 Average response time: {}μs", pool_stats.avg_response_time_us);
    
    Ok(())
}

async fn demo_performance_monitoring() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📊 DEMO 5: End-to-End Performance Monitoring");
    println!("---------------------------------------------");
    
    let monitor = TrenchBotPerformanceMonitor::new(MonitorConfig {
        latency_alert_threshold_us: 10_000, // 10ms
        throughput_alert_threshold: 100,     // 100 TPS
        ..Default::default()
    });
    
    // Start monitoring
    monitor.start_monitoring().await;
    tokio::time::sleep(Duration::from_millis(100)).await; // Let it initialize
    
    // Simulate some activity
    let start_time = Instant::now();
    
    // Create mock components for metrics collection
    let streaming_stats = StreamingStats {
        pending_transactions: 50,
        current_slot: 12345,
        active_subscriptions: 3,
    };
    
    let queue_stats = QueueStats {
        total_processed: 1000,
        total_dropped: 5,
        max_latency_us: 2500,
        avg_latency_us: 800,
        available_slots: 500,
    };
    
    let pool_stats = PoolStats {
        total_requests: 200,
        failed_requests: 2,
        failover_events: 1,
        avg_response_time_us: 1500,
        primary_nodes_count: 2,
        backup_nodes_count: 1,
    };
    
    let verification_stats = VerificationStats {
        total_signatures_verified: 500,
        total_verification_time_us: 250_000,
        invalid_signatures: 1,
        batch_verifications: 50,
        simd_ops_count: 25,
        avg_signature_verification_us: 500,
        signatures_per_second: 2000,
    };
    
    let executor_stats = ExecutorStats {
        total_executions: 150,
        successful_executions: 148,
        failed_executions: 2,
        total_execution_time_us: 450_000,
        queue_wait_time_us: 50_000,
        signature_verification_time_us: 75_000,
        network_latency_us: 300_000,
        avg_execution_time_us: 3000,
        success_rate: 98,
        transactions_per_second: 333,
    };
    
    // Record metrics
    monitor.record_metrics(
        &streaming_stats,
        &queue_stats,
        &pool_stats,
        &verification_stats,
        &executor_stats,
    );
    
    // Let monitoring process the data
    tokio::time::sleep(Duration::from_millis(500)).await;
    
    // Get performance report
    let report = monitor.get_performance_report();
    
    monitor.stop_monitoring();
    
    println!("  ✅ Performance monitoring completed in {:?}", start_time.elapsed());
    println!("  📊 System health score: {}", report.system_health.overall_health_score);
    println!("  📊 Total measurements: {}", report.latency_stats.total_measurements);
    println!("  📊 End-to-end latency: {}μs", report.latency_stats.end_to_end_latency_us);
    println!("  📊 Network latency: {}μs", report.latency_stats.network_latency_us);
    println!("  📊 Queue latency: {}μs", report.latency_stats.queue_latency_us);
    println!("  📊 Transactions/sec: {}", report.throughput_stats.transactions_per_second);
    println!("  📊 Active alerts: {}", report.active_alerts.len());
    println!("  📊 Monitoring uptime: {}s", report.uptime_seconds);
    
    // Print any alerts
    for alert in &report.active_alerts {
        println!("  🚨 Alert: {} - {}", alert.component, alert.message);
    }
    
    Ok(())
}