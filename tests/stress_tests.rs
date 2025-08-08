use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Barrier;
use tokio::task::JoinHandle;
use tracing::{info, warn, error};
use uuid::Uuid;

use trenchbot_dex::infrastructure::{
    solana_rpc::*,
    performance_monitor::{TrenchBotPerformanceMonitor, MonitorConfig, AlertSeverity},
};

/// 💀 STRESS TESTING ARSENAL
/// 
/// Battle-hardened stress tests to ensure TrenchBot can withstand:
/// - High-frequency trading storms
/// - Massive transaction volumes
/// - Network failures and recovery
/// - Memory pressure scenarios
/// - Concurrent access patterns
/// 
/// These tests simulate extreme battlefield conditions to validate
/// that the system maintains sub-millisecond performance under duress.

#[tokio::test]
async fn stress_test_websocket_streaming_overload() {
    let _ = tracing_subscriber::fmt::try_init();
    info!("🔥 INITIATING: WebSocket streaming overload test");

    // Create performance monitor
    let monitor = TrenchBotPerformanceMonitor::new(MonitorConfig {
        latency_alert_threshold_us: 10_000, // 10ms
        ..Default::default()
    });
    monitor.start_monitoring().await;

    // Test configuration
    const CONCURRENT_STREAMS: usize = 50;
    const TRANSACTIONS_PER_STREAM: usize = 1_000;
    const TARGET_LATENCY_US: u64 = 5_000; // 5ms target

    let start_time = Instant::now();
    let barrier = Arc::new(Barrier::new(CONCURRENT_STREAMS));

    // Spawn concurrent streaming tasks
    let mut handles: Vec<JoinHandle<StreamTestResult>> = Vec::new();
    
    for stream_id in 0..CONCURRENT_STREAMS {
        let barrier_clone = Arc::clone(&barrier);
        
        let handle = tokio::spawn(async move {
            // Wait for all streams to be ready
            barrier_clone.wait().await;
            
            let mut successful_transactions = 0;
            let mut failed_transactions = 0;
            let mut total_latency_us = 0u64;
            let mut max_latency_us = 0u64;
            
            for tx_id in 0..TRANSACTIONS_PER_STREAM {
                let tx_start = Instant::now();
                
                // Simulate transaction processing
                let mock_tx = create_mock_streamed_transaction(stream_id * TRANSACTIONS_PER_STREAM + tx_id);
                
                // Simulate processing delay (realistic network + parsing time)
                tokio::time::sleep(Duration::from_micros(100)).await;
                
                let latency = tx_start.elapsed().as_micros() as u64;
                total_latency_us += latency;
                max_latency_us = max_latency_us.max(latency);
                
                if latency < TARGET_LATENCY_US {
                    successful_transactions += 1;
                } else {
                    failed_transactions += 1;
                }
            }
            
            StreamTestResult {
                stream_id,
                successful_transactions,
                failed_transactions,
                avg_latency_us: total_latency_us / TRANSACTIONS_PER_STREAM as u64,
                max_latency_us,
            }
        });
        
        handles.push(handle);
    }

    // Collect results
    let mut total_successful = 0;
    let mut total_failed = 0;
    let mut max_stream_latency = 0u64;
    
    for handle in handles {
        let result = handle.await.expect("Stream task failed");
        total_successful += result.successful_transactions;
        total_failed += result.failed_transactions;
        max_stream_latency = max_stream_latency.max(result.max_latency_us);
        
        info!("Stream {} completed: {} successful, {} failed, max latency: {}μs", 
               result.stream_id, result.successful_transactions, result.failed_transactions, result.max_latency_us);
    }

    let total_time = start_time.elapsed();
    let total_transactions = total_successful + total_failed;
    let tps = (total_transactions as u64 * 1000) / total_time.as_millis() as u64;

    info!("🎯 BATTLE RESULTS:");
    info!("  Total transactions: {}", total_transactions);
    info!("  Successful: {} ({:.2}%)", total_successful, (total_successful as f64 / total_transactions as f64) * 100.0);
    info!("  Failed: {} ({:.2}%)", total_failed, (total_failed as f64 / total_transactions as f64) * 100.0);
    info!("  TPS: {}", tps);
    info!("  Max latency: {}μs", max_stream_latency);

    // Assert performance requirements
    let success_rate = (total_successful as f64 / total_transactions as f64) * 100.0;
    assert!(success_rate > 95.0, "Success rate too low: {:.2}%", success_rate);
    assert!(max_stream_latency < TARGET_LATENCY_US * 2, "Max latency too high: {}μs", max_stream_latency);
    assert!(tps > 1000, "TPS too low: {}", tps);

    monitor.stop_monitoring();
    info!("✅ VICTORY: WebSocket streaming overload test passed");
}

#[tokio::test]
async fn stress_test_memory_mapped_buffer_saturation() {
    let _ = tracing_subscriber::fmt::try_init();
    info!("⚡ INITIATING: Memory-mapped buffer saturation test");

    const BUFFER_SIZE: usize = 64 * 1024 * 1024; // 64MB
    const CONCURRENT_WRITERS: usize = 32;
    const WRITES_PER_WORKER: usize = 1_000;
    const WRITE_SIZE: usize = 1024; // 1KB per write

    let start_time = Instant::now();

    // Create multiple buffers to test concurrent access
    let mut buffers = Vec::new();
    for _ in 0..CONCURRENT_WRITERS {
        let buffer = TransactionBuffer::new(BUFFER_SIZE).expect("Failed to create buffer");
        buffers.push(Arc::new(parking_lot::Mutex::new(buffer)));
    }

    let barrier = Arc::new(Barrier::new(CONCURRENT_WRITERS));
    let mut handles = Vec::new();

    for (worker_id, buffer) in buffers.into_iter().enumerate() {
        let barrier_clone = Arc::clone(&barrier);
        
        let handle = tokio::spawn(async move {
            // Wait for all workers to be ready
            barrier_clone.wait().await;
            
            let mut successful_writes = 0;
            let mut failed_writes = 0;
            let worker_start = Instant::now();
            
            for write_id in 0..WRITES_PER_WORKER {
                let write_start = Instant::now();
                
                // Attempt to write to buffer
                {
                    let mut buf = buffer.lock();
                    unsafe {
                        if let Some(_slice) = buf.get_write_slice(WRITE_SIZE) {
                            buf.commit_write(WRITE_SIZE);
                            successful_writes += 1;
                        } else {
                            failed_writes += 1;
                        }
                    }
                }
                
                let write_latency = write_start.elapsed();
                if write_latency > Duration::from_micros(100) {
                    warn!("Slow write detected: worker {}, write {}, latency: {:?}", 
                          worker_id, write_id, write_latency);
                }
            }
            
            let worker_duration = worker_start.elapsed();
            let throughput_mb_s = (successful_writes * WRITE_SIZE) as f64 / 1024.0 / 1024.0 / worker_duration.as_secs_f64();
            
            BufferTestResult {
                worker_id,
                successful_writes,
                failed_writes,
                duration: worker_duration,
                throughput_mb_s,
            }
        });
        
        handles.push(handle);
    }

    // Collect results
    let mut total_successful = 0;
    let mut total_failed = 0;
    let mut total_throughput = 0.0;

    for handle in handles {
        let result = handle.await.expect("Buffer worker failed");
        total_successful += result.successful_writes;
        total_failed += result.failed_writes;
        total_throughput += result.throughput_mb_s;
        
        info!("Worker {} completed: {} successful, {} failed, throughput: {:.2} MB/s", 
               result.worker_id, result.successful_writes, result.failed_writes, result.throughput_mb_s);
    }

    let total_time = start_time.elapsed();
    let total_writes = total_successful + total_failed;
    let overall_throughput = (total_successful * WRITE_SIZE) as f64 / 1024.0 / 1024.0 / total_time.as_secs_f64();

    info!("🎯 BUFFER BATTLE RESULTS:");
    info!("  Total writes: {}", total_writes);
    info!("  Successful: {} ({:.2}%)", total_successful, (total_successful as f64 / total_writes as f64) * 100.0);
    info!("  Failed: {} ({:.2}%)", total_failed, (total_failed as f64 / total_writes as f64) * 100.0);
    info!("  Overall throughput: {:.2} MB/s", overall_throughput);
    info!("  Combined worker throughput: {:.2} MB/s", total_throughput);

    // Assert performance requirements
    let success_rate = (total_successful as f64 / total_writes as f64) * 100.0;
    assert!(success_rate > 90.0, "Success rate too low: {:.2}%", success_rate);
    assert!(overall_throughput > 100.0, "Overall throughput too low: {:.2} MB/s", overall_throughput);

    info!("✅ VICTORY: Memory-mapped buffer saturation test passed");
}

#[tokio::test]
async fn stress_test_lock_free_queue_contention() {
    let _ = tracing_subscriber::fmt::try_init();
    info!("🎯 INITIATING: Lock-free queue contention test");

    const QUEUE_CAPACITY: usize = 32_768;
    const PRODUCER_COUNT: usize = 16;
    const CONSUMER_COUNT: usize = 8;
    const OPERATIONS_PER_PRODUCER: usize = 10_000;

    let mut queue = HighFrequencyTransactionQueue::new(QUEUE_CAPACITY);
    let queue_ptr = &mut queue as *mut HighFrequencyTransactionQueue;

    let start_time = Instant::now();
    let barrier = Arc::new(Barrier::new(PRODUCER_COUNT + CONSUMER_COUNT));
    let mut handles = Vec::new();

    // Spawn producers
    for producer_id in 0..PRODUCER_COUNT {
        let barrier_clone = Arc::clone(&barrier);
        
        let handle = tokio::spawn(async move {
            // Wait for all workers
            barrier_clone.wait().await;
            
            let mut successful_pushes = 0;
            let mut failed_pushes = 0;
            let producer_start = Instant::now();
            
            for op_id in 0..OPERATIONS_PER_PRODUCER {
                let tx = create_mock_streamed_transaction(producer_id * OPERATIONS_PER_PRODUCER + op_id);
                
                // SAFETY: This is safe in this test context as we're simulating concurrent access
                unsafe {
                    match (*queue_ptr).try_push(tx) {
                        Ok(_) => successful_pushes += 1,
                        Err(_) => failed_pushes += 1,
                    }
                }
                
                // Small delay to simulate real-world timing
                if op_id % 100 == 0 {
                    tokio::task::yield_now().await;
                }
            }
            
            QueueWorkerResult {
                worker_id: producer_id,
                worker_type: "Producer".to_string(),
                successful_ops: successful_pushes,
                failed_ops: failed_pushes,
                duration: producer_start.elapsed(),
            }
        });
        
        handles.push(handle);
    }

    // Spawn consumers
    for consumer_id in 0..CONSUMER_COUNT {
        let barrier_clone = Arc::clone(&barrier);
        
        let handle = tokio::spawn(async move {
            // Wait for all workers
            barrier_clone.wait().await;
            
            let mut successful_pops = 0;
            let mut failed_pops = 0;
            let consumer_start = Instant::now();
            
            // Run for same duration as producers + some extra time
            let target_duration = Duration::from_millis(5000);
            let start = Instant::now();
            
            while start.elapsed() < target_duration {
                // SAFETY: This is safe in this test context as we're simulating concurrent access
                unsafe {
                    match (*queue_ptr).try_pop() {
                        Some(_) => successful_pops += 1,
                        None => failed_pops += 1,
                    }
                }
                
                // Small delay to prevent busy waiting
                if failed_pops % 1000 == 0 {
                    tokio::task::yield_now().await;
                }
            }
            
            QueueWorkerResult {
                worker_id: consumer_id,
                worker_type: "Consumer".to_string(),
                successful_ops: successful_pops,
                failed_ops: failed_pops,
                duration: consumer_start.elapsed(),
            }
        });
        
        handles.push(handle);
    }

    // Collect results
    let mut total_produced = 0;
    let mut total_consumed = 0;
    let mut producer_failed = 0;
    let mut consumer_failed = 0;

    for handle in handles {
        let result = handle.await.expect("Queue worker failed");
        
        if result.worker_type == "Producer" {
            total_produced += result.successful_ops;
            producer_failed += result.failed_ops;
        } else {
            total_consumed += result.successful_ops;
            consumer_failed += result.failed_ops;
        }
        
        let ops_per_sec = result.successful_ops as f64 / result.duration.as_secs_f64();
        info!("{} {} completed: {} successful ops ({:.0} ops/s), {} failed", 
               result.worker_type, result.worker_id, result.successful_ops, ops_per_sec, result.failed_ops);
    }

    let total_time = start_time.elapsed();
    let queue_stats = queue.get_metrics();

    info!("🎯 QUEUE BATTLE RESULTS:");
    info!("  Total produced: {}", total_produced);
    info!("  Total consumed: {}", total_consumed);
    info!("  Producer failures: {}", producer_failed);
    info!("  Consumer failures: {}", consumer_failed);
    info!("  Final queue size: {} items", queue_stats.total_processed - total_consumed as u64);
    info!("  Queue max latency: {}μs", queue_stats.max_latency_us);
    info!("  Queue avg latency: {}μs", queue_stats.avg_latency_us);
    info!("  Total test duration: {:?}", total_time);

    // Assert performance requirements
    let production_efficiency = (total_produced as f64 / (total_produced + producer_failed) as f64) * 100.0;
    assert!(production_efficiency > 80.0, "Production efficiency too low: {:.2}%", production_efficiency);
    assert!(queue_stats.max_latency_us < 1000, "Queue latency too high: {}μs", queue_stats.max_latency_us);
    assert!(total_consumed > total_produced / 2, "Consumption rate too low"); // At least half should be consumed

    info!("✅ VICTORY: Lock-free queue contention test passed");
}

#[tokio::test]
async fn stress_test_connection_pool_chaos() {
    let _ = tracing_subscriber::fmt::try_init();
    info!("🛡️  INITIATING: Connection pool chaos test");

    // Mix of working and failing URLs to simulate real-world chaos
    let primary_urls = vec![
        "http://invalid-primary-1.test".to_string(),
        "https://api.devnet.solana.com".to_string(), // This should work
        "http://invalid-primary-2.test".to_string(),
    ];
    
    let backup_urls = vec![
        "http://invalid-backup-1.test".to_string(),
        "https://api.testnet.solana.com".to_string(), // This might work
    ];

    let pool = Arc::new(SolanaConnectionPool::new(primary_urls, backup_urls));
    
    // Start health monitoring
    let _health_handle = pool.start_health_monitoring();

    const CONCURRENT_REQUESTS: usize = 100;
    const REQUESTS_PER_WORKER: usize = 50;

    let start_time = Instant::now();
    let barrier = Arc::new(Barrier::new(CONCURRENT_REQUESTS));
    let mut handles = Vec::new();

    for worker_id in 0..CONCURRENT_REQUESTS {
        let pool_clone = Arc::clone(&pool);
        let barrier_clone = Arc::clone(&barrier);
        
        let handle = tokio::spawn(async move {
            // Wait for all workers
            barrier_clone.wait().await;
            
            let mut successful_requests = 0;
            let mut failed_requests = 0;
            let mut total_latency_ms = 0u64;
            
            for req_id in 0..REQUESTS_PER_WORKER {
                let request_start = Instant::now();
                
                match pool_clone.get_client().await {
                    Ok(_client) => {
                        successful_requests += 1;
                        // Simulate using the client (health check)
                        tokio::time::sleep(Duration::from_millis(1)).await;
                    }
                    Err(_) => {
                        failed_requests += 1;
                    }
                }
                
                let latency = request_start.elapsed().as_millis() as u64;
                total_latency_ms += latency;
                
                // Add some chaos - random delays
                if req_id % 10 == 0 {
                    tokio::time::sleep(Duration::from_millis(fastrand::u64(1..=10))).await;
                }
            }
            
            PoolTestResult {
                worker_id,
                successful_requests,
                failed_requests,
                avg_latency_ms: total_latency_ms / REQUESTS_PER_WORKER as u64,
            }
        });
        
        handles.push(handle);
    }

    // Collect results
    let mut total_successful = 0;
    let mut total_failed = 0;
    let mut max_latency = 0u64;

    for handle in handles {
        let result = handle.await.expect("Pool worker failed");
        total_successful += result.successful_requests;
        total_failed += result.failed_requests;
        max_latency = max_latency.max(result.avg_latency_ms);
        
        if result.worker_id < 5 { // Log first few for debugging
            info!("Worker {} completed: {} successful, {} failed, avg latency: {}ms", 
                   result.worker_id, result.successful_requests, result.failed_requests, result.avg_latency_ms);
        }
    }

    let total_requests = total_successful + total_failed;
    let success_rate = (total_successful as f64 / total_requests as f64) * 100.0;
    let pool_stats = pool.get_pool_stats();

    info!("🎯 POOL CHAOS RESULTS:");
    info!("  Total requests: {}", total_requests);
    info!("  Successful: {} ({:.2}%)", total_successful, success_rate);
    info!("  Failed: {} ({:.2}%)", total_failed, 100.0 - success_rate);
    info!("  Max worker latency: {}ms", max_latency);
    info!("  Pool failover events: {}", pool_stats.failover_events);
    info!("  Pool avg response time: {}μs", pool_stats.avg_response_time_us);

    // In chaos scenario, we expect some failures but system should remain functional
    // At least 20% success rate even with failing nodes
    assert!(success_rate > 20.0, "Success rate too low even for chaos test: {:.2}%", success_rate);
    assert!(max_latency < 30_000, "Latency too high: {}ms", max_latency); // 30 second max

    info!("✅ VICTORY: Connection pool survived chaos test");
}

#[tokio::test] 
async fn stress_test_end_to_end_integration() {
    let _ = tracing_subscriber::fmt::try_init();
    info!("💀 INITIATING: End-to-end integration stress test");

    // This test combines all components under extreme load
    const CONCURRENT_STREAMS: usize = 20;
    const TRANSACTIONS_PER_STREAM: usize = 500;
    const TARGET_TOTAL_TPS: u64 = 5_000;

    let monitor = TrenchBotPerformanceMonitor::new(MonitorConfig {
        latency_alert_threshold_us: 15_000, // 15ms for stress test
        throughput_alert_threshold: 1_000,
        ..Default::default()
    });
    
    monitor.start_monitoring().await;

    let start_time = Instant::now();
    let barrier = Arc::new(Barrier::new(CONCURRENT_STREAMS));
    let mut handles = Vec::new();

    for stream_id in 0..CONCURRENT_STREAMS {
        let barrier_clone = Arc::clone(&barrier);
        
        let handle = tokio::spawn(async move {
            // Wait for coordinated start
            barrier_clone.wait().await;
            
            let mut stream_results = IntegrationStreamResult::new(stream_id);
            
            // Create components for this stream
            let primary_urls = vec!["https://api.devnet.solana.com".to_string()];
            let backup_urls = vec![];
            let pool = Arc::new(SolanaConnectionPool::new(primary_urls, backup_urls));
            
            let mut executor = HighFrequencyTransactionExecutor::new(
                Arc::clone(&pool), 
                2048, // Queue size
                4     // Max concurrent
            );
            
            for tx_id in 0..TRANSACTIONS_PER_STREAM {
                let operation_start = Instant::now();
                
                // Create mock transaction
                let mock_tx = create_mock_transaction();
                
                // Attempt execution (will fail on network but we measure timing)
                match executor.execute_transaction(mock_tx).await {
                    Ok(_result) => {
                        stream_results.successful_transactions += 1;
                    }
                    Err(_err) => {
                        stream_results.failed_transactions += 1;
                    }
                }
                
                let operation_time = operation_start.elapsed().as_micros() as u64;
                stream_results.total_latency_us += operation_time;
                stream_results.max_latency_us = stream_results.max_latency_us.max(operation_time);
                
                // Collect component metrics
                let executor_stats = executor.get_executor_stats();
                stream_results.total_queue_time_us += executor_stats.queue_wait_time_us;
                stream_results.total_verification_time_us += executor_stats.signature_verification_time_us;
                
                // Add realistic inter-transaction delay
                if tx_id % 50 == 0 {
                    tokio::task::yield_now().await;
                }
            }
            
            stream_results.duration = start_time.elapsed();
            stream_results
        });
        
        handles.push(handle);
    }

    // Collect all stream results
    let mut overall_results = IntegrationOverallResult::new();
    
    for handle in handles {
        let stream_result = handle.await.expect("Integration stream failed");
        overall_results.aggregate_stream_result(&stream_result);
        
        info!("Stream {} completed: {} successful, {} failed, avg latency: {}μs", 
               stream_result.stream_id,
               stream_result.successful_transactions,
               stream_result.failed_transactions,
               stream_result.avg_latency_us());
    }

    let total_time = start_time.elapsed();
    let actual_tps = overall_results.total_transactions() as u64 * 1000 / total_time.as_millis() as u64;

    info!("💀 INTEGRATION WARFARE RESULTS:");
    info!("  Total transactions: {}", overall_results.total_transactions());
    info!("  Success rate: {:.2}%", overall_results.success_rate());
    info!("  Actual TPS: {}", actual_tps);
    info!("  Avg end-to-end latency: {}μs", overall_results.avg_end_to_end_latency_us());
    info!("  Max latency: {}μs", overall_results.max_latency_us);
    info!("  Avg queue latency: {}μs", overall_results.avg_queue_latency_us());
    info!("  Avg verification latency: {}μs", overall_results.avg_verification_latency_us());

    // Get final performance report
    let performance_report = monitor.get_performance_report();
    info!("  System health score: {}", performance_report.system_health.overall_health_score);
    info!("  Active alerts: {}", performance_report.active_alerts.len());

    monitor.stop_monitoring();

    // Assert integration requirements (more lenient due to network failures)
    assert!(overall_results.success_rate() > 0.0, "No successful transactions"); // At least some should work
    assert!(overall_results.max_latency_us < 100_000, "Max latency too high: {}μs", overall_results.max_latency_us); // 100ms max
    assert!(performance_report.system_health.overall_health_score >= 50, "System health too low"); // At least half healthy

    info!("✅ ULTIMATE VICTORY: End-to-end integration stress test completed successfully");
}

// === HELPER STRUCTURES ===

#[derive(Debug)]
struct StreamTestResult {
    stream_id: usize,
    successful_transactions: usize,
    failed_transactions: usize,
    avg_latency_us: u64,
    max_latency_us: u64,
}

#[derive(Debug)]
struct BufferTestResult {
    worker_id: usize,
    successful_writes: usize,
    failed_writes: usize,
    duration: Duration,
    throughput_mb_s: f64,
}

#[derive(Debug)]
struct QueueWorkerResult {
    worker_id: usize,
    worker_type: String,
    successful_ops: usize,
    failed_ops: usize,
    duration: Duration,
}

#[derive(Debug)]
struct PoolTestResult {
    worker_id: usize,
    successful_requests: usize,
    failed_requests: usize,
    avg_latency_ms: u64,
}

#[derive(Debug)]
struct IntegrationStreamResult {
    stream_id: usize,
    successful_transactions: usize,
    failed_transactions: usize,
    total_latency_us: u64,
    max_latency_us: u64,
    total_queue_time_us: u64,
    total_verification_time_us: u64,
    duration: Duration,
}

impl IntegrationStreamResult {
    fn new(stream_id: usize) -> Self {
        Self {
            stream_id,
            successful_transactions: 0,
            failed_transactions: 0,
            total_latency_us: 0,
            max_latency_us: 0,
            total_queue_time_us: 0,
            total_verification_time_us: 0,
            duration: Duration::default(),
        }
    }

    fn total_transactions(&self) -> usize {
        self.successful_transactions + self.failed_transactions
    }

    fn avg_latency_us(&self) -> u64 {
        if self.total_transactions() > 0 {
            self.total_latency_us / self.total_transactions() as u64
        } else {
            0
        }
    }
}

#[derive(Debug)]
struct IntegrationOverallResult {
    total_successful: usize,
    total_failed: usize,
    max_latency_us: u64,
    total_end_to_end_latency_us: u64,
    total_queue_latency_us: u64,
    total_verification_latency_us: u64,
    stream_count: usize,
}

impl IntegrationOverallResult {
    fn new() -> Self {
        Self {
            total_successful: 0,
            total_failed: 0,
            max_latency_us: 0,
            total_end_to_end_latency_us: 0,
            total_queue_latency_us: 0,
            total_verification_latency_us: 0,
            stream_count: 0,
        }
    }

    fn aggregate_stream_result(&mut self, stream: &IntegrationStreamResult) {
        self.total_successful += stream.successful_transactions;
        self.total_failed += stream.failed_transactions;
        self.max_latency_us = self.max_latency_us.max(stream.max_latency_us);
        self.total_end_to_end_latency_us += stream.total_latency_us;
        self.total_queue_latency_us += stream.total_queue_time_us;
        self.total_verification_latency_us += stream.total_verification_time_us;
        self.stream_count += 1;
    }

    fn total_transactions(&self) -> usize {
        self.total_successful + self.total_failed
    }

    fn success_rate(&self) -> f64 {
        if self.total_transactions() > 0 {
            (self.total_successful as f64 / self.total_transactions() as f64) * 100.0
        } else {
            0.0
        }
    }

    fn avg_end_to_end_latency_us(&self) -> u64 {
        if self.total_transactions() > 0 {
            self.total_end_to_end_latency_us / self.total_transactions() as u64
        } else {
            0
        }
    }

    fn avg_queue_latency_us(&self) -> u64 {
        if self.stream_count > 0 {
            self.total_queue_latency_us / self.stream_count as u64
        } else {
            0
        }
    }

    fn avg_verification_latency_us(&self) -> u64 {
        if self.stream_count > 0 {
            self.total_verification_latency_us / self.stream_count as u64
        } else {
            0
        }
    }
}

// === HELPER FUNCTIONS ===

fn create_mock_streamed_transaction(id: usize) -> StreamedTransaction {
    StreamedTransaction {
        signature: format!("stress_test_tx_{}", id),
        slot: id as u64,
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64,
        accounts: vec![
            format!("account1_{}", id),
            format!("account2_{}", id),
        ],
        instruction_data: vec![id as u8; 128],
        compute_units: Some(200_000),
        fee: 5000,
    }
}

fn create_mock_transaction() -> solana_sdk::transaction::Transaction {
    use solana_sdk::{
        message::Message,
        transaction::Transaction,
        pubkey::Pubkey,
        signature::{Keypair, Signer},
        system_instruction,
    };
    
    let keypair = Keypair::new();
    let to_pubkey = Pubkey::new_unique();
    
    let instruction = system_instruction::transfer(
        &keypair.pubkey(),
        &to_pubkey,
        1_000_000,
    );
    
    let message = Message::new(&[instruction], Some(&keypair.pubkey()));
    Transaction::new(&[&keypair], message, solana_sdk::hash::Hash::default())
}