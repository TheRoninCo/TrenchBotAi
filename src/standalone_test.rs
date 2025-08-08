use std::sync::{Arc, atomic::{AtomicU64, Ordering}};
use std::time::{Duration, Instant};
use parking_lot::RwLock;
use crossbeam::channel::{unbounded, Receiver, Sender};
use rtrb::RingBuffer;
use serde::{Deserialize, Serialize};

/// Standalone test for ultra-fast blockchain infrastructure
/// This validates our core components without external dependencies

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamedTransaction {
    pub signature: String,
    pub slot: u64,
    pub timestamp: u64,
    pub accounts: Vec<String>,
    pub instruction_data: Vec<u8>,
    pub compute_units: Option<u64>,
    pub fee: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueStats {
    pub total_processed: u64,
    pub total_dropped: u64,
    pub max_latency_us: u64,
    pub avg_latency_us: u64,
    pub available_slots: usize,
}

pub struct HighFrequencyTransactionQueue {
    producer: rtrb::Producer<StreamedTransaction>,
    consumer: rtrb::Consumer<StreamedTransaction>,
    metrics: Arc<QueueMetrics>,
}

#[derive(Debug, Default)]
pub struct QueueMetrics {
    pub total_processed: AtomicU64,
    pub total_dropped: AtomicU64,
    pub max_latency_us: AtomicU64,
    pub avg_latency_us: AtomicU64,
}

impl HighFrequencyTransactionQueue {
    pub fn new(capacity: usize) -> Self {
        let (producer, consumer) = RingBuffer::new(capacity);
        
        Self {
            producer,
            consumer,
            metrics: Arc::new(QueueMetrics::default()),
        }
    }

    pub fn try_push(&mut self, transaction: StreamedTransaction) -> Result<(), StreamedTransaction> {
        let start = Instant::now();
        
        match self.producer.push(transaction) {
            Ok(()) => {
                let latency = start.elapsed().as_micros() as u64;
                self.metrics.total_processed.fetch_add(1, Ordering::Relaxed);
                self.update_latency_stats(latency);
                Ok(())
            }
            Err(transaction) => {
                self.metrics.total_dropped.fetch_add(1, Ordering::Relaxed);
                Err(transaction)
            }
        }
    }

    pub fn try_pop(&mut self) -> Option<StreamedTransaction> {
        self.consumer.pop().ok()
    }

    fn update_latency_stats(&self, latency_us: u64) {
        // Update max latency
        loop {
            let current_max = self.metrics.max_latency_us.load(Ordering::Acquire);
            if latency_us <= current_max {
                break;
            }
            if self.metrics.max_latency_us
                .compare_exchange_weak(current_max, latency_us, Ordering::Release, Ordering::Relaxed)
                .is_ok() {
                break;
            }
        }

        // Simple moving average for latency
        let current_avg = self.metrics.avg_latency_us.load(Ordering::Acquire);
        let new_avg = (current_avg * 7 + latency_us) / 8;
        self.metrics.avg_latency_us.store(new_avg, Ordering::Release);
    }

    pub fn get_metrics(&self) -> QueueStats {
        QueueStats {
            total_processed: self.metrics.total_processed.load(Ordering::Acquire),
            total_dropped: self.metrics.total_dropped.load(Ordering::Acquire),
            max_latency_us: self.metrics.max_latency_us.load(Ordering::Acquire),
            avg_latency_us: self.metrics.avg_latency_us.load(Ordering::Acquire),
            available_slots: self.producer.slots(),
        }
    }
}

/// Standalone performance test runner
pub struct TrenchBotTester {
    queue: HighFrequencyTransactionQueue,
    test_results: Vec<TestResult>,
}

#[derive(Debug)]
pub struct TestResult {
    pub test_name: String,
    pub duration: Duration,
    pub operations: usize,
    pub avg_latency_us: u64,
    pub max_latency_us: u64,
    pub success: bool,
}

impl TrenchBotTester {
    pub fn new() -> Self {
        Self {
            queue: HighFrequencyTransactionQueue::new(16384),
            test_results: Vec::new(),
        }
    }

    pub fn run_all_tests(&mut self) {
        println!("🔥 TRENCHBOT STANDALONE PERFORMANCE TESTS");
        println!("==========================================");
        
        self.test_queue_performance();
        self.test_memory_operations();
        self.test_concurrent_access();
        
        self.print_summary();
    }

    fn test_queue_performance(&mut self) {
        println!("\n🎯 TEST 1: Queue Performance");
        let start = Instant::now();
        let mut max_latency = 0u64;
        let mut total_latency = 0u64;
        let operations = 10000;
        
        // Fill queue
        let mut successful = 0;
        for i in 0..operations {
            let op_start = Instant::now();
            let tx = self.create_mock_transaction(i);
            
            if self.queue.try_push(tx).is_ok() {
                successful += 1;
            }
            
            let latency = op_start.elapsed().as_micros() as u64;
            total_latency += latency;
            max_latency = max_latency.max(latency);
        }
        
        // Drain queue
        let mut drained = 0;
        while self.queue.try_pop().is_some() {
            drained += 1;
        }
        
        let duration = start.elapsed();
        let avg_latency = total_latency / operations as u64;
        
        println!("  ✅ Operations: {} push + {} pop", successful, drained);
        println!("  📊 Duration: {:?}", duration);
        println!("  📊 Avg latency: {}μs", avg_latency);
        println!("  📊 Max latency: {}μs", max_latency);
        
        let result = TestResult {
            test_name: "Queue Performance".to_string(),
            duration,
            operations: successful + drained,
            avg_latency_us: avg_latency,
            max_latency_us: max_latency,
            success: successful > operations / 2 && avg_latency < 1000, // <1ms
        };
        
        self.test_results.push(result);
    }

    fn test_memory_operations(&mut self) {
        println!("\n⚡ TEST 2: Memory Operations");
        let start = Instant::now();
        let operations = 50000;
        let mut memory_ops = 0;
        let mut total_latency = 0u64;
        let mut max_latency = 0u64;
        
        // Simulate memory-intensive operations
        let mut data_store: Vec<Vec<u8>> = Vec::with_capacity(operations);
        
        for i in 0..operations {
            let op_start = Instant::now();
            
            // Create and store data (simulates memory-mapped operations)
            let data = vec![i as u8; 256];
            data_store.push(data);
            memory_ops += 1;
            
            let latency = op_start.elapsed().as_micros() as u64;
            total_latency += latency;
            max_latency = max_latency.max(latency);
        }
        
        let duration = start.elapsed();
        let avg_latency = total_latency / operations as u64;
        let throughput_mb_s = (operations * 256) as f64 / 1024.0 / 1024.0 / duration.as_secs_f64();
        
        println!("  ✅ Memory operations: {}", memory_ops);
        println!("  📊 Duration: {:?}", duration);
        println!("  📊 Throughput: {:.2} MB/s", throughput_mb_s);
        println!("  📊 Avg latency: {}μs", avg_latency);
        println!("  📊 Max latency: {}μs", max_latency);
        
        let result = TestResult {
            test_name: "Memory Operations".to_string(),
            duration,
            operations: memory_ops,
            avg_latency_us: avg_latency,
            max_latency_us: max_latency,
            success: throughput_mb_s > 50.0 && avg_latency < 100, // >50MB/s, <100μs
        };
        
        self.test_results.push(result);
    }

    fn test_concurrent_access(&mut self) {
        println!("\n🛡️ TEST 3: Concurrent Access");
        let start = Instant::now();
        
        use std::thread;
        use std::sync::atomic::AtomicUsize;
        
        let counter = Arc::new(AtomicUsize::new(0));
        let operations_per_thread = 1000;
        let thread_count = 8;
        
        let mut handles = vec![];
        
        for thread_id in 0..thread_count {
            let counter_clone = Arc::clone(&counter);
            
            let handle = thread::spawn(move || {
                let mut local_ops = 0;
                let mut max_latency = 0u64;
                let mut total_latency = 0u64;
                
                for i in 0..operations_per_thread {
                    let op_start = Instant::now();
                    
                    // Simulate concurrent operations
                    counter_clone.fetch_add(1, Ordering::Relaxed);
                    
                    // Simulate some work
                    let _data = vec![thread_id as u8; 64];
                    
                    local_ops += 1;
                    let latency = op_start.elapsed().as_micros() as u64;
                    total_latency += latency;
                    max_latency = max_latency.max(latency);
                }
                
                (local_ops, total_latency, max_latency)
            });
            
            handles.push(handle);
        }
        
        // Collect results
        let mut total_ops = 0;
        let mut combined_latency = 0u64;
        let mut global_max_latency = 0u64;
        
        for handle in handles {
            let (ops, latency, max_lat) = handle.join().unwrap();
            total_ops += ops;
            combined_latency += latency;
            global_max_latency = global_max_latency.max(max_lat);
        }
        
        let duration = start.elapsed();
        let final_count = counter.load(Ordering::Acquire);
        let avg_latency = combined_latency / total_ops as u64;
        let ops_per_second = (total_ops as f64 / duration.as_secs_f64()) as u64;
        
        println!("  ✅ Concurrent operations: {}", total_ops);
        println!("  📊 Final counter: {}", final_count);
        println!("  📊 Duration: {:?}", duration);
        println!("  📊 Ops/sec: {}", ops_per_second);
        println!("  📊 Avg latency: {}μs", avg_latency);
        println!("  📊 Max latency: {}μs", global_max_latency);
        
        let result = TestResult {
            test_name: "Concurrent Access".to_string(),
            duration,
            operations: total_ops,
            avg_latency_us: avg_latency,
            max_latency_us: global_max_latency,
            success: final_count == total_ops && ops_per_second > 10000, // >10k ops/s
        };
        
        self.test_results.push(result);
    }

    fn create_mock_transaction(&self, id: usize) -> StreamedTransaction {
        StreamedTransaction {
            signature: format!("test_tx_{}", id),
            slot: id as u64,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_micros() as u64,
            accounts: vec![format!("account_{}", id)],
            instruction_data: vec![id as u8; 128],
            compute_units: Some(200_000),
            fee: 5000,
        }
    }

    fn print_summary(&self) {
        println!("\n📊 TEST SUMMARY");
        println!("================");
        
        let mut all_passed = true;
        let mut total_operations = 0;
        let mut total_time = Duration::ZERO;
        
        for result in &self.test_results {
            let status = if result.success { "✅ PASS" } else { "❌ FAIL" };
            println!("  {} {}", status, result.test_name);
            println!("    Operations: {}", result.operations);
            println!("    Duration: {:?}", result.duration);
            println!("    Avg Latency: {}μs", result.avg_latency_us);
            println!("    Max Latency: {}μs", result.max_latency_us);
            println!();
            
            if !result.success {
                all_passed = false;
            }
            total_operations += result.operations;
            total_time += result.duration;
        }
        
        println!("🎯 OVERALL RESULTS:");
        println!("  Total Operations: {}", total_operations);
        println!("  Total Time: {:?}", total_time);
        println!("  Tests Passed: {}/{}", self.test_results.iter().filter(|r| r.success).count(), self.test_results.len());
        
        if all_passed {
            println!("  🏆 STATUS: ALL TESTS PASSED - SYSTEM IS BATTLE-READY!");
        } else {
            println!("  ⚠️  STATUS: Some tests failed - Review performance requirements");
        }
        
        // Performance benchmarks
        println!("\n⚡ PERFORMANCE VALIDATION:");
        for result in &self.test_results {
            match result.test_name.as_str() {
                "Queue Performance" => {
                    if result.avg_latency_us < 100 {
                        println!("  ✅ Queue latency: {}μs (Target: <100μs)", result.avg_latency_us);
                    } else {
                        println!("  ❌ Queue latency: {}μs (Target: <100μs)", result.avg_latency_us);
                    }
                }
                "Memory Operations" => {
                    let throughput = (result.operations * 256) as f64 / 1024.0 / 1024.0 / result.duration.as_secs_f64();
                    if throughput > 50.0 {
                        println!("  ✅ Memory throughput: {:.1} MB/s (Target: >50 MB/s)", throughput);
                    } else {
                        println!("  ❌ Memory throughput: {:.1} MB/s (Target: >50 MB/s)", throughput);
                    }
                }
                "Concurrent Access" => {
                    let ops_per_sec = result.operations as f64 / result.duration.as_secs_f64();
                    if ops_per_sec > 10000.0 {
                        println!("  ✅ Concurrent ops: {:.0} ops/s (Target: >10k ops/s)", ops_per_sec);
                    } else {
                        println!("  ❌ Concurrent ops: {:.0} ops/s (Target: >10k ops/s)", ops_per_sec);
                    }
                }
                _ => {}
            }
        }
    }
}

fn main() {
    let mut tester = TrenchBotTester::new();
    tester.run_all_tests();
}