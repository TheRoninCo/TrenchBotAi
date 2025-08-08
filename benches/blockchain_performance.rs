use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use std::time::Duration;
use tokio::runtime::Runtime;
use trenchbot_dex::infrastructure::solana_rpc::*;
use rayon::prelude::*;

/// Comprehensive benchmarking suite for ultra-low latency blockchain operations
/// 
/// 🎯 COMBAT EFFECTIVENESS METRICS:
/// - WebSocket streaming latency
/// - Memory-mapped buffer throughput
/// - Lock-free queue performance
/// - SIMD signature verification speed
/// - End-to-end transaction execution
/// - Connection pool failover time

fn bench_websocket_streaming(c: &mut Criterion) {
    let mut group = c.benchmark_group("🔥 WebSocket Streaming");
    group.measurement_time(Duration::from_secs(10));
    
    let rt = Runtime::new().unwrap();
    
    for tx_count in [100, 1_000, 10_000, 100_000].iter() {
        group.throughput(Throughput::Elements(*tx_count as u64));
        
        group.bench_with_input(
            BenchmarkId::new("transaction_parsing", tx_count),
            tx_count,
            |b, &count| {
                b.to_async(&rt).iter(|| async {
                    // Simulate parsing streamed transactions
                    let mut parsed_count = 0;
                    for i in 0..count {
                        let mock_transaction = create_mock_streamed_transaction(i);
                        parsed_count += black_box(mock_transaction.accounts.len());
                    }
                    parsed_count
                });
            },
        );
    }
    
    group.finish();
}

fn bench_memory_mapped_buffer(c: &mut Criterion) {
    let mut group = c.benchmark_group("⚡ Memory-Mapped Buffer");
    group.measurement_time(Duration::from_secs(5));
    
    for buffer_size in [1024, 8192, 65536, 1_048_576].iter() {
        group.throughput(Throughput::Bytes(*buffer_size as u64));
        
        group.bench_with_input(
            BenchmarkId::new("zero_copy_writes", buffer_size),
            buffer_size,
            |b, &size| {
                b.iter(|| {
                    // Test zero-copy write performance
                    let mut buffer = TransactionBuffer::new(size).unwrap();
                    let write_size = 64; // Typical transaction size
                    let mut total_written = 0;
                    
                    while total_written + write_size < size {
                        unsafe {
                            if let Some(_slice) = buffer.get_write_slice(write_size) {
                                buffer.commit_write(write_size);
                                total_written += write_size;
                            } else {
                                break;
                            }
                        }
                    }
                    total_written
                });
            },
        );
    }
    
    group.finish();
}

fn bench_lock_free_queue(c: &mut Criterion) {
    let mut group = c.benchmark_group("🎯 Lock-Free Queue");
    group.measurement_time(Duration::from_secs(8));
    
    for capacity in [1024, 8192, 65536].iter() {
        group.throughput(Throughput::Elements(*capacity as u64));
        
        group.bench_with_input(
            BenchmarkId::new("single_producer_single_consumer", capacity),
            capacity,
            |b, &cap| {
                b.iter(|| {
                    let mut queue = HighFrequencyTransactionQueue::new(cap);
                    let mut successful_pushes = 0;
                    
                    // Fill the queue
                    for i in 0..cap {
                        let tx = create_mock_streamed_transaction(i);
                        if queue.try_push(tx).is_ok() {
                            successful_pushes += 1;
                        }
                    }
                    
                    // Drain the queue
                    let mut successful_pops = 0;
                    while queue.try_pop().is_some() {
                        successful_pops += 1;
                    }
                    
                    successful_pushes + successful_pops
                });
            },
        );
    }
    
    // Multi-threaded stress test
    group.bench_function("multi_threaded_stress", |b| {
        b.iter(|| {
            let mut queue = HighFrequencyTransactionQueue::new(16384);
            let iterations = 1000;
            
            // Simulate concurrent producers
            (0..iterations).into_par_iter().for_each(|i| {
                let tx = create_mock_streamed_transaction(i);
                let _ = black_box(queue.try_push(tx));
            });
            
            // Drain in single thread
            let mut drained = 0;
            while queue.try_pop().is_some() {
                drained += 1;
            }
            drained
        });
    });
    
    group.finish();
}

fn bench_simd_signature_verification(c: &mut Criterion) {
    let mut group = c.benchmark_group("⚔️ SIMD Signature Verification");
    group.measurement_time(Duration::from_secs(15));
    
    let verifier = SIMDSignatureVerifier::new(32);
    
    for signature_count in [1, 4, 16, 64, 256, 1024].iter() {
        group.throughput(Throughput::Elements(*signature_count as u64));
        
        group.bench_with_input(
            BenchmarkId::new("batch_verification", signature_count),
            signature_count,
            |b, &count| {
                let signatures = create_mock_signatures(count);
                
                b.iter(|| {
                    let results = verifier.verify_signatures_batch(black_box(&signatures));
                    results.len()
                });
            },
        );
    }
    
    // Compare SIMD vs sequential
    let signatures_64 = create_mock_signatures(64);
    
    group.bench_function("simd_vs_sequential_64", |b| {
        b.iter(|| {
            // SIMD batch verification
            let simd_results = verifier.verify_signatures_batch(black_box(&signatures_64));
            
            // Sequential verification
            let sequential_results: Vec<bool> = signatures_64
                .iter()
                .map(|(pk, msg, sig)| verifier.verify_single_signature(pk, msg, sig))
                .collect();
            
            simd_results.len() + sequential_results.len()
        });
    });
    
    group.finish();
}

fn bench_connection_pool_failover(c: &mut Criterion) {
    let mut group = c.benchmark_group("🛡️ Connection Pool Failover");
    group.measurement_time(Duration::from_secs(8));
    
    let rt = Runtime::new().unwrap();
    
    group.bench_function("healthy_primary_selection", |b| {
        let primary_urls = vec![
            "https://api.mainnet-beta.solana.com".to_string(),
            "https://api.devnet.solana.com".to_string(),
        ];
        let backup_urls = vec!["https://api.testnet.solana.com".to_string()];
        let pool = SolanaConnectionPool::new(primary_urls, backup_urls);
        
        b.to_async(&rt).iter(|| async {
            // Simulate rapid client selection
            let mut selections = 0;
            for _ in 0..100 {
                if pool.get_client().await.is_ok() {
                    selections += 1;
                }
            }
            selections
        });
    });
    
    group.bench_function("failover_simulation", |b| {
        let primary_urls = vec!["http://invalid-url-1.com".to_string()];
        let backup_urls = vec!["https://api.devnet.solana.com".to_string()];
        let pool = SolanaConnectionPool::new(primary_urls, backup_urls);
        
        b.to_async(&rt).iter(|| async {
            // Simulate failover scenarios
            let mut successful_failovers = 0;
            for _ in 0..10 {
                if pool.get_client().await.is_ok() {
                    successful_failovers += 1;
                }
            }
            successful_failovers
        });
    });
    
    group.finish();
}

fn bench_end_to_end_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("💀 END-TO-END WARFARE");
    group.measurement_time(Duration::from_secs(20));
    
    let rt = Runtime::new().unwrap();
    
    group.bench_function("full_transaction_pipeline", |b| {
        let primary_urls = vec!["https://api.devnet.solana.com".to_string()];
        let backup_urls = vec![];
        let pool = std::sync::Arc::new(SolanaConnectionPool::new(primary_urls, backup_urls));
        
        b.to_async(&rt).iter(|| async {
            // Complete pipeline simulation
            let mut executor = HighFrequencyTransactionExecutor::new(
                std::sync::Arc::clone(&pool),
                1024,
                16
            );
            
            // Create mock transaction
            let mock_tx = create_mock_transaction();
            
            // Measure end-to-end latency (will fail on network but measure timing)
            let start = std::time::Instant::now();
            let _ = executor.execute_transaction(mock_tx).await;
            start.elapsed().as_micros()
        });
    });
    
    group.finish();
}

// === UTILITY FUNCTIONS ===

fn create_mock_streamed_transaction(id: usize) -> StreamedTransaction {
    StreamedTransaction {
        signature: format!("mock_signature_{}", id),
        slot: id as u64,
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64,
        accounts: vec![
            format!("account_1_{}", id),
            format!("account_2_{}", id),
        ],
        instruction_data: vec![0u8; 64],
        compute_units: Some(200_000),
        fee: 5000,
    }
}

fn create_mock_signatures(count: usize) -> Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    (0..count)
        .map(|i| {
            let public_key = vec![i as u8; 32];
            let message = format!("transaction_message_{}", i).into_bytes();
            let signature = vec![(i % 256) as u8; 64];
            (public_key, message, signature)
        })
        .collect()
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
        1_000_000, // 1 SOL in lamports
    );
    
    let message = Message::new(&[instruction], Some(&keypair.pubkey()));
    Transaction::new(&[&keypair], message, solana_sdk::hash::Hash::default())
}

criterion_group!(
    blockchain_benches,
    bench_websocket_streaming,
    bench_memory_mapped_buffer,
    bench_lock_free_queue,
    bench_simd_signature_verification,
    bench_connection_pool_failover,
    bench_end_to_end_latency,
);

criterion_main!(blockchain_benches);