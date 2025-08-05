//! Pipeline Latency Benchmark: Scalar vs. Quantum SIMD
//! Run with: cargo bench --bench pipeline_latency

use criterion::{criterion_group, criterion_main, Criterion, black_box};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use trenchbot_dex::{
    cpu::CpuPlatoon,
    transaction::Transaction,
    mev::MevOpportunity,
};

// ---- Helper: Generate Random Transactions ----
fn random_transactions(n: usize) -> Vec<Transaction> {
    let mut rng = StdRng::seed_from_u64(42);
    (0..n).map(|_| Transaction::random_with_rng(&mut rng)).collect()
}

fn bench_scalar(c: &mut Criterion) {
    let mut platoon = CpuPlatoon::new();
    let txs = random_transactions(8192);

    c.bench_function("Scalar/Legacy volley", |b| {
        b.iter(|| {
            for chunk in txs.chunks(8) {
                // Fallback to scalar if SIMD not available
                let batch: [Transaction; 8] = chunk.try_into().expect("len=8");
                black_box(platoon.scalar_fallback(&batch));
            }
        });
    });
}

fn bench_quantum_simd(c: &mut Criterion) {
    let mut platoon = CpuPlatoon::new();
    let txs = random_transactions(8192);

    c.bench_function("Quantum/AVX2 SIMD volley", |b| {
        b.iter(|| {
            for chunk in txs.chunks(8) {
                let batch: [Transaction; 8] = chunk.try_into().expect("len=8");
                black_box(platoon.detect_batch(&batch));
            }
        });
    });
}

// Optional: GPU path if you add CUDA support
// fn bench_gpu(c: &mut Criterion) { ... }

criterion_group!(benches, bench_scalar, bench_quantum_simd);
criterion_main!(benches);
