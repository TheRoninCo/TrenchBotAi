//! 𝗞𝗜𝗟𝗟𝗙𝗘𝗘𝗗 𝗩3.𝟭 - 𝗣𝗥𝗢𝗗𝗨𝗖𝗧𝗜𝗢𝗡 𝗚𝗥𝗔𝗗𝗘
//! Quant-ready event tracking with nanosecond instrumentation
//! and zero-overhead GPU abstraction

use std::{
    sync::atomic::{AtomicU64, AtomicF64, Ordering},
    time::{Instant, Duration},
    num::NonZeroUsize,
};
use dashmap::{DashMap, TryReserveError};
use fxhash::{FxBuildHasher, FxHasher64};
use hdrhistogram::Histogram;
use crossbeam_channel::{Sender, bounded};
use serde::Serialize;
use prometheus::{Gauge, register_gauge};
use num_cpus;

// ---- Constants ----
const EVENT_CAPACITY: usize = 10_000;
const MAX_LATENCY_NS: u64 = 1_000_000; // 1ms
const DEFAULT_SHARDS: usize = 16;

// ---- Core Structures ----
#[repr(align(64))]
struct LeaderboardShard {
    entries: DashMap<String, f64, FxBuildHasher>,
    _pad: [u8; 64 - 8], // Cache padding
}

/// Primary killfeed engine
pub struct Killfeed {
    // Core state
    leaderboard: Vec<LeaderboardShard>,
    events: DashMap<String, KillEvent, FxBuildHasher>,
    
    // Analytics
    stats: Stats,
    latency: Histogram<u64>,
    anomaly_scores: DashMap<String, f32, FxBuildHasher>,
    
    // GPU bridge
    gpu_tx: Option<Sender<NormalizedEvent>>,
    
    // Prometheus
    prom_events: Gauge,
    prom_profit: Gauge,
}

// ---- Stats ----
#[derive(Debug)]
struct Stats {
    total_events: AtomicU64,
    total_profit: AtomicF64,
}

// ---- Event Types ----
#[derive(Debug, Clone, Serialize)]
pub struct KillEvent {
    pub wallet: String,
    pub profit: f64,
    pub weapon: WeaponType,
    pub chain: ChainType,
    pub timestamp: u64,
    pub features: [f32; 8],
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq)]
pub enum WeaponType {
    Sniper = 0,
    Grunt = 1,
    Sandwich = 2,
    Liquidator = 3,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq)]
pub enum ChainType {
    Ethereum = 0,
    Solana = 1,
    Arbitrum = 2,
}

// GPU-optimized event
#[derive(Debug, Clone)]
pub struct NormalizedEvent {
    pub wallet_hash: u64,
    pub profit: f32,
    pub weapon_idx: u8,
    pub chain_idx: u8,
    pub features: [f32; 8],
}

// ---- Builder ----
pub struct KillfeedBuilder {
    shard_count: Option<NonZeroUsize>,
    enable_gpu: bool,
    prometheus: bool,
}

impl KillfeedBuilder {
    pub fn new() -> Self {
        Self {
            shard_count: None,
            enable_gpu: false,
            prometheus: false,
        }
    }

    pub fn shard_count(mut self, count: NonZeroUsize) -> Self {
        self.shard_count = Some(count);
        self
    }

    pub fn enable_gpu(mut self) -> Self {
        self.enable_gpu = true;
        self
    }

    pub fn enable_prometheus(mut self) -> Self {
        self.prometheus = true;
        self
    }

    pub fn build(self) -> Result<Killfeed, KillfeedError> {
        let shard_count = self.shard_count
            .map(|n| n.get())
            .unwrap_or_else(|| num_cpus::get().max(DEFAULT_SHARDS));

        let mut killfeed = Killfeed {
            leaderboard: Vec::with_capacity(shard_count),
            events: DashMap::with_capacity_and_hasher(
                EVENT_CAPACITY,
                FxBuildHasher::default(),
            ),
            stats: Stats {
                total_events: AtomicU64::new(0),
                total_profit: AtomicF64::new(0.0),
            },
            latency: Histogram::new(3)?,
            anomaly_scores: DashMap::with_capacity_and_hasher(
                1000,
                FxBuildHasher::default(),
            ),
            gpu_tx: if self.enable_gpu {
                let (tx, _) = bounded(10_000);
                Some(tx)
            } else {
                None
            },
            prom_events: if self.prometheus {
                register_gauge!("killfeed_events_total", "Total events")?
            } else {
                Gauge::new("dummy", "Dummy")?
            },
            prom_profit: if self.prometheus {
                register_gauge!("killfeed_profit_total", "Total profit")?
            } else {
                Gauge::new("dummy", "Dummy")?
            },
        };

        // Initialize shards
        for _ in 0..shard_count {
            killfeed.leaderboard.push(LeaderboardShard {
                entries: DashMap::with_capacity_and_hasher(
                    EVENT_CAPACITY / shard_count,
                    FxBuildHasher::default(),
                ),
                _pad: [0; 64 - 8],
            });
        }

        Ok(killfeed)
    }
}

// ---- Core Implementation ----
impl Killfeed {
    /// Creates a new killfeed builder
    pub fn builder() -> KillfeedBuilder {
        KillfeedBuilder::new()
    }

    /// Logs a harpoon event (hot path)
    #[inline]
    pub fn log_harpoon(
        &self,
        wallet: &str,
        profit: f64,
        weapon: WeaponType,
        chain: ChainType,
    ) -> Result<(), KillfeedError> {
        let _timer = ScopedTimer::new(&self.latency);

        // 1. Update leaderboard
        let shard_idx = FxHasher64::hash(wallet.as_bytes()) as usize % self.leaderboard.len();
        self.leaderboard[shard_idx].entries
            .entry(wallet.to_string())
            .and_modify(|v| *v += profit)
            .or_insert(profit);

        // 2. Create and store event
        let event = KillEvent {
            wallet: wallet.to_string(),
            profit,
            weapon,
            chain,
            timestamp: fast_time::now_ms(),
            features: self.extract_features(profit, weapon, chain),
        };
        self.events.insert(nanoid::nanoid!(), event.clone());

        // 3. Send to GPU if enabled
        if let Some(tx) = &self.gpu_tx {
            tx.try_send(event.normalize())
                .map_err(KillfeedError::GpuSend)?;
        }

        // 4. Update stats
        self.stats.total_events.fetch_add(1, Ordering::Relaxed);
        self.stats.total_profit.fetch_add(profit, Ordering::Relaxed);
        self.prom_events.inc();
        self.prom_profit.add(profit);

        Ok(())
    }

    #[inline(always)]
    fn extract_features(&self, profit: f64, weapon: WeaponType, chain: ChainType) -> [f32; 8] {
        [
            profit as f32,
            weapon as u8 as f32,
            chain as u8 as f32,
            // ... additional features
        ]
    }

    // ---- Analytics API ----
    pub fn snapshot(&self) -> KillfeedSnapshot {
        KillfeedSnapshot {
            top_performers: self.top_performers(20),
            total_events: self.stats.total_events.load(Ordering::Relaxed),
            total_profit: self.stats.total_profit.load(Ordering::Relaxed),
            latency: self.latency_stats(),
        }
    }

    pub fn top_performers(&self, n: usize) -> Vec<(String, f64)> {
        let mut all: Vec<_> = self.leaderboard.iter()
            .flat_map(|shard| shard.entries.iter())
            .map(|e| (e.key().clone(), *e.value()))
            .collect();
        
        all.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        all.truncate(n);
        all
    }

    pub fn latency_stats(&self) -> LatencyStats {
        LatencyStats {
            p50: self.latency.value_at_quantile(0.5),
            p90: self.latency.value_at_quantile(0.9),
            p99: self.latency.value_at_quantile(0.99),
            max: self.latency.max(),
        }
    }
}

// ---- Normalization ----
impl KillEvent {
    #[inline]
    pub fn normalize(&self) -> NormalizedEvent {
        NormalizedEvent {
            wallet_hash: FxHasher64::hash(self.wallet.as_bytes()),
            profit: (self.profit / 100.0).clamp(-1.0, 1.0) as f32,
            weapon_idx: self.weapon as u8,
            chain_idx: self.chain as u8,
            features: self.features,
        }
    }
}

// ---- Instrumentation ----
pub struct ScopedTimer<'a> {
    start: Instant,
    histo: &'a Histogram<u64>,
}

impl<'a> ScopedTimer<'a> {
    #[inline(always)]
    pub fn new(histo: &'a Histogram<u64>) -> Self {
        Self { start: Instant::now(), histo }
    }
}

impl<'a> Drop for ScopedTimer<'a> {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed().as_nanos() as u64;
        self.histo.record(elapsed).unwrap();
    }
}

// ---- Data Exports ----
#[derive(Debug, Serialize)]
pub struct KillfeedSnapshot {
    pub top_performers: Vec<(String, f64)>,
    pub total_events: u64,
    pub total_profit: f64,
    pub latency: LatencyStats,
}

#[derive(Debug, Serialize)]
pub struct LatencyStats {
    pub p50: u64,
    pub p90: u64,
    pub p99: u64,
    pub max: u64,
}

// ---- Error Handling ----
#[derive(Debug, thiserror::Error)]
pub enum KillfeedError {
    #[error("GPU channel full")]
    GpuSend(#[from] crossbeam_channel::TrySendError<NormalizedEvent>),
    #[error("Histogram init failed")]
    HistogramInit(#[from] hdrhistogram::CreationError),
    #[error("Prometheus error")]
    Prometheus(#[from] prometheus::Error),
    #[error("Capacity exceeded")]
    Capacity(#[from] TryReserveError),
}