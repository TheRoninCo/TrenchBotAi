//! TRENCHIE WAR ROOM CORE
//! Combines branchless execution with real-time analytics and monitoring

use std::sync::{Arc, atomic::{AtomicU64, Ordering}};
use dashmap::DashMap;
use serde::{Serialize, Deserialize};
use prometheus::{Registry, IntCounter, Gauge};
pub mod killfeed;
pub use killfeed::{Killfeed, KillEvent, WeaponType};
// Hardware subsystems
pub mod hardware {
    pub mod branchless_volley;
    pub mod medic;
    pub mod bunker;
    pub mod killfeed;
    pub mod performance;
    pub mod api;
}

// Re-export key components
pub use hardware::{
    branchless_volley::execute,
    performance::{PerformanceStats, log_report_to_file},
    api::build_api
};

/// Central combat coordination hub
#[repr(C, align(64))]
pub struct BattleStation {
    /// Branchless execution engine
    pub executor: Arc<hardware::branchless_volley::Executor>,
    /// Performance analytics
    pub stats: Arc<PerformanceStats>,
    /// Bunker protocol
    pub bunker: hardware::bunker::BunkerProtocol,
    /// Kill tracking
    pub killfeed: hardware::killfeed::Killfeed,
    /// Error recovery
    pub medic: hardware::medic::Medic,
    /// Metrics registry
    registry: Registry,
}

impl BattleStation {
    pub fn new() -> Self {
        let registry = Registry::new();
        let stats = Arc::new(PerformanceStats::new(&registry));
        
        BattleStation {
            executor: Arc::new(hardware::branchless_volley::Executor::new()),
            stats: stats.clone(),
            bunker: hardware::bunker::BunkerProtocol::new(3),
            killfeed: hardware::killfeed::Killfeed::new(),
            medic: hardware::medic::Medic::new(),
            registry,
        }
    }

    /// Execute engagement with full analytics
    pub fn engage(&self, targets: &[Transaction]) -> BattleReport {
        let report = self.executor.execute(
            &self.bunker,
            &self.killfeed,
            &self.medic,
            targets,
            32, // Optimal cache-aligned batch size
        );

        // Record analytics
        self.stats.record_battle(&report);
        log_report_to_file(&report);
        
        // Update prometheus metrics
        self.update_metrics(&report);
        
        report
    }

    /// Build API router with current stats
    pub fn build_api(&self) -> axum::Router {
        hardware::api::build_api(self.stats.clone())
    }

    fn update_metrics(&self, report: &BattleReport) {
        // Metrics updated automatically through PerformanceStats
    }
}

/// Extended BattleReport with analytics tags
#[derive(Serialize, Deserialize, Clone)]
pub struct BattleReport {
    pub confirmed_kills: u32,
    pub profit: f64,
    pub rekt: u32,
    pub tags: Vec<BattleTag>,
    pub timestamp: DateTime<Utc>,
}

#[derive(Serialize, Deserialize, Clone)]
pub enum BattleTag {
    WhaleHarpoon,
    SandwichAttack,
    Frontrun,
    Backrun,
    Liquidated,
}

/// Transaction type with MEV signals
#[derive(Clone)]
pub struct Transaction {
    pub value: f64,
    pub mev_type: MevType,
    pub origin: String,
}

pub enum MevType {
    Arbitrage,
    Liquidation,
    Sandwich,
    Frontrun,
    Backrun,
}