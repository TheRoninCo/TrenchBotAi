//! TRENCH COMMAND HQ
use crate::core::{
    hardware::{execution_router::ExecutionRouter, detection::BatchOpportunityDetector},
    metrics::WhaleMetrics,
};
use battle::{BattleReport, Medic};
use quartermaster::GearCache;
use scout::ReconReport;

pub struct TrenchieHQ {
    pub grunts: CpuPlatoon,
    pub snipers: Option<GpuSquad>,
    pub medic: Medic,
    pub recon: ReconTeam,
    pub quartermaster: GearCache,
    pub current_rank: AtomicRank,
    pub executor: ExecutionRouter, // NEW: Unified execution
}

impl TrenchieHQ {
    pub async fn mobilize(config: &HardwareRations) -> Result<Self> {
        let grunts = CpuPlatoon::enlist(config.grunts)?;
        let snipers = GpuSquad::deploy(config.sniper_gear).await?;
        let executor = ExecutionRouter::new(); // NEW
        
        Ok(Self {
            grunts,
            snipers,
            medic: Medic::new(),
            recon: ReconTeam::scout(),
            quartermaster: GearCache::load()?,
            current_rank: AtomicRank::new(snipers.is_some()),
            executor, // NEW
        })
    }

    pub fn engage(&self, targets: &[WhaleSignal], user_plan: UserPlan) -> BattleReport {
        // NEW: Unified execution path
        let opportunities = self.executor.execute(targets, &user_plan);
        
        // Existing battle logic
        let ammo = self.quartermaster.draw_ammo(targets);
        match self.current_rank.load() {
            Rank::Recruit => self.grunts.volley_fire(ammo),
            _ => self.snipers.as_ref()
                .and_then(|s| s.take_shot(ammo).ok()
                .unwrap_or_else(|| {
                    log::warn!("SNIPER MISSED! Grunts advancing...");
                    self.grunts.volley_fire(ammo)
                })
        }
    }
}

// Transaction trait should be in its own module (e.g. src/core/transaction.rs)
pub mod transaction {
    pub trait Transaction {
        fn detect_arbitrage(&self) -> bool;
        fn detect_liquidation(&self) -> bool;
        // ... other detection methods
    }
}

// Re-exports at module root (src/core/mod.rs)
pub mod hardware {
    pub mod detection;
    pub mod execution_router;
    pub use execution_router::ExecutionRouter;
}//! TRENCH COMMAND HQ

pub struct TrenchieHQ {
    // YOUR EXISTING FIELDS
    pub grunts: CpuPlatoon,
    pub snipers: Option<GpuSquad>,
    pub medic: Medic,
    pub recon: ReconTeam,
    pub quartermaster: GearCache,
    pub current_rank: AtomicRank,
    
    // NEW FIELD
    pub executor: ExecutionRouter,
}

impl TrenchieHQ {
    pub async fn mobilize(config: &HardwareRations) -> Result<Self> {
        // ... your existing initialization ...
        
        Ok(Self {
            // ... your existing fields ...
            executor: ExecutionRouter::new(), // NEW
        })
    }

    pub fn engage(&self, targets: &[Transaction], user: &UserPlan) -> BattleReport {
        // NEW: Detection phase
        let opportunities = self.executor.execute(targets, user);
        
        // YOUR EXISTING BATTLE LOGIC
        let ammo = self.quartermaster.draw_ammo(targets);
        match self.current_rank.load() {
            Rank::Recruit => self.grunts.volley_fire(ammo),
            _ => self.snipers.as_ref()
                .and_then(|s| s.take_shot(ammo).ok())
                .unwrap_or_else(|| self.grunts.volley_fire(ammo))
        }
    }
}