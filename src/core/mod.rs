//! TRENCH COMMAND HQ
use crate::core::metrics::WhaleMetrics;
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
}

impl TrenchieHQ {
    pub async fn mobilize(config: &HardwareRations) -> Result<Self> {
        let grunts = CpuPlatoon::enlist(config.grunts)?;
        let snipers = GpuSquad::deploy(config.sniper_gear).await?;
        
        Ok(Self {
            grunts,
            snipers,
            medic: Medic::new(),
            recon: ReconTeam::scout(),
            quartermaster: GearCache::load()?,
            current_rank: AtomicRank::new(snipers.is_some()),
        })
    }

    pub fn engage(&self, targets: &[WhaleSignal]) -> BattleReport {
        let ammo = self.quartermaster.draw_ammo(targets);
        
        match self.current_rank.load() {
            Rank::Recruit => self.grunts.volley_fire(ammo),
            _ => self.snipers.as_ref()
                .and_then(|s| s.take_shot(ammo).ok())
                .unwrap_or_else(|| {
                    log::warn!("SNIPER MISSED! Grunts advancing...");
                    self.grunts.volley_fire(ammo)
                })
        }
    }
}