// src/services/mev_service.rs
use trenchie_core::killfeed::{Killfeed, WeaponType, ChainType};

pub struct MevService {
    killfeed: Killfeed,
}

impl MevService {
    pub fn new(killfeed: Killfeed) -> Self {
        Self { killfeed }
    }

    pub fn handle_transaction(&self, tx: Transaction) -> anyhow::Result<()> {
        // ---- Core Event Logging ----
        self.killfeed.log_harpoon(
            &tx.wallet,
            tx.profit,
            WeaponType::Sniper,  // Determine from TX analysis
            ChainType::Ethereum,  // Chain detection logic
        )?;

        // Additional business logic...
        Ok(())
    }
}