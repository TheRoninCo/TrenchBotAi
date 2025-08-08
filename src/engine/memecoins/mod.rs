pub mod momentum;
pub mod whale;
pub mod sniper;

pub mod momentum {
    use crate::types::Opportunity;
    use anyhow::Result;
    
    pub async fn scan_momentum_opportunities() -> Result<Vec<Opportunity>> {
        Ok(vec![])
    }
}

pub mod whale {
    use crate::types::AlphaWalletSignal;
    use anyhow::Result;
    
    pub async fn track_alpha_wallets() -> Result<Vec<AlphaWalletSignal>> {
        Ok(vec![])
    }
}

pub mod sniper {
    use crate::types::Opportunity;
    use anyhow::Result;
    
    pub async fn snipe_new_listings() -> Result<Vec<Opportunity>> {
        Ok(vec![])
    }
}
