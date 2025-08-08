pub mod sandwich;
pub mod arbitrage;
pub mod liquidation;

pub mod arbitrage {
    use crate::types::Opportunity;
    use anyhow::Result;
    
    pub async fn scan_arbitrage_opportunities() -> Result<Vec<Opportunity>> {
        Ok(vec![])
    }
}

pub mod liquidation {
    use crate::types::Opportunity;
    use anyhow::Result;
    
    pub async fn scan_liquidation_opportunities() -> Result<Vec<Opportunity>> {
        Ok(vec![])
    }
}
