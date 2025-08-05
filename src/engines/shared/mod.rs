pub mod filters;
pub mod scoring;

pub mod filters {
    use crate::types::Opportunity;
    use anyhow::Result;
    
    pub struct FilterPipeline;
    
    impl FilterPipeline {
        pub fn new() -> Self { Self }
        
        pub async fn process(&self, opportunities: Vec<Opportunity>) -> Result<Vec<Opportunity>> {
            Ok(opportunities)
        }
    }
}

pub mod scoring {
    use crate::types::Opportunity;
    
    pub fn calculate_opportunity_score(opportunity: &Opportunity) -> f64 {
        opportunity.confidence * opportunity.expected_profit
    }
}
