//! Intelligent gas price optimization
pub struct GasOptimizer {
    pub base_price: u64,
    pub urgency_multiplier: f64,
    pub competition_factor: f64,
}

impl GasOptimizer {
    pub fn calculate_optimal_gas(&self, opportunity_value: f64, competition_level: u8) -> u64 {
        // Your gas optimization algorithm
        // Critical for sandwich attack profitability
        self.base_price
    }
}
