//! Utility functions
use anyhow::Result;

pub fn calculate_gas_price() -> Result<u64> {
    Ok(50_000)
}

pub fn format_sol_amount(lamports: u64) -> String {
    format!("{:.6} SOL", lamports as f64 / 1_000_000_000.0)
}

pub fn calculate_slippage_impact(amount: f64, liquidity: f64) -> f64 {
    // Simple slippage calculation
    (amount / liquidity).powi(2)
}
