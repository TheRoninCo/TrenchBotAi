use prometheus::core::{AtomicI64, GenericCounter};
use serde::{Serialize, Deserialize};
use std::fmt;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum CombatRation {
    KetchupPacket,    // < $100
    ShrimpSnack,      // $100-$500
    HappyMeal,        // $500-$2k
    DoubleRation,     // $2k-$5k
    FieldSteak,       // $5k-$10k
    MessHall,         // $10k-$50k
    AdmiralsFeast,    // $50k-$100k
    KrakenHarvest,    // > $100k
}

impl CombatRation {
    pub fn classify(profit: f64) -> Self {
        match profit.abs() {
            p if p < 100.0 => Self::KetchupPacket,
            p if p < 500.0 => Self::ShrimpSnack,
            p if p < 2_000.0 => Self::HappyMeal,
            p if p < 5_000.0 => Self::DoubleRation,
            p if p < 10_000.0 => Self::FieldSteak,
            p if p < 50_000.0 => Self::MessHall,
            p if p < 100_000.0 => Self::AdmiralsFeast,
            _ => Self::KrakenHarvest,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::KetchupPacket => "Ketchup Packet",
            Self::ShrimpSnack => "Shrimp Snack",
            Self::HappyMeal => "Happy Meal",
            Self::DoubleRation => "Double Ration",
            Self::FieldSteak => "Field Steak",
            Self::MessHall => "Mess Hall",
            Self::AdmiralsFeast => "Admiral's Feast",
            Self::KrakenHarvest => "Kraken Harvest",
        }
    }

    pub fn emoji(&self) -> &'static str {
        match self {
            Self::KetchupPacket => "🥫",
            Self::ShrimpSnack => "🍤",
            Self::HappyMeal => "🍔",
            Self::DoubleRation => "🍱",
            Self::FieldSteak => "🥩",
            Self::MessHall => "🍲",
            Self::AdmiralsFeast => "🎖️",
            Self::KrakenHarvest => "🐙",
        }
    }

    pub fn threshold(&self) -> (f64, f64) {
        match self {
            Self::KetchupPacket => (0.0, 100.0),
            Self::ShrimpSnack => (100.0, 500.0),
            // ... others ...
        }
    }
}

// Prometheus integration
impl From<CombatRation> for &'static str {
    fn from(val: CombatRation) -> Self {
        val.label()
    }
}