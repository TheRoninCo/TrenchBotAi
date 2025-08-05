// src/bread/mod.rs// 

pub enum CombatRation {
    KetchupPacket(f64),    // <$50 (Losses)
    HappyMeal(f64),         // $50-$500
    DoubleRation(f64),      // $500-$1K
    FieldSteak(f64),        // $1K-$5K
    MessHall(f64),          // $5K-$10K
    AdmiralFeast(f64),      // $10K-$50K
    KrakenHarvest(f64),     // $50K+
}

impl CombatRation {
    pub fn classify(profit: f64) -> String {
        match profit {
            p if p < 0 => format!("💔 KETCHUP PACKET (-${:.2})", p.abs()),
            p if p < 50.0 => format!("🍟 SHRIMP SNACK (${:.2})", p),
            p if p < 500.0 => format!("🍔 HAPPY MEAL (${:.2})", p),
            p if p < 1000.0 => format!("🍱 DOUBLE RATION (${:.2})", p),
            p if p < 5000.0 => format!("🥩 FIELD STEAK (${:.2})", p),
            p if p < 10000.0 => format!("🍽️ MESS HALL (${:.2})", p),
            p if p < 50000.0 => format!("🎖️ ADMIRAL'S FEAST (${:.2})", p),
            _ => format!("🐙 KRAKEN HARVEST (${:.2})", profit)
        }
    }
}