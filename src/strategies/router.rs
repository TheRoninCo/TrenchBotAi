impl MevRouter {
    pub fn handle_profit(&self, profit: f64) -> ControlFlow {
        match CombatRation::classify(profit) {
            CombatRation::AdmiralsFeast | CombatRation::KrakenHarvest => {
                ControlFlow::Aggressive
            },
            CombatRation::MessHall => ControlFlow::Normal,
            _ => ControlFlow::Skip,
        }
    }
}