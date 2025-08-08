use std::ops::ControlFlow;
use crate::modules::shared::classifier::CombatRation;

pub struct MevRouter;

impl MevRouter {
    pub fn handle_profit(&self, profit: f64) -> ControlFlow<(), ()> {
        match CombatRation::classify(profit) {
            CombatRation::AdmiralsFeast | CombatRation::KrakenHarvest => {
                ControlFlow::Continue(())
            },
            CombatRation::MessHall => ControlFlow::Continue(()),
            _ => ControlFlow::Break(()),
        }
    }
}