//! CHEF'S KITCHEN - Dynamic optimization
pub struct Chef {
    recipes: [OptimizationRecipe; 3],
    current_recipe: AtomicUsize,
}

impl Chef {
    pub fn adjust_rations(&self, performance: f64) {
        let new_recipe = match performance {
            p if p < 0.0 => 0,  // RationCut
            p if p < 1.0 => 1,  // Standard
            _ => 2,             // FeastMode
        };
        self.current_recipe.store(new_recipe, Ordering::Release);
    }
}