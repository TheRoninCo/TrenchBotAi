// mem.rs
impl AmmoDump {
    pub fn issue_ammo(&self, targets: &[Transaction]) -> Vec<AmmoCrate> {
        targets.iter().map(|tx| {
            match tx.classify() {
                Heavy => self.artillery.alloc(tx),
                Medium => self.medium_cache.alloc(tx),
                Light => self.small_arms.alloc(tx),
            }
        }).collect()
    }
}