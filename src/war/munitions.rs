// src/war/munitions.rs
fn fire_weapon(target: Target) -> Result<()> {
    observability::log(CombatLog {
        system: "munitions".into(),
        event_type: "plasma_shot".into(),
        data: json!({ "target": target.id, "energy": target.energy }),
        ..Default::default()
    })?;
}