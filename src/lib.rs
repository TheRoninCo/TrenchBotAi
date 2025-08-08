use serde::{Deserialize, Serialize};
use thiserror::Error;
use figment::{Figment, providers::{Toml, Env, Format}};

pub mod sdk;
pub mod analytics;
pub mod strategies;
pub mod infrastructure;
pub mod modules;
pub mod types;
pub mod ai_engines;
pub mod war;
pub mod ui;
pub mod observability;
pub mod training;

/// Core configuration and error types
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "lowercase")]
pub enum FireMode {
    Cold,      // Recon only
    Warm,      // Simulated engagements
    Hot,       // Weapons hot
    Inferno,   // Full combat mode
}

impl Default for FireMode {
    fn default() -> Self {
        FireMode::Cold
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct SafetyProtocol {
    pub max_daily_loss: f64,
    pub position_cap: f64,
    pub gas_abort_threshold: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct WarChest {
    pub total_rounds: f64,
    pub round_size: f64,
    pub reserve_ammo: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct SectorRules {
    #[serde(default = "default_equity_weight")]
    pub equity_weight: f64,
    #[serde(default = "default_stable_weight")]
    pub stable_weight: f64,
}

fn default_equity_weight() -> f64 { 0.7 }
fn default_stable_weight() -> f64 { 0.3 }

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct Killswitch {
    pub active: bool,
    pub reason: String,
    pub timestamp: i64,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub enum FireStatus {
    Scouting,
    DryFire,
    Live,
    Apocalypse,
}

#[derive(Error, Debug)]
pub enum CombatError {
    #[error("Safety lock engaged: {0}")]
    SafetyLock(String),
    #[error("Ammo depleted: {0} rounds left")]
    AmmoFailure(f64),
    #[error("Target acquisition failed: {0}")]
    TargetError(String),
    #[error("Config load failed: {0}")]
    ConfigError(String),
}

#[derive(Serialize, Deserialize, Clone, Default)]
pub struct TrenchConfig {
    pub fire_control: FireMode,
    pub safety: SafetyProtocol,
    pub warchest: WarChest,
    #[serde(default)]
    pub allocation: SectorRules,
    #[serde(default)]
    pub killswitch: Killswitch,
}

impl TrenchConfig {
    pub fn arm(&mut self) -> Result<(), CombatError> {
        if self.killswitch.active {
            Err(CombatError::SafetyLock(format!(
                "Killswitch active: {}",
                self.killswitch.reason
            )))
        } else {
            Ok(())
        }
    }

    pub fn engage(&self) -> FireStatus {
        match self.fire_control {
            FireMode::Cold => FireStatus::Scouting,
            FireMode::Warm => FireStatus::DryFire,
            FireMode::Hot => FireStatus::Live,
            FireMode::Inferno => FireStatus::Apocalypse,
        }
    }

    pub fn load() -> Result<Self, CombatError> {
        Figment::new()
            .merge(Toml::file("trench.toml").nested())
            .merge(Env::prefixed("TRENCH_"))
            .extract()
            .map_err(|e| CombatError::ConfigError(e.to_string()))
    }
}