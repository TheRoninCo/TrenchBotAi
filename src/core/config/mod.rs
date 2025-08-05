//! Battle-tested configuration management for MEV warfare
use serde::{Deserialize, Serialize};
use validator::Validate;
use anyhow::{Result, Context};
use std::{env, path::PathBuf};
use figment::{Figment, providers::{Format, Toml, Env, Serialized}};

/// Root configuration container
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct Config {
    #[validate]
    pub trading: TradingConfig,
    
    #[validate]
    pub hardware: HardwareConfig,
    
    #[validate]
    pub network: NetworkConfig,
    
    #[validate]
    pub sandwich: SandwichConfig,
    
    #[validate]
    pub development: DevelopmentConfig,
    
    #[serde(skip)]
    pub config_path: Option<PathBuf>,
}

impl Config {
    /// Load config from multiple sources with validation
    pub fn load() -> Result<Self> {
        let base_path = env::var("CONFIG_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("configs/mev.toml"));
        
        let config: Self = Figment::new()
            .merge(Serialized::defaults(Config::default()))
            .merge(Toml::file(&base_path))
            .merge(Env::prefixed("TRENCH_"))
            .extract()
            .context("Failed to load config")?;
            
        config.validate()
            .context("Config validation failed")?;
            
        Ok(Self {
            config_path: Some(base_path),
            ..config
        })
    }
    
    /// Reload configuration (for hot-reloading)
    pub fn reload(&self) -> Result<Self> {
        Self::load()
    }
}

/// Trading parameters
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct TradingConfig {
    #[validate(range(min = 0.01, max = 1.0))]
    pub max_position_size: f64,
    
    #[validate(range(min = 0.0, max = 0.1))]
    pub slippage_tolerance: f64,
    
    #[validate(range(min = 1, max = 1000))]
    pub gas_price_gwei: u64,
    
    #[validate(range(min = 0.0))]
    pub min_profit_threshold: f64,
}

/// Hardware acceleration settings
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct HardwareConfig {
    pub enable_gpu: bool,
    pub enable_fpga: bool,
    
    #[validate(range(min = 0))]
    pub gpu_device_id: Option<u32>,
    
    #[validate(range(min = 1, max = 64))]
    pub cpu_threads: Option<usize>,
}

/// Network connectivity
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct NetworkConfig {
    #[validate(url)]
    pub rpc_url: String,
    
    #[validate(url)]
    pub ws_url: String,
    
    #[validate(length(min = 32))]
    pub helius_api_key: String,
    
    pub dev_mode: bool,
}

/// Sandwich attack parameters
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct SandwichConfig {
    #[validate(range(min = 0.0))]
    pub min_profit_threshold: f64,
    
    #[validate(range(min = 0.0))]
    pub max_position_size: f64,
    
    #[validate(range(min = 0.0, max = 0.5))]
    pub slippage_tolerance: f64,
    
    #[validate(range(min = 1.0, max = 5.0))]
    pub gas_multiplier: f64,
    
    #[validate(range(min = 0.5, max = 1.0))]
    pub confidence_threshold: f64,
    
    #[validate(range(min = 0.0))]
    pub max_front_run_amount: f64,
}

/// Development settings
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct DevelopmentConfig {
    pub simulate_data: bool,
    
    #[validate(contains = ["trace", "debug", "info", "warn", "error"])]
    pub log_level: String,
    
    pub mock_trades: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            trading: TradingConfig {
                max_position_size: 1000.0,
                slippage_tolerance: 0.01,
                gas_price_gwei: 20,
                min_profit_threshold: 0.05,
            },
            hardware: HardwareConfig {
                enable_gpu: false,
                enable_fpga: false,
                gpu_device_id: None,
                cpu_threads: Some(4),
            },
            network: NetworkConfig {
                rpc_url: "https://api.devnet.solana.com".to_string(),
                ws_url: "wss://api.devnet.solana.com".to_string(),
                helius_api_key: "".to_string(),
                dev_mode: true,
            },
            sandwich: SandwichConfig {
                min_profit_threshold: 0.05,
                max_position_size: 500.0,
                slippage_tolerance: 0.02,
                gas_multiplier: 1.5,
                confidence_threshold: 0.7,
                max_front_run_amount: 100.0,
            },
            development: DevelopmentConfig {
                simulate_data: true,
                log_level: "info".to_string(),
                mock_trades: true,
            },
            config_path: None,
        }
    }
}