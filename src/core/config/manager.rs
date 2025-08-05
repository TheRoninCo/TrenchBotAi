use notify::{RecommendedWatcher, Watcher};
use std::{path::PathBuf, sync::Arc};
use tokio::sync::RwLock;
use super::Config;

pub struct ConfigManager {
    pub current: Arc<RwLock<Config>>,
    watcher: RecommendedWatcher,
}

use notify::{RecommendedWatcher, Watcher, EventKind};
use std::sync::Arc;
use parking_lot::RwLock;
use crossbeam::channel::{Sender, unbounded};

pub struct ConfigManager {
    current: Arc<RwLock<TrenchConfig>>,
    watcher: RecommendedWatcher,
    change_tx: Sender<ConfigUpdate>,
}

impl ConfigManager {
    pub fn new() -> anyhow::Result<Self> {
        let config = Arc::new(RwLock::new(load_all_configs()?));
        let (change_tx, change_rx) = unbounded();
        
        let mut watcher = RecommendedWatcher::new(
            move |res| {
                if let Ok(event) = res {
                    if matches!(event.kind, EventKind::Modify(_)) {
                        if let Err(e) = handle_config_change(&config, &change_tx) {
                            error!("Config reload failed: {}", e);
                        }
                    }
                }
            },
            notify::Config::default()
        )?;
        
        // Watch all config files
        watcher.watch("configs/mev.toml", RecursiveMode::NonRecursive)?;
        watcher.watch("configs/sonar.toml", RecursiveMode::NonRecursive)?;
        // ... other files
        
        Ok(Self { current: config, watcher, change_tx })
    }
    
    pub fn current(&self) -> Arc<RwLock<TrenchConfig>> {
        self.current.clone()
    }
    
    pub fn subscribe(&self) -> crossbeam::channel::Receiver<ConfigUpdate> {
        self.change_tx.subscribe()
    }
}

fn handle_config_change(
    config: &Arc<RwLock<TrenchConfig>>,
    tx: &Sender<ConfigUpdate>
) -> anyhow::Result<()> {
    let new_config = load_all_configs()?;
    let mut current = config.write();
    
    // Atomic swap of entire config
    *current = new_config;
    
    tx.send(ConfigUpdate::Reloaded)?;
    Ok(())
}

#[derive(Debug)]
pub enum ConfigUpdate {
    Reloaded,
    PartialUpdate(&'static str), // Module name
    Error(String),
}impl ConfigManager {
    pub async fn new() -> anyhow::Result<Self> {
        let config = Arc::new(RwLock::new(Config::load()?));
        let config_path = config.read().await.config_path.clone();
        
        let watcher = if let Some(path) = config_path {
            let mut watcher = notify::recommended_watcher({
                let config = config.clone();
                move |res| {
                    if let Ok(event) = res {
                        if event.kind.is_modify() {
                            let _ = Self::reload_config(&config);
                        }
                    }
                }
            })?;
            
            watcher.watch(&path, notify::RecursiveMode::NonRecursive)?;
            Some(watcher)
        } else {
            None
        };

        Ok(Self {
            current: config,
            watcher: watcher.unwrap(),
        })
    }

    async fn reload_config(config: &Arc<RwLock<Config>>) -> anyhow::Result<()> {
        let new_config = Config::load()?;
        *config.write().await = new_config;
        Ok(())
    }
}