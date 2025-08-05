//! GEAR MANAGEMENT - Hot reload configs
use notify::{RecommendedWatcher, Watcher};

pub struct Quartermaster {
    watcher: RecommendedWatcher,
    current_rations: Arc<RwLock<HardwareRations>>,
}

impl Quartermaster {
    pub fn watch(config_path: &str) -> Result<Self> {
        let rations = Arc::new(RwLock::new(load_rations(config_path)?));
        let watcher = notify::recommended_watcher({
            let rations = rations.clone();
            move |res| {
                if let Ok(event) = res {
                    if event.kind.is_modify() {
                        reload_rations(&rations, &event.paths);
                    }
                }
            }
        })?;
        
        Ok(Self { watcher, current_rations: rations })
    }
}