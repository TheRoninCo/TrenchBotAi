use std::path::Path;
use memmap2::Mmap;
use serde::de::DeserializeOwned;

pub fn parse_toml<T: DeserializeOwned>(path: &Path) -> anyhow::Result<T> {
    let file = std::fs::File::open(path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    
    // Zero-copy parsing from memory map
    let config = toml::from_slice(&mmap)?;
    Ok(config)
}

pub fn load_all_configs() -> anyhow::Result<TrenchConfig> {
    Ok(TrenchConfig {
        mev: parse_toml("configs/mev.toml")?,
        sonar: parse_toml("configs/sonar.toml")?,
        whale: parse_toml("configs/whale.toml")?,
        sinks: parse_toml("configs/sinks.toml")?,
        loaded_from: None,
    })
}