// src/modules/shared/training/wolfram.rs
use serde::Serialize;
use std::path::Path;

pub struct WolframExporter;

impl WolframExporter {
    pub fn export_mev_score(path: impl AsRef<Path>, score: &super::MevScore) -> anyhow::Result<()> {
        let file = std::fs::File::create(path)?;
        let mut writer = std::io::BufWriter::new(file);
        wolfram_wxf::encode_to_writer(&score, &mut writer)?;
        Ok(())
    }

    pub fn export_features(path: impl AsRef<Path>, features: &super::BundleFeatures) -> anyhow::Result<()> {
        let file = std::fs::File::create(path)?;
        let mut writer = std::io::BufWriter::new(file);
        wolfram_wxf::encode_to_writer(&features, &mut writer)?;
        Ok(())
    }
}