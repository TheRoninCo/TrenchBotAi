use polars::prelude::*;
use std::path::PathBuf;

pub struct WolframExporter {
    output_dir: PathBuf,
}

impl WolframExporter {
    pub fn export_bundles(&self, df: &DataFrame) -> Result<PathBuf> {
        let path = self.output_dir.join(format!(
            "bundles_{}.parquet",
            chrono::Utc::now().timestamp()
        ));
        let mut file = std::fs::File::create(&path)?;
        ParquetWriter::new(&mut file).finish(df)?;
        Ok(path)
    }
}