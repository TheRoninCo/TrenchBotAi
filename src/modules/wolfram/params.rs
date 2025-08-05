use serde_json::from_str;
use std::fs;

#[derive(Deserialize)]
pub struct TipModelParams {
    pub optimal_tip_lamports: u64,
    pub cu_limit: u32,
    pub success_probability: f32,
}

pub fn load_latest_params() -> anyhow::Result<TipModelParams> {
    let latest_file = fs::read_dir("model_outputs")?
        .filter_map(|e| e.ok())
        .max_by_key(|e| e.metadata().unwrap().modified().unwrap())
        .ok_or(anyhow!("No model files found"))?;

    let contents = fs::read_to_string(latest_file.path())?;
    Ok(from_str(&contents)?)
}