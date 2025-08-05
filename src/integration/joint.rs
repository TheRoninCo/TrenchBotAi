use serde_json::from_str;
use std::path::Path;

pub struct JointStrategy {
    wolfram_model_path: String,
    last_params: Mutex<JointParams>,
}

#[derive(Serialize, Deserialize)]
struct JointParams {
    gorilla_tip_multiplier: f64,
    shark_cu_boost: u32,
}

impl JointStrategy {
    pub fn new(model_path: &str) -> Self {
        Self {
            wolfram_model_path: model_path.to_string(),
            last_params: Mutex::new(JointParams {
                gorilla_tip_multiplier: 1.2,
                shark_cu_boost: 50_000,
            }),
        }
    }

    pub async fn reload_params(&self) -> anyhow::Result<()> {
        let json = tokio::fs::read_to_string(&self.wolfram_model_path).await?;
        *self.last_params.lock().await = from_str(&json)?;
        Ok(())
    }

    pub fn adjust_for_whale(&self, bundle: &mut Bundle, whale: Option<&Whale>) {
        let params = self.last_params.lock().unwrap();
        if let Some(w) = whale {
            match w.personality {
                Personality::Gorilla => {
                    bundle.tip_lamports = (bundle.tip_lamports as f64 * params.gorilla_tip_multiplier) as u64;
                }
                Personality::Shark => {
                    bundle.cu_limit += params.shark_cu_boost;
                }
                _ => {}
            }
        }
    }
}