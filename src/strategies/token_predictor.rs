use std::{collections::HashMap, path::Path, sync::Arc};
#[cfg(feature = "ai")]
use tract_onnx::prelude::*;
#[cfg(not(feature = "ai"))]
type RunnableModel<T> = ();

pub struct TokenPredictor {
    models: HashMap<String, RunnableModel<f32>>,
    fallback_model: RunnableModel<f32>,
}

impl TokenPredictor {
    pub async fn new(model_dir: &str) -> anyhow::Result<Self> {
        let mut models = HashMap::new();
        
        // Load token-specific models
        for entry in std::fs::read_dir(Path::new(model_dir).join("tokens"))? {
            let path = entry?.path();
            if let Some(token) = path.file_stem().and_then(|s| s.to_str()) {
                let model = tract_onnx::onnx()
                    .model_for_path(&path)?
                    .into_runnable()?;
                models.insert(token.to_string(), model);
            }
        }

        // Fallback for unknown tokens
        let fallback = tract_onnx::onnx()
            .model_for_path(Path::new(model_dir).join("fallback.wlnet"))?
            .into_runnable()?;

        Ok(Self { models, fallback_model: fallback })
    }

    pub fn predict(
        &self,
        token: &str,
        personality: f32,
        velocity: f32,
        imbalance: f32,
        tip: f32,
        cu: f32,
    ) -> anyhow::Result<f32> {
        #[cfg(feature = "ai")]
        let input = ndarray::arr1(&[personality, velocity, imbalance, tip, cu]).into_dyn().into();
        #[cfg(not(feature = "ai"))]
        let input = ();
        
        #[cfg(feature = "ai")]
        match self.models.get(token) {
            Some(model) => Ok(model.run(tvec!(input))?[0].to_scalar::<f32>()?),
            None => Ok(self.fallback_model.run(tvec!(input))?[0].to_scalar::<f32>()?),
        }
        
        #[cfg(not(feature = "ai"))]
        Ok(0.5) // Fallback prediction
    }
}