use tract_onnx::prelude::*;
use serde_json::json;

pub struct ProfitPredictor {
    model: RunnableModel<f32>,
    rpc: Arc<RpcClient>,
}

impl ProfitPredictor {
    pub async fn new(model_path: &str, rpc: Arc<RpcClient>) -> Self {
        let model = tract_onnx::onnx()
            .model_for_path(model_path)
            .unwrap()
            .into_runnable()
            .unwrap();
        
        Self { model, rpc }
    }

    pub async fn predict(
        &self,
        bundle: &Bundle,
        whale: Option<&Whale>,
        pool_imbalance: f32,
    ) -> anyhow::Result<f32> {
        let personality = match whale {
            Some(w) => match w.personality {
                Personality::Gorilla => 1.0,
                Personality::Shark => 2.0,
                _ => 0.0,
            },
            None => 0.0,
        };

        let velocity = whale.map(|w| w.velocity).unwrap_or(0.0);
        let input = tract_ndarray::arr1(&[
            personality,
            velocity,
            pool_imbalance,
            bundle.tip_lamports as f32,
            bundle.cu_limit as f32,
        ]).into();

        let output = self.model.run(tvec!(input))?;
        Ok(output[0].to_scalar::<f32>()?)
    }
}