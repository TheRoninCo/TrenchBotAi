use super::{
    config::{DetectionConfig, MevConfig},
    error::MevError,
    types::{MevScore, MevType},
    BundleFeatures, RpcSimulateTransactionResult,
};
use lazy_static::lazy_static;
use prometheus::{IntCounterVec, register_int_counter_vec};
use solana_sdk::transaction::Transaction;
use std::sync::Arc;
use tracing::{info, instrument, warn};

lazy_static! {
    static ref DETECTION_METRICS: IntCounterVec = register_int_counter_vec!(
        "mev_detections_total",
        "MEV detection counts by type",
        &["type", "detector"]
    ).unwrap();
}

pub trait MevDetector: Send + Sync {
    #[instrument(skip(self, txs, sim), fields(detector = std::any::type_name::<Self>()))]
    fn detect(
        &self,
        txs: &[Transaction],
        sim: &RpcSimulateTransactionResult,
        features: &BundleFeatures,
    ) -> Option<MevScore>;
}

pub struct DetectorPipeline {
    detectors: Vec<Arc<dyn MevDetector>>,
    fallback: Arc<dyn MevDetector>,
    config: DetectionConfig,
}

impl DetectorPipeline {
    pub fn new(config: &MevConfig) -> Self {
        Self {
            detectors: vec![
                Arc::new(SandwichDetector::new(&config.detection.sandwich)),
                Arc::new(StatisticalDetector::new(&config.detection.statistical)),
            ],
            fallback: Arc::new(StatisticalDetector::new(&config.detection.statistical)),
            config: config.detection.clone(),
        }
    }

    #[instrument(skip_all, fields(num_txs = txs.len()))]
    pub fn detect(
        &self,
        txs: &[Transaction],
        sim: &RpcSimulateTransactionResult,
        features: &BundleFeatures,
    ) -> Result<MevScore, MevError> {
        let score = self.detectors
            .iter()
            .find_map(|d| d.detect(txs, sim, features))
            .unwrap_or_else(|| {
                warn!("Falling back to statistical detector");
                self.fallback.detect(txs, sim, features).unwrap()
            });

        DETECTION_METRICS
            .with_label_values(&[score.ty.to_string().as_str(), std::any::type_name::<dyn MevDetector>()])
            .inc();

        Ok(score)
    }
}

// Detector implementations omitted for brevity