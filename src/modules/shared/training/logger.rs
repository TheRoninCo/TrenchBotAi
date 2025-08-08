// src/modules/shared/training/logger.rs
use {
    chrono::Utc,
    serde_json::json,
    std::{
        fs::OpenOptions,
        io::Write,
        path::Path,
        sync::{Arc, Mutex},
    },
    crate::modules::mev::types::{BundleFeatures, MevScore},
};

pub struct TrainingLogger {
    file: Arc<Mutex<std::fs::File>>,
}

impl TrainingLogger {
    pub fn new(path: impl AsRef<Path>) -> Self {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .expect("Failed to open training log file");
        
        Self {
            file: Arc::new(Mutex::new(file)),
        }
    }

    pub fn log_detection(&self, score: &MevScore, features: &BundleFeatures) {
        let obj = json!({
            "score": {
                "confidence": score.confidence,
                "risk": score.risk,
                "profit": score.expected_profit,
                "type": score.ty.to_string(),
            },
            "features": {
                "slot": features.slot,
                "tip": features.tip_lamports,
                "programs": features.programs,
            },
            "timestamp": Utc::now().timestamp_millis(),
        });

        let mut file = self.file.lock().unwrap();
        writeln!(file, "{}", obj).unwrap();
    }
}