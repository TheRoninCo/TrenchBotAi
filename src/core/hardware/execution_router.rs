//! Hardware-aware execution routing with tiered processing

use std::collections::HashMap;
use super::detection::{BatchOpportunityDetector, GpuDetector};
use crate::{
    transaction::Transaction,
    mev::MevOpportunity,
    user::UserPlan,
};

/// Configuration for execution tiers
#[derive(Debug, Clone)]
struct ExecutionConfig {
    batch_size: usize,
    gpu_allowed: bool,
    max_parallelism: usize,
}

/// Routes transactions to optimal execution backend
pub struct ExecutionRouter {
    detectors: Vec<Box<dyn BatchOpportunityDetector>>,
    tier_config: HashMap<UserPlan, ExecutionConfig>,
}

impl ExecutionRouter {
    /// Creates a new router with auto-detected hardware
    pub fn new() -> Self {
        let mut tier_config = HashMap::new();
        
        // Free tier: CPU-only with small batches
        tier_config.insert(UserPlan::Free, ExecutionConfig {
            batch_size: 4,
            gpu_allowed: false,
            max_parallelism: 2,
        });

        // Pro tier: GPU-enabled with full parallelism
        tier_config.insert(UserPlan::Pro, ExecutionConfig {
            batch_size: 8,
            gpu_allowed: true,
            max_parallelism: 8,
        });

        Self {
            detectors: Self::init_detectors(),
            tier_config,
        }
    }

    /// Initializes available detectors (GPU first if available)
    fn init_detectors() -> Vec<Box<dyn BatchOpportunityDetector>> {
        let mut detectors: Vec<Box<dyn BatchOpportunityDetector>> = Vec::with_capacity(2);
        
        #[cfg(feature = "cuda")]
        {
            if let Some(gpu) = GpuSquad::deploy() {
                detectors.push(Box::new(gpu));
            }
        }

        // CPU detector is always available
        detectors.push(Box::new(CpuPlatoon::new()));
        detectors
    }

    /// Executes transactions according to user's plan
    pub fn execute(&self, txs: &[Transaction], user: &UserPlan) -> Vec<MevOpportunity> {
        let config = self.tier_config
            .get(user)
            .unwrap_or_else(|| self.tier_config.get(&UserPlan::Free).unwrap());
        
        let detector = self.select_detector(config.gpu_allowed);
        let batch_size = detector.optimal_batch_size().min(config.batch_size);

        txs.chunks(batch_size)
            .flat_map(|chunk| {
                let mut batch = [Transaction::default(); 8];
                batch[..chunk.len()].copy_from_slice(chunk);
                detector.detect_batch(&batch)[..chunk.len()].to_vec()
            })
            .collect()
    }

    #[inline]
    fn select_detector(&self, allow_gpu: bool) -> &dyn BatchOpportunityDetector {
        if allow_gpu {
            self.detectors.first()
                .filter(|d| d.is_gpu())
                .unwrap_or_else(|| &*self.detectors.last().unwrap())
        } else {
            &*self.detectors.last().unwrap() // CPU is always last
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{transaction::MockTransaction, user::UserPlan};

    #[test]
    fn test_router_initialization() {
        let router = ExecutionRouter::new();
        assert!(!router.detectors.is_empty());
    }

    #[test]
    fn test_tier_configuration() {
        let router = ExecutionRouter::new();
        let free_config = router.tier_config.get(&UserPlan::Free).unwrap();
        assert!(!free_config.gpu_allowed);
        
        let pro_config = router.tier_config.get(&UserPlan::Pro).unwrap();
        assert!(pro_config.gpu_allowed);
    }

    #[test]
    fn test_execution_path() {
        let router = ExecutionRouter::new();
        let txs = vec![MockTransaction::default(); 10];
        
        // Test free tier
        let results = router.execute(&txs, &UserPlan::Free);
        assert_eq!(results.len(), txs.len());
        
        // Test pro tier
        let results = router.execute(&txs, &UserPlan::Pro);
        assert_eq!(results.len(), txs.len());
    }
}