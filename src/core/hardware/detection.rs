//! Unified detection interface for CPU/GPU execution

use crate::{
    transaction::Transaction,
    mev::MevOpportunity,
};

/// Batch detection trait (implement for CPU/GPU)
pub trait BatchOpportunityDetector: Send + Sync {
    /// Processes 8 transactions in parallel
    fn detect_batch(&self, txs: &[Transaction; 8]) -> [MevOpportunity; 8];
    
    /// Human-readable engine name (for logs/metrics)
    fn engine_name(&self) -> &'static str;
    
    /// Whether this detector uses GPU acceleration
    fn is_gpu(&self) -> bool { false }
    
    /// Default batch size for optimal performance
    fn optimal_batch_size(&self) -> usize { 8 }
}

/// Marker trait for GPU implementations
pub trait GpuDetector: BatchOpportunityDetector {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transaction::MockTransaction;
    
    struct TestDetector;
    
    impl BatchOpportunityDetector for TestDetector {
        fn detect_batch(&self, _: &[Transaction; 8]) -> [MevOpportunity; 8] {
            [MevOpportunity::default(); 8]
        }
        
        fn engine_name(&self) -> &'static str {
            "TestDetector"
        }
    }
    
    #[test]
    fn test_detector_trait() {
        let detector = TestDetector;
        let txs = [MockTransaction::default(); 8];
        let results = detector.detect_batch(&txs);
        assert_eq!(results.len(), 8);
        assert!(!detector.is_gpu());
    }
}