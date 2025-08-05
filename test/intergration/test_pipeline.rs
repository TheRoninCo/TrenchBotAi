// Place in: tests/integration/test_pipeline.rs
#[cfg(test)]
mod integration {
    use crate::{
        core::hardware::cpu::CpuPlatoon,
        modules::mev::analyzer,
        shared::metrics::{self, MetricRecorder},
        transaction::MockTransaction,
    };

    #[test]
    fn test_hardware_module_metrics() {
        // Step 1: Simulate 32 test transactions (replace with real/mock as needed)
        let txs: Vec<MockTransaction> = (0..32).map(|_| MockTransaction::default()).collect();

        // Step 2: Route through CPU hardware (SIMD/scalar)
        let cpu = CpuPlatoon::new();
        let batch: &[MockTransaction; 8] = &txs[0..8].try_into().unwrap();
        let opportunities = cpu.detect_batch(batch);

        // Step 3: Analyze via MEV analyzer module (replace with your actual func)
        let mev_signals = analyzer::analyze_opportunities(&opportunities);

        // Step 4: Record metrics via shared/metrics trait/logic
        metrics::record_opportunity_metrics(&mev_signals);

        // Step 5: Assert: are metrics in the global registry?
        let registry = prometheus::default_registry();
        let metric_families = registry.gather();
        let found = metric_families.iter().any(|fam| fam.get_name() == "mev_opportunity_count");

        assert!(found, "MEV opportunity metrics not exported to global Prometheus registry");
    }
}
