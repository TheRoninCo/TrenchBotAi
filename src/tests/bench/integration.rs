// tests/integration.rs
#[test]
fn test_cross_engine_consistency() {
    let txs = test_transactions();
    let cpu_results = CpuPlatoon::new().detect_batch(&txs);
    
    #[cfg(feature = "cuda")]
    {
        let gpu_results = CudaDetector::new().detect_batch(&txs);
        assert_eq!(cpu_results, gpu_results);
    }
}