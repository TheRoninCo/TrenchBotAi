//! Hybrid CPU Execution Engine - SIMD Quantum & Scalar Fallback
use std::sync::Arc;
use super::detection::BatchOpportunityDetector;
use crate::{
    transaction::Transaction,
    mev::MevOpportunity,
    battle::{BattleReport, AmmoCrate},
};
use rayon::prelude::*;

const QUANTUM_LANE_WIDTH: usize = 8; // AVX2/AVX-512 alignment
const GRUNT_SQUAD_SIZE: usize = 4;   // Optimal chunk size for scalar

pub struct CpuPlatoon {
    quantum_engine: Option<Arc<QuantumCore>>,
    grunt_squads: Vec<GruntSquad>,
    work_pool: TaskDistributor,
}

impl CpuPlatoon {
    /// Initializes platoon with available hardware capabilities
    pub fn new() -> Self {
        Self {
            quantum_engine: QuantumCore::probe().map(Arc::new),
            grunt_squads: (0..num_cpus::get()).map(GruntSquad::new).collect(),
            work_pool: TaskDistributor::new("CPU-Workers").unwrap(),
        }
    }

    /// LEGACY: Parallel scalar processing (via Rayon)
    pub fn volley_fire(&self, ammo: &[AmmoCrate]) -> BattleReport {
        self.work_pool.distribute(|| {
            ammo.par_chunks(GRUNT_SQUAD_SIZE)
                .map(|crate| analyze_whale_movement(crate))
                .reduce(|| BattleReport::default(), |a, b| a.merge(b))
        })
    }
}

// ========= SIMD Quantum Path =========
#[cfg(target_feature = "avx2")]
impl BatchOpportunityDetector for CpuPlatoon {
    fn detect_batch(&self, txs: &[Transaction; 8]) -> [MevOpportunity; 8] {
        self.quantum_engine.as_ref()
            .map(|q| q.quantum_scan(txs))
            .unwrap_or_else(|| self.scalar_fallback(txs))
    }

    fn engine_name(&self) -> &'static str {
        self.quantum_engine.as_ref()
            .map(|_| "CPU (Quantum/AVX2)")
            .unwrap_or("CPU (Scalar Fallback)")
    }
}

#[cfg(target_feature = "avx2")]
impl CpuPlatoon {
    #[inline(always)]
    fn scalar_fallback(&self, txs: &[Transaction; 8]) -> [MevOpportunity; 8] {
        txs.map(|tx| MevOpportunity {
            arb: tx.detect_arbitrage_scalar(),
            liq: tx.detect_liquidation_scalar(),
            sandwich: tx.detect_sandwich_scalar(),
        })
    }
}

// ========= Scalar-Only Path =========
#[cfg(not(target_feature = "avx2"))]
impl BatchOpportunityDetector for CpuPlatoon {
    fn detect_batch(&self, txs: &[Transaction; 8]) -> [MevOpportunity; 8] {
        self.scalar_fallback(txs)
    }

    fn engine_name(&self) -> &'static str {
        "CPU (Scalar)"
    }
}

// ======= Quantum Core (Private) ======
struct QuantumCore {
    #[cfg(target_feature = "avx2")]
    execution_lanes: [SimdExecutionLane; 8],
}

impl QuantumCore {
    fn probe() -> Option<Self> {
        #[cfg(target_feature = "avx2")]
        {
            Some(Self {
                execution_lanes: SimdExecutionLane::initialize()?,
            })
        }
        #[cfg(not(target_feature = "avx2"))]
        None
    }

    #[cfg(target_feature = "avx2")]
    fn quantum_scan(&self, txs: &[Transaction; 8]) -> [MevOpportunity; 8] {
        unsafe {
            let arb = simd_detect_arbitrage(txs);
            let liq = simd_detect_liquidation(txs);
            let sandwich = simd_detect_sandwich(txs);
            
            std::simd::simd_swizzle!(
                arb | liq | sandwich,
                [0, 1, 2, 3, 4, 5, 6, 7] // Lane ordering
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transaction::MockTransaction;

    #[test]
    fn test_platoon_initialization() {
        let platoon = CpuPlatoon::new();
        assert!(!platoon.grunt_squads.is_empty());
    }

    #[test]
    fn test_batch_detection() {
        let platoon = CpuPlatoon::new();
        let txs = [MockTransaction::default(); 8];
        let results = platoon.detect_batch(&txs);
        
        assert_eq!(results.len(), 8);
        println!("Execution mode: {}", platoon.engine_name());
    }

    #[test]
    fn test_legacy_volley() {
        let platoon = CpuPlatoon::new();
        let ammo = vec![AmmoCrate::dummy(); 16];
        let report = platoon.volley_fire(&ammo);
        
        assert!(report.success_rate > 0.0);
    }
}