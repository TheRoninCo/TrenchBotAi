// src/modules/mev/analyzer.rs
use super::{
    types::{BundleFeatures, MevType},
    RpcSimulateTransactionResult,
};
use solana_sdk::{
    pubkey::Pubkey,
    transaction::{Transaction, VersionedTransaction},
};
use std::collections::{HashMap, HashSet};
use tracing::{debug, instrument};

/// Feature extraction and bundle analysis
pub struct BundleAnalyzer {
    known_programs: HashSet<Pubkey>,
    hot_accounts: HashMap<Pubkey, u64>,
    metrics: AnalyzerMetrics,
}

/// Prometheus metrics for analyzer
struct AnalyzerMetrics {
    #[cfg(feature = "monitoring")]
    extraction_time: prometheus::HistogramVec,
    #[cfg(feature = "monitoring")]
    account_access: prometheus::IntCounterVec,
}

impl BundleAnalyzer {
    pub fn new(known_programs: Vec<Pubkey>) -> Self {
        Self {
            known_programs: known_programs.into_iter().collect(),
            hot_accounts: HashMap::new(),
            metrics: Self::init_metrics(),
        }
    }

    #[instrument(skip_all, fields(num_txs = txs.len()))]
    pub fn analyze_bundle(
        &mut self,
        txs: &[Transaction],
        sim_result: &RpcSimulateTransactionResult,
        slot: u64,
    ) -> BundleFeatures {
        #[cfg(feature = "monitoring")]
        let _timer = self.metrics.extraction_time
            .with_label_values(&["full"])
            .start_timer();

        BundleFeatures {
            slot,
            tip_lamports: self.extract_tip(&txs[0]),
            cu_limit: self.extract_compute_units(&txs[0]).0,
            cu_price: self.extract_compute_units(&txs[0]).1,
            accounts: self.extract_accounts(txs),
            programs: self.extract_programs(txs),
            mev_type: MevType::Unknown, // Detector will populate this
            success: sim_result.err.is_none(),
            timestamp: chrono::Utc::now().timestamp(),
            signer: txs[0].message.account_keys[0],
        }
    }

    #[instrument(skip(self, txs))]
    fn extract_accounts(&mut self, txs: &[Transaction]) -> Vec<Pubkey> {
        #[cfg(feature = "monitoring")]
        let _timer = self.metrics.extraction_time
            .with_label_values(&["accounts"])
            .start_timer();

        let mut accounts = HashMap::new();
        for tx in txs {
            for key in &tx.message.account_keys {
                *accounts.entry(*key).or_insert(0) += 1;
                #[cfg(feature = "monitoring")]
                self.metrics.account_access
                    .with_label_values(&[&key.to_string()])
                    .inc();
            }
        }

        let mut sorted: Vec<_> = accounts.into_iter().collect();
        sorted.sort_by(|(_, a), (_, b)| b.cmp(a));
        
        sorted.into_iter()
            .map(|(acc, count)| {
                *self.hot_accounts.entry(acc).or_insert(0) += count;
                acc
            })
            .collect()
    }

    #[instrument(skip(self, txs))]
    fn extract_programs(&self, txs: &[Transaction]) -> Vec<String> {
        #[cfg(feature = "monitoring")]
        let _timer = self.metrics.extraction_time
            .with_label_values(&["programs"])
            .start_timer();

        let mut programs = HashSet::new();
        for tx in txs {
            for ix in &tx.message.instructions {
                programs.insert(ix.program_id.to_string());
            }
        }
        programs.into_iter().collect()
    }

    fn extract_compute_units(&self, tx: &Transaction) -> (u32, u32) {
        let mut cu_limit = 200_000; // Default
        let mut cu_price = 0u32; // Initialize as u32

        for ix in &tx.message.instructions {
            if let Ok(budget_ix) = solana_sdk::compute_budget::ComputeBudgetInstruction::try_from(ix) {
                match budget_ix {
                    solana_sdk::compute_budget::ComputeBudgetInstruction::SetComputeUnitLimit(limit) => {
                        cu_limit = limit;
                    }
                    solana_sdk::compute_budget::ComputeBudgetInstruction::SetComputeUnitPrice(price) => {
                        cu_price = price;
                    }
                    _ => {}
                }
            }
        }

        (cu_limit, cu_price)
    }

    fn extract_tip(&self, tx: &Transaction) -> u64 {
        tx.message.instructions.iter()
            .find(|ix| ix.program_id == solana_sdk::system_program::id())
            .and_then(|ix| ix.data.get(0..8).map(|d| u64::from_le_bytes(d.try_into().unwrap_or_default()))) // Use unwrap_or_default
            .unwrap_or(0)
    }

    pub fn is_swap_program(&self, program_id: &Pubkey) -> bool {
        self.known_programs.contains(program_id)
    }

    fn init_metrics() -> AnalyzerMetrics {
        AnalyzerMetrics {
            #[cfg(feature = "monitoring")]
            extraction_time: prometheus::register_histogram_vec!(
                "mev_analyzer_extraction_time_seconds",
                "Time taken for feature extraction",
                &["stage"]
            ).unwrap(),
            #[cfg(feature = "monitoring")]
            account_access: prometheus::register_int_counter_vec!(
                "mev_account_access_total",
                "Count of account accesses",
                &["account"]
            ).unwrap(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use solana_sdk::{instruction::Instruction, pubkey::Pubkey};

    fn test_tx() -> Transaction {
        Transaction {
            signatures: vec![],
            message: solana_sdk::message::Message {
                account_keys: vec![Pubkey::new_unique()],
                instructions: vec![Instruction {
                    program_id: Pubkey::new_unique(),
                    accounts: vec![],
                    data: vec![],
                }],
                ..Default::default()
            },
        }
    }

    #[test]
    fn test_extract_programs() {
        let analyzer = BundleAnalyzer::new(vec![]);
        let txs = vec![test_tx()];
        let programs = analyzer.extract_programs(&txs);
        assert_eq!(programs.len(), 1);
    }

    #[test]
    fn test_analyze_bundle() {
        let mut analyzer = BundleAnalyzer::new(vec![]);
        let txs = vec![test_tx()];
        let sim_result = RpcSimulateTransactionResult::default();
        let features = analyzer.analyze_bundle(&txs, &sim_result, 123);
        
        assert_eq!(features.slot, 123);
        assert_eq!(features.accounts.len(), 1);
    }

    #[test]
    fn test_swap_program_detection() {
        let known_program = Pubkey::new_unique();
        let analyzer = BundleAnalyzer::new(vec![known_program]);
        
        assert!(analyzer.is_swap_program(&known_program));
        assert!(!analyzer.is_swap_program(&Pubkey::new_unique()));
    }
}