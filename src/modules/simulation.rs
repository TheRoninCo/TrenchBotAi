use rand::{Rng, distributions::Alphanumeric};
use solana_sdk::pubkey::Pubkey;
use std::collections::HashMap;

#[derive(Clone)]
pub struct WhaleSimulator {
    rng: rand::rngs::ThreadRng,
    personalities: HashMap<Pubkey, WhalePersonality>,
}

impl WhaleSimulator {
    pub fn new() -> Self {
        Self {
            rng: rand::thread_rng(),
            personalities: HashMap::new(),
        }
    }

    /// Generates realistic mock transactions
    pub fn generate_bundle(&mut self, num_txs: usize) -> Vec<Transaction> {
        (0..num_txs)
            .map(|_| self.mock_transaction())
            .collect()
    }

    fn mock_transaction(&mut self) -> Transaction {
        let mut tx = Transaction::default();
        
        // Simulate SPL token transfer
        let from = self.random_pubkey();
        let to = self.random_pubkey();
        let amount = self.rng.gen_range(10..1000) as u64;
        
        // ... build instruction ...
        tx
    }

    pub fn assign_personality(&mut self, wallet: Pubkey) -> WhalePersonality {
        let personality = match self.rng.gen_range(0..6) {
            0 => WhalePersonality::Gorilla,
            1 => WhalePersonality::Shark,
            // ... other variants
        };
        self.personalities.insert(wallet, personality);
        personality
    }
}