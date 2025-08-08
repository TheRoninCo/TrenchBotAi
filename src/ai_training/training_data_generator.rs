//! Training Data Generation for AI Models
//! 
//! Generate realistic MEV coordination patterns and rug pull scenarios
//! for training our AI detection models

use super::{TrainingExample, Transaction, TransactionType};
use anyhow::Result;
use chrono::{DateTime, Utc, Duration};
use rand::Rng;
use std::collections::HashMap;

pub struct TrainingDataGenerator {
    pub seed: u64,
    pub coordination_probability: f64,
    pub rug_pull_probability: f64,
}

impl Default for TrainingDataGenerator {
    fn default() -> Self {
        Self {
            seed: 42,
            coordination_probability: 0.3, // 30% of examples show coordination
            rug_pull_probability: 0.15,    // 15% of examples have rug pulls
        }
    }
}

impl TrainingDataGenerator {
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            coordination_probability: 0.3,
            rug_pull_probability: 0.15,
        }
    }

    /// Generate a complete training dataset for MEV detection
    pub fn generate_training_dataset(&mut self, num_examples: usize) -> Result<Vec<TrainingExample>> {
        tracing::info!("🎲 Generating {} training examples for AI models", num_examples);
        
        let mut training_data = Vec::with_capacity(num_examples);
        let mut rng = rand::thread_rng();
        
        for i in 0..num_examples {
            let example = if rng.gen::<f64>() < self.coordination_probability {
                // Generate coordinated attack pattern
                if rng.gen::<f64>() < self.rug_pull_probability {
                    self.generate_rug_pull_example(i)?
                } else {
                    self.generate_coordinated_example(i)?
                }
            } else {
                // Generate normal transaction pattern
                self.generate_normal_example(i)?
            };
            
            training_data.push(example);
            
            if i % 1000 == 0 {
                tracing::debug!("Generated {} training examples", i);
            }
        }
        
        tracing::info!("✅ Generated complete training dataset with:");
        let coordinated_count = training_data.iter().filter(|e| e.coordination_label > 0.5).count();
        let rug_pull_count = training_data.iter().filter(|e| e.rug_pull_occurred).count();
        
        tracing::info!("  📈 {} coordinated examples ({:.1}%)", 
                      coordinated_count, 
                      100.0 * coordinated_count as f64 / num_examples as f64);
        tracing::info!("  💀 {} rug pull examples ({:.1}%)", 
                      rug_pull_count, 
                      100.0 * rug_pull_count as f64 / num_examples as f64);
        
        Ok(training_data)
    }

    /// Generate example of coordinated MEV attack
    fn generate_coordinated_example(&mut self, seed: usize) -> Result<TrainingExample> {
        let mut rng = rand::thread_rng();
        let base_time = Utc::now() - Duration::minutes(rng.gen_range(1..1440)); // Random time in last day
        
        // Create coordinated wallet cluster
        let coordinator_wallets = self.generate_wallet_cluster(3, 8); // 3-8 coordinated wallets
        let target_token = self.generate_token_mint();
        
        let mut transactions = Vec::new();
        
        // Phase 1: Coordinated positioning (within 30 seconds)
        let position_start = base_time;
        for (i, wallet) in coordinator_wallets.iter().enumerate() {
            let tx_time = position_start + Duration::seconds(rng.gen_range(0..30));
            let amount = rng.gen_range(50.0..500.0); // Significant amounts
            
            transactions.push(Transaction {
                signature: format!("coord_pos_{}_{}", seed, i),
                wallet: wallet.clone(),
                token_mint: target_token.clone(),
                amount_sol: amount,
                timestamp: tx_time,
                transaction_type: TransactionType::Swap,
            });
        }
        
        // Phase 2: Coordinated execution (within next 60 seconds)  
        let execution_start = position_start + Duration::seconds(30);
        for (i, wallet) in coordinator_wallets.iter().enumerate() {
            let tx_time = execution_start + Duration::seconds(rng.gen_range(0..60));
            let amount = rng.gen_range(100.0..1000.0);
            
            transactions.push(Transaction {
                signature: format!("coord_exec_{}_{}", seed, i),
                wallet: wallet.clone(),
                token_mint: target_token.clone(),
                amount_sol: amount,
                timestamp: tx_time,
                transaction_type: if rng.gen::<f64>() > 0.5 { TransactionType::Swap } else { TransactionType::LiquidityRemove },
            });
        }
        
        // Add some normal transactions to make it realistic
        self.add_noise_transactions(&mut transactions, &base_time, &target_token, 3, 10)?;
        
        // Sort by timestamp
        transactions.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
        
        Ok(TrainingExample {
            transaction_sequence: transactions,
            coordination_label: rng.gen_range(0.7..0.95), // High coordination signal
            rug_pull_occurred: false,
            time_to_rug_pull: None,
            profit_amount: Some(rng.gen_range(1000.0..50000.0)), // MEV profit extracted
        })
    }

    /// Generate example of rug pull pattern
    fn generate_rug_pull_example(&mut self, seed: usize) -> Result<TrainingExample> {
        let mut rng = rand::thread_rng();
        let base_time = Utc::now() - Duration::minutes(rng.gen_range(60..2880)); // 1-48 hours ago
        
        // Create rug pull setup
        let rug_puller_wallet = format!("rug_puller_{}", seed);
        let victim_wallets = self.generate_wallet_cluster(5, 15); // 5-15 victim wallets
        let rug_token = self.generate_token_mint();
        
        let mut transactions = Vec::new();
        
        // Phase 1: Liquidity provision (rug pull setup)
        let setup_time = base_time;
        transactions.push(Transaction {
            signature: format!("rug_setup_{}", seed),
            wallet: rug_puller_wallet.clone(),
            token_mint: rug_token.clone(),
            amount_sol: rng.gen_range(1000.0..10000.0), // Large liquidity provision
            timestamp: setup_time,
            transaction_type: TransactionType::LiquidityAdd,
        });
        
        // Phase 2: Victim accumulation (over several minutes/hours)
        let accumulation_period = rng.gen_range(300..7200); // 5 minutes to 2 hours
        for (i, wallet) in victim_wallets.iter().enumerate() {
            let tx_time = setup_time + Duration::seconds(rng.gen_range(60..accumulation_period));
            let amount = rng.gen_range(10.0..200.0); // Victim amounts
            
            transactions.push(Transaction {
                signature: format!("victim_buy_{}_{}", seed, i),
                wallet: wallet.clone(),
                token_mint: rug_token.clone(),
                amount_sol: amount,
                timestamp: tx_time,
                transaction_type: TransactionType::Swap,
            });
        }
        
        // Phase 3: Rug pull execution (coordinated massive dump)
        let rug_time = setup_time + Duration::seconds(accumulation_period);
        let rug_transactions = 3 + rng.gen_range(0..5); // Multiple large sells
        
        for i in 0..rug_transactions {
            let tx_time = rug_time + Duration::seconds(rng.gen_range(0..30)); // Quick succession
            let amount = rng.gen_range(2000.0..20000.0); // Massive sells
            
            transactions.push(Transaction {
                signature: format!("rug_pull_{}_{}", seed, i),
                wallet: rug_puller_wallet.clone(),
                token_mint: rug_token.clone(),
                amount_sol: amount,
                timestamp: tx_time,
                transaction_type: TransactionType::LiquidityRemove,
            });
        }
        
        // Add some panicked victim sells after the rug pull
        for (i, wallet) in victim_wallets.iter().take(5).enumerate() {
            let tx_time = rug_time + Duration::seconds(rng.gen_range(30..300));
            let amount = rng.gen_range(5.0..50.0); // Small panic sells
            
            transactions.push(Transaction {
                signature: format!("panic_sell_{}_{}", seed, i),
                wallet: wallet.clone(),
                token_mint: rug_token.clone(),
                amount_sol: amount,
                timestamp: tx_time,
                transaction_type: TransactionType::Swap,
            });
        }
        
        // Sort by timestamp
        transactions.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
        
        Ok(TrainingExample {
            transaction_sequence: transactions,
            coordination_label: rng.gen_range(0.8..1.0), // Very high coordination (rug pull is ultimate coordination)
            rug_pull_occurred: true,
            time_to_rug_pull: Some((rug_time - base_time).num_seconds()),
            profit_amount: Some(rng.gen_range(5000.0..100000.0)), // Rug pull profits
        })
    }

    /// Generate example of normal trading activity
    fn generate_normal_example(&mut self, seed: usize) -> Result<TrainingExample> {
        let mut rng = rand::thread_rng();
        let base_time = Utc::now() - Duration::minutes(rng.gen_range(1..1440));
        
        // Create random independent wallets
        let num_wallets = rng.gen_range(3..12);
        let wallets = self.generate_wallet_cluster(num_wallets, num_wallets);
        let tokens = vec![
            self.generate_token_mint(),
            self.generate_token_mint(),
            self.generate_token_mint(),
        ];
        
        let mut transactions = Vec::new();
        
        // Generate random independent transactions over time
        let time_span = rng.gen_range(300..3600); // 5 minutes to 1 hour
        
        for (i, wallet) in wallets.iter().enumerate() {
            let num_txs = rng.gen_range(1..4); // Each wallet does 1-3 transactions
            
            for j in 0..num_txs {
                let tx_time = base_time + Duration::seconds(rng.gen_range(0..time_span));
                let amount = rng.gen_range(1.0..100.0); // Normal trading amounts
                let token = &tokens[rng.gen_range(0..tokens.len())];
                
                let tx_type = match rng.gen_range(0..100) {
                    0..60 => TransactionType::Swap,        // 60% swaps
                    60..75 => TransactionType::Transfer,    // 15% transfers
                    75..85 => TransactionType::LiquidityAdd, // 10% liquidity adds
                    85..95 => TransactionType::LiquidityRemove, // 10% liquidity removes
                    95..100 => TransactionType::FlashLoan, // 5% flash loans
                    _ => TransactionType::Swap,
                };
                
                transactions.push(Transaction {
                    signature: format!("normal_{}_{}_{}", seed, i, j),
                    wallet: wallet.clone(),
                    token_mint: token.clone(),
                    amount_sol: amount,
                    timestamp: tx_time,
                    transaction_type: tx_type,
                });
            }
        }
        
        // Sort by timestamp
        transactions.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
        
        Ok(TrainingExample {
            transaction_sequence: transactions,
            coordination_label: rng.gen_range(0.0..0.3), // Low coordination signal
            rug_pull_occurred: false,
            time_to_rug_pull: None,
            profit_amount: Some(rng.gen_range(10.0..500.0)), // Normal trading profits
        })
    }

    /// Generate a cluster of related wallet addresses
    fn generate_wallet_cluster(&self, min_count: usize, max_count: usize) -> Vec<String> {
        let mut rng = rand::thread_rng();
        let count = rng.gen_range(min_count..=max_count);
        let base_prefix = format!("{:08x}", rng.gen::<u32>());
        
        (0..count)
            .map(|i| format!("{}_{:04x}{}", base_prefix, i, self.generate_wallet_suffix()))
            .collect()
    }

    fn generate_wallet_suffix(&self) -> String {
        let mut rng = rand::thread_rng();
        format!("{:016x}", rng.gen::<u64>())
    }

    fn generate_token_mint(&self) -> String {
        let mut rng = rand::thread_rng();
        format!("token_{:016x}{:016x}", rng.gen::<u64>(), rng.gen::<u64>())
    }

    fn add_noise_transactions(
        &self,
        transactions: &mut Vec<Transaction>,
        base_time: &DateTime<Utc>,
        target_token: &str,
        min_noise: usize,
        max_noise: usize,
    ) -> Result<()> {
        let mut rng = rand::thread_rng();
        let noise_count = rng.gen_range(min_noise..=max_noise);
        
        for i in 0..noise_count {
            let tx_time = *base_time + Duration::seconds(rng.gen_range(0..300)); // Within 5 minutes
            let wallet = format!("noise_wallet_{}", rng.gen::<u32>());
            let amount = rng.gen_range(1.0..50.0);
            
            transactions.push(Transaction {
                signature: format!("noise_tx_{}", i),
                wallet,
                token_mint: target_token.to_string(),
                amount_sol: amount,
                timestamp: tx_time,
                transaction_type: TransactionType::Swap,
            });
        }
        
        Ok(())
    }

    /// Generate synthetic flash loan attack pattern
    pub fn generate_flash_loan_example(&mut self, seed: usize) -> Result<TrainingExample> {
        let mut rng = rand::thread_rng();
        let base_time = Utc::now() - Duration::minutes(rng.gen_range(1..60));
        
        let attacker_wallet = format!("flash_attacker_{}", seed);
        let target_token = self.generate_token_mint();
        let mut transactions = Vec::new();
        
        // Flash loan sequence (all within same block/transaction)
        let flash_time = base_time;
        
        // 1. Flash loan borrow
        transactions.push(Transaction {
            signature: format!("flash_borrow_{}", seed),
            wallet: attacker_wallet.clone(),
            token_mint: "SOL".to_string(),
            amount_sol: rng.gen_range(10000.0..100000.0), // Large flash loan
            timestamp: flash_time,
            transaction_type: TransactionType::FlashLoan,
        });
        
        // 2. Multiple swaps for manipulation
        for i in 0..3 {
            transactions.push(Transaction {
                signature: format!("flash_swap_{}_{}", seed, i),
                wallet: attacker_wallet.clone(),
                token_mint: target_token.clone(),
                amount_sol: rng.gen_range(5000.0..20000.0),
                timestamp: flash_time,
                transaction_type: TransactionType::Swap,
            });
        }
        
        // 3. Flash loan repay
        transactions.push(Transaction {
            signature: format!("flash_repay_{}", seed),
            wallet: attacker_wallet.clone(),
            token_mint: "SOL".to_string(),
            amount_sol: rng.gen_range(10100.0..100100.0), // Slightly more (with fee)
            timestamp: flash_time,
            transaction_type: TransactionType::FlashLoan,
        });
        
        Ok(TrainingExample {
            transaction_sequence: transactions,
            coordination_label: 0.9, // Flash loans are highly coordinated by definition
            rug_pull_occurred: false,
            time_to_rug_pull: None,
            profit_amount: Some(rng.gen_range(100.0..5000.0)), // Flash loan arbitrage profit
        })
    }

    /// Generate balanced dataset with various attack patterns
    pub fn generate_balanced_dataset(&mut self, total_examples: usize) -> Result<Vec<TrainingExample>> {
        let mut dataset = Vec::new();
        
        // Distribution of example types
        let normal_examples = (total_examples as f64 * 0.5) as usize; // 50% normal
        let coordinated_examples = (total_examples as f64 * 0.25) as usize; // 25% coordinated
        let rug_pull_examples = (total_examples as f64 * 0.15) as usize; // 15% rug pulls
        let flash_loan_examples = total_examples - normal_examples - coordinated_examples - rug_pull_examples; // 10% flash loans
        
        tracing::info!("🎯 Generating balanced training dataset:");
        tracing::info!("  Normal examples: {}", normal_examples);
        tracing::info!("  Coordinated examples: {}", coordinated_examples);
        tracing::info!("  Rug pull examples: {}", rug_pull_examples);
        tracing::info!("  Flash loan examples: {}", flash_loan_examples);
        
        // Generate each type
        for i in 0..normal_examples {
            dataset.push(self.generate_normal_example(i)?);
        }
        
        for i in 0..coordinated_examples {
            dataset.push(self.generate_coordinated_example(i + normal_examples)?);
        }
        
        for i in 0..rug_pull_examples {
            dataset.push(self.generate_rug_pull_example(i + normal_examples + coordinated_examples)?);
        }
        
        for i in 0..flash_loan_examples {
            dataset.push(self.generate_flash_loan_example(i + normal_examples + coordinated_examples + rug_pull_examples)?);
        }
        
        // Shuffle the dataset
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        dataset.shuffle(&mut rng);
        
        tracing::info!("✅ Generated balanced dataset with {} total examples", dataset.len());
        Ok(dataset)
    }
}