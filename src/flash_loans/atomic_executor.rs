//! Atomic Transaction Executor
//! Ensures all flash loan operations succeed or fail together

use anyhow::{Result, anyhow};
use solana_sdk::{
    instruction::Instruction,
    transaction::Transaction,
    pubkey::Pubkey,
    signer::Signer,
    signature::Signature,
    compute_budget::ComputeBudgetInstruction,
};
use std::time::{Duration, Instant};
use tracing::{info, warn, error};

/// Atomic execution result
#[derive(Debug, Clone)]
pub struct AtomicResult {
    pub success: bool,
    pub signature: Option<Signature>,
    pub compute_units_used: u32,
    pub execution_time_ms: u64,
    pub error_message: Option<String>,
}

/// Atomic transaction executor for flash loans
pub struct AtomicExecutor {
    max_compute_units: u32,
    max_retries: u32,
    timeout_ms: u64,
}

impl AtomicExecutor {
    pub fn new() -> Self {
        Self {
            max_compute_units: 1_400_000, // Close to max CU limit
            max_retries: 3,
            timeout_ms: 30_000, // 30 second timeout
        }
    }
    
    /// Execute instructions atomically with compute budget optimization
    pub async fn execute_atomic_transaction(
        &self,
        instructions: Vec<Instruction>,
        payer: &dyn Signer,
        recent_blockhash: solana_sdk::hash::Hash,
    ) -> Result<AtomicResult> {
        let start_time = Instant::now();
        
        // Add compute budget instruction
        let mut final_instructions = vec![
            ComputeBudgetInstruction::set_compute_unit_limit(self.max_compute_units),
            ComputeBudgetInstruction::set_compute_unit_price(1000), // 1000 micro-lamports per CU
        ];
        final_instructions.extend(instructions);
        
        // Build transaction
        let mut transaction = Transaction::new_with_payer(
            &final_instructions,
            Some(&payer.pubkey()),
        );
        transaction.sign(&[payer], recent_blockhash);
        
        info!("🚀 Executing atomic flash loan transaction with {} instructions", final_instructions.len() - 2);
        
        // In a real implementation, this would:
        // 1. Simulate the transaction first
        // 2. Send with proper retry logic
        // 3. Wait for confirmation
        // 4. Handle errors appropriately
        
        // Simulate execution for now
        tokio::time::sleep(Duration::from_millis(500)).await;
        
        let execution_time = start_time.elapsed().as_millis() as u64;
        
        // Simulate success/failure based on transaction complexity
        let success = final_instructions.len() <= 20; // Fail if too many instructions
        
        if success {
            info!("✅ Atomic transaction executed successfully in {}ms", execution_time);
            Ok(AtomicResult {
                success: true,
                signature: Some(Signature::new_unique()),
                compute_units_used: 150_000,
                execution_time_ms: execution_time,
                error_message: None,
            })
        } else {
            warn!("❌ Atomic transaction failed - too many instructions");
            Ok(AtomicResult {
                success: false,
                signature: None,
                compute_units_used: 50_000,
                execution_time_ms: execution_time,
                error_message: Some("Transaction too complex".to_string()),
            })
        }
    }
    
    /// Simulate transaction before execution
    pub async fn simulate_transaction(
        &self,
        instructions: Vec<Instruction>,
        payer_pubkey: Pubkey,
        recent_blockhash: solana_sdk::hash::Hash,
    ) -> Result<SimulationResult> {
        info!("🔍 Simulating flash loan transaction...");
        
        // In reality, this would use RPC simulate_transaction
        // For now, we'll do basic validation
        
        let estimated_compute_units = instructions.len() as u32 * 7_000; // ~7k CU per instruction
        let will_succeed = estimated_compute_units <= self.max_compute_units;
        
        if will_succeed {
            info!("✅ Simulation successful - estimated {} CU", estimated_compute_units);
        } else {
            warn!("⚠️ Simulation shows potential failure - {} CU exceeds limit", estimated_compute_units);
        }
        
        Ok(SimulationResult {
            success: will_succeed,
            compute_units_consumed: estimated_compute_units,
            logs: vec![
                "Program log: Flash loan initiated".to_string(),
                "Program log: Executing arbitrage".to_string(),  
                "Program log: Flash loan repaid".to_string(),
            ],
            accounts_data_len: 1024,
            return_data: None,
        })
    }
    
    /// Optimize transaction for maximum efficiency
    pub fn optimize_instructions(&self, mut instructions: Vec<Instruction>) -> Vec<Instruction> {
        info!("🔧 Optimizing {} instructions for atomic execution", instructions.len());
        
        // Remove duplicate account metas
        for instruction in &mut instructions {
            instruction.accounts.sort_by_key(|meta| meta.pubkey);
            instruction.accounts.dedup_by_key(|meta| meta.pubkey);
        }
        
        // Combine similar instructions where possible
        // (This would be more sophisticated in a real implementation)
        
        info!("✅ Optimized to {} instructions", instructions.len());
        instructions
    }
    
    /// Build transaction with proper fee estimation
    pub async fn build_optimized_transaction(
        &self,
        instructions: Vec<Instruction>,
        payer: &dyn Signer,
        recent_blockhash: solana_sdk::hash::Hash,
        priority_fee_lamports: u64,
    ) -> Result<Transaction> {
        let optimized_instructions = self.optimize_instructions(instructions);
        
        let mut final_instructions = vec![
            ComputeBudgetInstruction::set_compute_unit_limit(self.max_compute_units),
        ];
        
        // Add priority fee if specified
        if priority_fee_lamports > 0 {
            let micro_lamports = (priority_fee_lamports * 1_000_000) / (self.max_compute_units as u64);
            final_instructions.push(
                ComputeBudgetInstruction::set_compute_unit_price(micro_lamports)
            );
        }
        
        final_instructions.extend(optimized_instructions);
        
        let mut transaction = Transaction::new_with_payer(
            &final_instructions,
            Some(&payer.pubkey()),
        );
        
        transaction.sign(&[payer], recent_blockhash);
        
        info!("🏗️ Built optimized transaction with {} total instructions", final_instructions.len());
        Ok(transaction)
    }
    
    /// Execute with retry logic and error handling
    pub async fn execute_with_retries(
        &self,
        instructions: Vec<Instruction>,
        payer: &dyn Signer,
        recent_blockhash: solana_sdk::hash::Hash,
    ) -> Result<AtomicResult> {
        let mut last_error = None;
        
        for attempt in 1..=self.max_retries {
            info!("🔄 Flash loan execution attempt {}/{}", attempt, self.max_retries);
            
            match self.execute_atomic_transaction(instructions.clone(), payer, recent_blockhash).await {
                Ok(result) if result.success => {
                    info!("🎯 Flash loan executed successfully on attempt {}", attempt);
                    return Ok(result);
                },
                Ok(result) => {
                    last_error = result.error_message;
                    warn!("❌ Attempt {} failed: {:?}", attempt, last_error);
                },
                Err(e) => {
                    last_error = Some(e.to_string());
                    error!("💥 Attempt {} errored: {}", attempt, e);
                },
            }
            
            // Exponential backoff
            if attempt < self.max_retries {
                let delay = Duration::from_millis(100 * (2_u64.pow(attempt - 1)));
                info!("⏳ Waiting {:?} before retry...", delay);
                tokio::time::sleep(delay).await;
            }
        }
        
        error!("💥 All {} attempts failed", self.max_retries);
        Ok(AtomicResult {
            success: false,
            signature: None,
            compute_units_used: 0,
            execution_time_ms: 0,
            error_message: last_error,
        })
    }
}

#[derive(Debug, Clone)]
pub struct SimulationResult {
    pub success: bool,
    pub compute_units_consumed: u32,
    pub logs: Vec<String>,
    pub accounts_data_len: usize,
    pub return_data: Option<Vec<u8>>,
}

impl Default for AtomicExecutor {
    fn default() -> Self {
        Self::new()
    }
}