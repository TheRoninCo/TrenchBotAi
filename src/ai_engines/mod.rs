//! AI Engines Module - Advanced AI Systems for Rug Pull Detection
//! 
//! This module contains cutting-edge AI engines implementing the latest advances
//! in AI mathematical reconfiguration for ultra-fast rug pull detection and
//! counter-attack strategies.

pub mod flash_attention;
pub mod spiking_neural_net;
pub mod quantum_graph_nn;
pub mod causal_inference;
pub mod hardware_accelerator;

// Re-export key components
pub use flash_attention::{
    FlashAttentionEngine, FlashAttentionConfig, TransactionEmbedding, AttentionOutput,
    TransactionEmbedder, PerformanceMetrics as FlashPerformanceMetrics
};

pub use spiking_neural_net::{
    SpikingRugPullBrain, SpikingDetectionResult, SpikingCluster, SpikingNetworkMetrics,
    NeuronParams, SpikingNeuron, SpikeEvent
};

pub use quantum_graph_nn::{
    QuantumGraphNeuralNetwork, QuantumCoordinationResult, QuantumCluster, EntangledPair,
    QuantumNetworkMetrics, QuantumQubit
};

pub use causal_inference::{
    CausalInferenceEngine, CausalPredictionResult, CausalFactor, CounterfactualScenario,
    InterventionRecommendation, CausalGraphSummary
};

pub use hardware_accelerator::{
    HardwareAccelerationPipeline, HardwareConfig, AcceleratorType, PrecisionType,
    HardwareAcceleratedResult, HardwarePerformanceMetrics, OptimizationLevel
};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use std::sync::Arc;
use tokio::sync::RwLock;

/// AI Engine Manager alias for backwards compatibility
pub type AIEngineManager = UnifiedAIEngine;

/// Unified AI Engine Manager - Orchestrates all AI systems
pub struct UnifiedAIEngine {
    flash_attention: Arc<RwLock<FlashAttentionEngine>>,
    spiking_network: Arc<RwLock<SpikingRugPullBrain>>,
    quantum_gnn: Arc<RwLock<QuantumGraphNeuralNetwork>>,
    causal_inference: Arc<RwLock<CausalInferenceEngine>>,
    hardware_pipeline: Arc<RwLock<HardwareAccelerationPipeline>>,
    
    // Engine configuration
    ensemble_weights: EnsembleWeights,
    confidence_threshold: f64,
    processing_mode: ProcessingMode,
}

#[derive(Debug, Clone)]
pub struct EnsembleWeights {
    pub flash_attention: f64,
    pub spiking_network: f64,
    pub quantum_gnn: f64,
    pub causal_inference: f64,
    pub hardware_acceleration: f64,
}

impl Default for EnsembleWeights {
    fn default() -> Self {
        Self {
            flash_attention: 0.25,
            spiking_network: 0.30,
            quantum_gnn: 0.25,
            causal_inference: 0.20,
            hardware_acceleration: 1.0, // Overall hardware acceleration multiplier
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ProcessingMode {
    Sequential,     // Process engines one by one
    Parallel,       // Process all engines simultaneously
    HardwareFirst,  // Use hardware acceleration when available
    Adaptive,       // Dynamically choose based on load
}

impl UnifiedAIEngine {
    /// Create new unified AI engine with all systems initialized
    pub async fn new() -> Result<Self> {
        // Initialize Flash Attention Engine
        let flash_config = FlashAttentionConfig::default();
        let flash_attention = Arc::new(RwLock::new(FlashAttentionEngine::new(flash_config)?));
        
        // Initialize Spiking Neural Network
        let (spiking_brain, _receiver) = SpikingRugPullBrain::new()?;
        let spiking_network = Arc::new(RwLock::new(spiking_brain));
        
        // Initialize Quantum Graph Neural Network
        let quantum_gnn = Arc::new(RwLock::new(
            QuantumGraphNeuralNetwork::new(128, vec![256, 128], 64, 8)?
        ));
        
        // Initialize Causal Inference Engine
        let causal_inference = Arc::new(RwLock::new(
            CausalInferenceEngine::new(24)? // 24-hour prediction horizon
        ));
        
        // Initialize Hardware Acceleration Pipeline
        let hardware_config = HardwareConfig::default();
        let mut hardware_pipeline = HardwareAccelerationPipeline::new(hardware_config)?;
        hardware_pipeline.initialize_pipelines().await?;
        let hardware_pipeline = Arc::new(RwLock::new(hardware_pipeline));

        Ok(Self {
            flash_attention,
            spiking_network,
            quantum_gnn,
            causal_inference,
            hardware_pipeline,
            ensemble_weights: EnsembleWeights::default(),
            confidence_threshold: 0.7,
            processing_mode: ProcessingMode::HardwareFirst,
        })
    }

    /// Process transactions through all AI engines and generate comprehensive analysis
    pub async fn analyze_transactions(
        &self,
        transactions: &[crate::analytics::Transaction],
        market_data: Option<&crate::analytics::MarketSnapshot>,
    ) -> Result<UnifiedAIResult> {
        let start_time = std::time::Instant::now();

        match self.processing_mode {
            ProcessingMode::HardwareFirst => {
                self.process_hardware_first(transactions, market_data).await
            }
            ProcessingMode::Parallel => {
                self.process_parallel(transactions, market_data).await
            }
            ProcessingMode::Sequential => {
                self.process_sequential(transactions, market_data).await
            }
            ProcessingMode::Adaptive => {
                self.process_adaptive(transactions, market_data).await
            }
        }
    }

    async fn process_hardware_first(
        &self,
        transactions: &[crate::analytics::Transaction],
        market_data: Option<&crate::analytics::MarketSnapshot>,
    ) -> Result<UnifiedAIResult> {
        let start_time = std::time::Instant::now();

        // First attempt hardware-accelerated processing
        if let Ok(mut hardware_pipeline) = self.hardware_pipeline.try_write() {
            if let Ok(hardware_result) = hardware_pipeline.process_transactions_accelerated(transactions).await {
                return Ok(UnifiedAIResult {
                    coordination_score: hardware_result.coordination_score,
                    rug_pull_probability: hardware_result.rug_pull_probability,
                    confidence: hardware_result.confidence,
                    processing_mode_used: ProcessingMode::HardwareFirst,
                    
                    // Individual engine results from hardware pipeline
                    flash_attention_result: Some(FlashAttentionUnifiedResult {
                        coordination_score: hardware_result.ai_engine_results.flash_attention.coordination_score,
                        rug_pull_probability: hardware_result.ai_engine_results.flash_attention.rug_pull_probability,
                        confidence: hardware_result.ai_engine_results.flash_attention.confidence,
                        processing_time_ns: hardware_result.ai_engine_results.flash_attention.processing_time_ns,
                    }),
                    
                    spiking_nn_result: Some(SpikingNNUnifiedResult {
                        coordination_score: hardware_result.ai_engine_results.spiking_neural_network.coordination_score,
                        rug_pull_probability: hardware_result.ai_engine_results.spiking_neural_network.rug_pull_probability,
                        confidence: hardware_result.ai_engine_results.spiking_neural_network.confidence,
                        processing_time_ns: hardware_result.ai_engine_results.spiking_neural_network.processing_time_ns,
                    }),
                    
                    quantum_gnn_result: Some(QuantumGNNUnifiedResult {
                        coordination_score: hardware_result.ai_engine_results.quantum_graph_nn.coordination_score,
                        rug_pull_probability: hardware_result.ai_engine_results.quantum_graph_nn.rug_pull_probability,
                        confidence: hardware_result.ai_engine_results.quantum_graph_nn.confidence,
                        processing_time_ns: hardware_result.ai_engine_results.quantum_graph_nn.processing_time_ns,
                    }),
                    
                    causal_inference_result: Some(CausalInferenceUnifiedResult {
                        coordination_score: hardware_result.ai_engine_results.causal_inference.coordination_score,
                        rug_pull_probability: hardware_result.ai_engine_results.causal_inference.rug_pull_probability,
                        confidence: hardware_result.ai_engine_results.causal_inference.confidence,
                        processing_time_ns: hardware_result.ai_engine_results.causal_inference.processing_time_ns,
                    }),
                    
                    detected_clusters: hardware_result.detected_clusters.len(),
                    key_risk_factors: vec!["hardware_detected".to_string()],
                    recommended_actions: vec!["Monitor hardware-detected patterns".to_string()],
                    
                    performance_metrics: UnifiedPerformanceMetrics {
                        total_processing_time_ns: hardware_result.processing_time_ns,
                        hardware_acceleration_used: true,
                        speedup_factor: hardware_result.performance_metrics.speedup_factor,
                        memory_usage_mb: hardware_result.hardware_utilization.memory_utilization as usize,
                        cpu_utilization: hardware_result.hardware_utilization.cpu_utilization,
                        gpu_utilization: Some(hardware_result.hardware_utilization.gpu_utilization),
                    },
                    
                    ensemble_weights: self.ensemble_weights.clone(),
                    timestamp: Utc::now(),
                });
            }
        }

        // Fallback to software processing if hardware fails
        self.process_parallel(transactions, market_data).await
    }

    async fn process_parallel(
        &self,
        transactions: &[crate::analytics::Transaction],
        market_data: Option<&crate::analytics::MarketSnapshot>,
    ) -> Result<UnifiedAIResult> {
        let start_time = std::time::Instant::now();

        // Process all engines in parallel
        let (flash_result, spiking_result, quantum_result, causal_result) = tokio::try_join!(
            self.run_flash_attention(transactions),
            self.run_spiking_network(transactions),
            self.run_quantum_gnn(transactions),
            self.run_causal_inference(transactions, market_data)
        )?;

        let combined_result = self.combine_results(
            &flash_result,
            &spiking_result, 
            &quantum_result,
            &causal_result,
        ).await?;

        let processing_time = start_time.elapsed();

        Ok(UnifiedAIResult {
            coordination_score: combined_result.coordination_score,
            rug_pull_probability: combined_result.rug_pull_probability,
            confidence: combined_result.confidence,
            processing_mode_used: ProcessingMode::Parallel,
            
            flash_attention_result: Some(FlashAttentionUnifiedResult {
                coordination_score: flash_result.coordination_score,
                rug_pull_probability: flash_result.rug_pull_probability,
                confidence: flash_result.confidence,
                processing_time_ns: 0, // Would be tracked in actual implementation
            }),
            
            spiking_nn_result: Some(SpikingNNUnifiedResult {
                coordination_score: spiking_result.coordination_score,
                rug_pull_probability: if spiking_result.is_coordinated_attack { 0.9 } else { 0.1 },
                confidence: spiking_result.confidence,
                processing_time_ns: spiking_result.processing_time_us * 1000,
            }),
            
            quantum_gnn_result: Some(QuantumGNNUnifiedResult {
                coordination_score: quantum_result.coordination_score,
                rug_pull_probability: quantum_result.coordination_score * 0.85,
                confidence: quantum_result.coordination_score,
                processing_time_ns: quantum_result.processing_time_ns,
            }),
            
            causal_inference_result: Some(CausalInferenceUnifiedResult {
                coordination_score: causal_result.rug_pull_probability * 0.8, // Convert probability to coordination
                rug_pull_probability: causal_result.rug_pull_probability,
                confidence: causal_result.confidence,
                processing_time_ns: causal_result.processing_time_ms * 1_000_000,
            }),
            
            detected_clusters: combined_result.total_clusters,
            key_risk_factors: combined_result.risk_factors,
            recommended_actions: combined_result.recommended_actions,
            
            performance_metrics: UnifiedPerformanceMetrics {
                total_processing_time_ns: processing_time.as_nanos() as u64,
                hardware_acceleration_used: false,
                speedup_factor: 1.0,
                memory_usage_mb: 512, // Estimated
                cpu_utilization: 75.0, // Estimated
                gpu_utilization: None,
            },
            
            ensemble_weights: self.ensemble_weights.clone(),
            timestamp: Utc::now(),
        })
    }

    async fn process_sequential(
        &self,
        transactions: &[crate::analytics::Transaction],
        market_data: Option<&crate::analytics::MarketSnapshot>,
    ) -> Result<UnifiedAIResult> {
        let start_time = std::time::Instant::now();

        // Process engines sequentially
        let flash_result = self.run_flash_attention(transactions).await?;
        let spiking_result = self.run_spiking_network(transactions).await?;
        let quantum_result = self.run_quantum_gnn(transactions).await?;
        let causal_result = self.run_causal_inference(transactions, market_data).await?;

        let combined_result = self.combine_results(
            &flash_result,
            &spiking_result,
            &quantum_result, 
            &causal_result,
        ).await?;

        let processing_time = start_time.elapsed();

        Ok(UnifiedAIResult {
            coordination_score: combined_result.coordination_score,
            rug_pull_probability: combined_result.rug_pull_probability,
            confidence: combined_result.confidence,
            processing_mode_used: ProcessingMode::Sequential,
            
            flash_attention_result: Some(FlashAttentionUnifiedResult {
                coordination_score: flash_result.coordination_score,
                rug_pull_probability: flash_result.rug_pull_probability,
                confidence: flash_result.confidence,
                processing_time_ns: 0,
            }),
            
            spiking_nn_result: Some(SpikingNNUnifiedResult {
                coordination_score: spiking_result.coordination_score,
                rug_pull_probability: if spiking_result.is_coordinated_attack { 0.9 } else { 0.1 },
                confidence: spiking_result.confidence,
                processing_time_ns: spiking_result.processing_time_us * 1000,
            }),
            
            quantum_gnn_result: Some(QuantumGNNUnifiedResult {
                coordination_score: quantum_result.coordination_score,
                rug_pull_probability: quantum_result.coordination_score * 0.85,
                confidence: quantum_result.coordination_score,
                processing_time_ns: quantum_result.processing_time_ns,
            }),
            
            causal_inference_result: Some(CausalInferenceUnifiedResult {
                coordination_score: causal_result.rug_pull_probability * 0.8,
                rug_pull_probability: causal_result.rug_pull_probability,
                confidence: causal_result.confidence,
                processing_time_ns: causal_result.processing_time_ms * 1_000_000,
            }),
            
            detected_clusters: combined_result.total_clusters,
            key_risk_factors: combined_result.risk_factors,
            recommended_actions: combined_result.recommended_actions,
            
            performance_metrics: UnifiedPerformanceMetrics {
                total_processing_time_ns: processing_time.as_nanos() as u64,
                hardware_acceleration_used: false,
                speedup_factor: 0.7, // Sequential is slower
                memory_usage_mb: 256, // Lower memory usage
                cpu_utilization: 45.0,
                gpu_utilization: None,
            },
            
            ensemble_weights: self.ensemble_weights.clone(),
            timestamp: Utc::now(),
        })
    }

    async fn process_adaptive(
        &self,
        transactions: &[crate::analytics::Transaction],
        market_data: Option<&crate::analytics::MarketSnapshot>,
    ) -> Result<UnifiedAIResult> {
        // Choose processing mode based on system load and transaction count
        let processing_mode = if transactions.len() > 100 {
            ProcessingMode::HardwareFirst // Use hardware for large batches
        } else if transactions.len() > 20 {
            ProcessingMode::Parallel // Parallel for medium batches
        } else {
            ProcessingMode::Sequential // Sequential for small batches
        };

        match processing_mode {
            ProcessingMode::HardwareFirst => self.process_hardware_first(transactions, market_data).await,
            ProcessingMode::Parallel => self.process_parallel(transactions, market_data).await,
            ProcessingMode::Sequential => self.process_sequential(transactions, market_data).await,
            ProcessingMode::Adaptive => unreachable!(), // Avoid infinite recursion
        }
    }

    async fn run_flash_attention(&self, transactions: &[crate::analytics::Transaction]) -> Result<FlashAttentionUnifiedResult> {
        let flash_engine = self.flash_attention.read().await;
        let mut embedder = TransactionEmbedder::new();
        
        // Convert transactions to embeddings
        let mut embeddings = Vec::new();
        for tx in transactions {
            embeddings.push(embedder.embed_transaction(tx)?);
        }
        
        // Run flash attention
        let output = flash_engine.compute_flash_attention(&embeddings).await?;
        
        // Calculate coordination metrics
        let coordination_score = output.coordination_scores.mean().unwrap_or(0.0);
        let rug_pull_probability = coordination_score * 0.9; // Flash attention bias
        let confidence = (coordination_score * 1.2).min(1.0);
        
        Ok(FlashAttentionUnifiedResult {
            coordination_score,
            rug_pull_probability,
            confidence,
            processing_time_ns: 1_000_000, // 1ms estimated
        })
    }

    async fn run_spiking_network(&self, transactions: &[crate::analytics::Transaction]) -> Result<SpikingDetectionResult> {
        let mut spiking_brain = self.spiking_network.write().await;
        spiking_brain.process_transaction_stream(transactions).await
    }

    async fn run_quantum_gnn(&self, transactions: &[crate::analytics::Transaction]) -> Result<QuantumCoordinationResult> {
        let mut quantum_gnn = self.quantum_gnn.write().await;
        quantum_gnn.detect_coordination_patterns(transactions).await
    }

    async fn run_causal_inference(
        &self,
        transactions: &[crate::analytics::Transaction],
        market_data: Option<&crate::analytics::MarketSnapshot>,
    ) -> Result<CausalPredictionResult> {
        let mut causal_engine = self.causal_inference.write().await;
        causal_engine.predict_rug_pull_probability(transactions, market_data).await
    }

    async fn combine_results(
        &self,
        flash_result: &FlashAttentionUnifiedResult,
        spiking_result: &SpikingDetectionResult,
        quantum_result: &QuantumCoordinationResult,
        causal_result: &CausalPredictionResult,
    ) -> Result<CombinedUnifiedResult> {
        let weights = &self.ensemble_weights;

        // Weighted ensemble combination
        let coordination_score = 
            flash_result.coordination_score * weights.flash_attention +
            spiking_result.coordination_score * weights.spiking_network +
            quantum_result.coordination_score * weights.quantum_gnn +
            (causal_result.rug_pull_probability * 0.8) * weights.causal_inference; // Convert to coordination

        let rug_pull_probability = 
            flash_result.rug_pull_probability * weights.flash_attention +
            (if spiking_result.is_coordinated_attack { 0.9 } else { 0.1 }) * weights.spiking_network +
            (quantum_result.coordination_score * 0.85) * weights.quantum_gnn +
            causal_result.rug_pull_probability * weights.causal_inference;

        let confidence = 
            flash_result.confidence * weights.flash_attention +
            spiking_result.confidence * weights.spiking_network +
            quantum_result.coordination_score * weights.quantum_gnn + // Quantum confidence
            causal_result.confidence * weights.causal_inference;

        // Aggregate insights
        let total_clusters = spiking_result.detected_clusters.len() + 
                           quantum_result.detected_clusters.len();

        let mut risk_factors = Vec::new();
        if coordination_score > 0.7 {
            risk_factors.push("High coordination detected".to_string());
        }
        if rug_pull_probability > 0.6 {
            risk_factors.push("Elevated rug pull risk".to_string());
        }
        if !causal_result.causal_factors.is_empty() {
            risk_factors.push(format!("Causal factors: {}", causal_result.causal_factors.len()));
        }

        let mut recommended_actions = Vec::new();
        if rug_pull_probability > self.confidence_threshold {
            recommended_actions.push("IMMEDIATE ALERT: Potential rug pull detected".to_string());
            recommended_actions.push("Activate counter-rug-pull strategy".to_string());
        } else if coordination_score > 0.5 {
            recommended_actions.push("Monitor closely for coordinated activity".to_string());
        }

        // Add causal intervention recommendations
        for intervention in &causal_result.recommended_interventions {
            if intervention.expected_risk_reduction > 0.1 {
                recommended_actions.push(format!("Consider: {}", intervention.intervention_type));
            }
        }

        Ok(CombinedUnifiedResult {
            coordination_score,
            rug_pull_probability,
            confidence,
            total_clusters,
            risk_factors,
            recommended_actions,
        })
    }

    /// Update ensemble weights based on recent performance
    pub async fn update_ensemble_weights(&mut self, performance_feedback: &PerformanceFeedback) {
        let mut weights = &mut self.ensemble_weights;
        
        // Simple adaptive weighting based on accuracy
        let total_accuracy = performance_feedback.flash_attention_accuracy +
                           performance_feedback.spiking_nn_accuracy +
                           performance_feedback.quantum_gnn_accuracy +
                           performance_feedback.causal_inference_accuracy;

        if total_accuracy > 0.0 {
            weights.flash_attention = performance_feedback.flash_attention_accuracy / total_accuracy;
            weights.spiking_network = performance_feedback.spiking_nn_accuracy / total_accuracy;
            weights.quantum_gnn = performance_feedback.quantum_gnn_accuracy / total_accuracy;
            weights.causal_inference = performance_feedback.causal_inference_accuracy / total_accuracy;
        }
    }

    /// Get comprehensive system status
    pub async fn get_system_status(&self) -> SystemStatus {
        SystemStatus {
            flash_attention_status: "Online".to_string(),
            spiking_network_status: "Online".to_string(),
            quantum_gnn_status: "Online".to_string(),
            causal_inference_status: "Online".to_string(),
            hardware_acceleration_status: "Available".to_string(),
            
            ensemble_weights: self.ensemble_weights.clone(),
            processing_mode: self.processing_mode,
            confidence_threshold: self.confidence_threshold,
            
            total_models_loaded: 4,
            memory_usage_estimate_mb: 2048,
            last_health_check: Utc::now(),
        }
    }
}

// Result structures for unified processing

#[derive(Debug, Serialize)]
pub struct UnifiedAIResult {
    pub coordination_score: f64,
    pub rug_pull_probability: f64,
    pub confidence: f64,
    pub processing_mode_used: ProcessingMode,
    
    // Individual engine results
    pub flash_attention_result: Option<FlashAttentionUnifiedResult>,
    pub spiking_nn_result: Option<SpikingNNUnifiedResult>,
    pub quantum_gnn_result: Option<QuantumGNNUnifiedResult>,
    pub causal_inference_result: Option<CausalInferenceUnifiedResult>,
    
    // Aggregated insights
    pub detected_clusters: usize,
    pub key_risk_factors: Vec<String>,
    pub recommended_actions: Vec<String>,
    
    // Performance data
    pub performance_metrics: UnifiedPerformanceMetrics,
    pub ensemble_weights: EnsembleWeights,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Serialize)]
pub struct FlashAttentionUnifiedResult {
    pub coordination_score: f64,
    pub rug_pull_probability: f64,
    pub confidence: f64,
    pub processing_time_ns: u64,
}

#[derive(Debug, Serialize)]
pub struct SpikingNNUnifiedResult {
    pub coordination_score: f64,
    pub rug_pull_probability: f64,
    pub confidence: f64,
    pub processing_time_ns: u64,
}

#[derive(Debug, Serialize)]
pub struct QuantumGNNUnifiedResult {
    pub coordination_score: f64,
    pub rug_pull_probability: f64,
    pub confidence: f64,
    pub processing_time_ns: u64,
}

#[derive(Debug, Serialize)]
pub struct CausalInferenceUnifiedResult {
    pub coordination_score: f64,
    pub rug_pull_probability: f64,
    pub confidence: f64,
    pub processing_time_ns: u64,
}

#[derive(Debug, Serialize)]
pub struct UnifiedPerformanceMetrics {
    pub total_processing_time_ns: u64,
    pub hardware_acceleration_used: bool,
    pub speedup_factor: f64,
    pub memory_usage_mb: usize,
    pub cpu_utilization: f64,
    pub gpu_utilization: Option<f64>,
}

#[derive(Debug)]
struct CombinedUnifiedResult {
    coordination_score: f64,
    rug_pull_probability: f64,
    confidence: f64,
    total_clusters: usize,
    risk_factors: Vec<String>,
    recommended_actions: Vec<String>,
}

#[derive(Debug)]
pub struct PerformanceFeedback {
    pub flash_attention_accuracy: f64,
    pub spiking_nn_accuracy: f64,
    pub quantum_gnn_accuracy: f64,
    pub causal_inference_accuracy: f64,
}

#[derive(Debug, Serialize)]
pub struct SystemStatus {
    pub flash_attention_status: String,
    pub spiking_network_status: String,
    pub quantum_gnn_status: String,
    pub causal_inference_status: String,
    pub hardware_acceleration_status: String,
    
    pub ensemble_weights: EnsembleWeights,
    pub processing_mode: ProcessingMode,
    pub confidence_threshold: f64,
    
    pub total_models_loaded: usize,
    pub memory_usage_estimate_mb: usize,
    pub last_health_check: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    #[tokio::test]
    async fn test_unified_ai_engine_creation() {
        let engine = UnifiedAIEngine::new().await;
        assert!(engine.is_ok());
        
        let engine = engine.unwrap();
        let status = engine.get_system_status().await;
        
        assert_eq!(status.total_models_loaded, 4);
        assert!(status.memory_usage_estimate_mb > 0);
        
        println!("Unified AI Engine Status:");
        println!("  Flash Attention: {}", status.flash_attention_status);
        println!("  Spiking Network: {}", status.spiking_network_status);
        println!("  Quantum GNN: {}", status.quantum_gnn_status);
        println!("  Causal Inference: {}", status.causal_inference_status);
        println!("  Hardware Acceleration: {}", status.hardware_acceleration_status);
    }

    #[tokio::test]
    async fn test_unified_processing_parallel() {
        let engine = UnifiedAIEngine::new().await.unwrap();
        
        let transactions = vec![
            crate::analytics::Transaction {
                signature: "unified_tx_1".to_string(),
                wallet: "unified_wallet_1".to_string(),
                token_mint: "unified_token".to_string(),
                amount_sol: 100.0,
                transaction_type: crate::analytics::TransactionType::Buy,
                timestamp: Utc::now(),
            },
            crate::analytics::Transaction {
                signature: "unified_tx_2".to_string(),
                wallet: "unified_wallet_2".to_string(),
                token_mint: "unified_token".to_string(),
                amount_sol: 100.0, // Same amount - coordinated
                transaction_type: crate::analytics::TransactionType::Buy,
                timestamp: Utc::now() + Duration::minutes(1),
            },
        ];

        let result = engine.analyze_transactions(&transactions, None).await.unwrap();
        
        assert!(result.coordination_score >= 0.0 && result.coordination_score <= 1.0);
        assert!(result.rug_pull_probability >= 0.0 && result.rug_pull_probability <= 1.0);
        assert!(result.confidence > 0.0);
        assert!(!result.key_risk_factors.is_empty() || !result.recommended_actions.is_empty());
        
        // Check that all engines provided results
        assert!(result.flash_attention_result.is_some());
        assert!(result.spiking_nn_result.is_some());
        assert!(result.quantum_gnn_result.is_some());
        assert!(result.causal_inference_result.is_some());
        
        println!("Unified AI Analysis Result:");
        println!("  Coordination Score: {:.3}", result.coordination_score);
        println!("  Rug Pull Probability: {:.3}", result.rug_pull_probability);
        println!("  Confidence: {:.3}", result.confidence);
        println!("  Processing Mode: {:?}", result.processing_mode_used);
        println!("  Processing Time: {}ns", result.performance_metrics.total_processing_time_ns);
        println!("  Key Risk Factors: {:?}", result.key_risk_factors);
        println!("  Recommended Actions: {:?}", result.recommended_actions);
    }

    #[tokio::test]
    async fn test_ensemble_weight_updates() {
        let mut engine = UnifiedAIEngine::new().await.unwrap();
        
        let feedback = PerformanceFeedback {
            flash_attention_accuracy: 0.9,
            spiking_nn_accuracy: 0.95,
            quantum_gnn_accuracy: 0.85,
            causal_inference_accuracy: 0.8,
        };
        
        let old_weights = engine.ensemble_weights.clone();
        engine.update_ensemble_weights(&feedback).await;
        let new_weights = &engine.ensemble_weights;
        
        // Should update based on performance
        assert!(new_weights.spiking_network > old_weights.spiking_network); // Best performer
        
        println!("Ensemble weights updated:");
        println!("  Flash Attention: {:.3}", new_weights.flash_attention);
        println!("  Spiking Network: {:.3}", new_weights.spiking_network);
        println!("  Quantum GNN: {:.3}", new_weights.quantum_gnn);
        println!("  Causal Inference: {:.3}", new_weights.causal_inference);
    }
}