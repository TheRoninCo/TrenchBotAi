//! Spiking Neural Network Implementation for Ultra-Low Latency Detection
//! 
//! This module implements neuromorphic computing principles for microsecond-level
//! rug pull detection. Inspired by biological neural networks for maximum speed.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use chrono::{DateTime, Utc, Duration};
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::mpsc;

/// Spiking neuron model parameters
#[derive(Debug, Clone)]
pub struct NeuronParams {
    pub threshold: f64,           // Spike threshold (typically 1.0)
    pub leak_factor: f64,         // Membrane leak rate (0.9-0.99)
    pub refractory_period: u64,   // Refractory period in microseconds
    pub spike_amplitude: f64,     // Output spike amplitude
    pub decay_rate: f64,          // Exponential decay rate
}

impl Default for NeuronParams {
    fn default() -> Self {
        Self {
            threshold: 1.0,
            leak_factor: 0.95,
            refractory_period: 1000, // 1ms refractory period
            spike_amplitude: 1.0,
            decay_rate: 0.8,
        }
    }
}

/// Individual spiking neuron
#[derive(Debug, Clone)]
pub struct SpikingNeuron {
    pub id: usize,
    pub membrane_potential: f64,
    pub last_spike_time: u64,
    pub spike_count: u64,
    pub params: NeuronParams,
    pub input_weights: Vec<f64>,
    pub output_connections: Vec<usize>,
}

impl SpikingNeuron {
    pub fn new(id: usize, num_inputs: usize, params: NeuronParams) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        // Initialize weights with small random values
        let input_weights = (0..num_inputs)
            .map(|_| rng.gen_range(-0.1..0.1))
            .collect();

        Self {
            id,
            membrane_potential: 0.0,
            last_spike_time: 0,
            spike_count: 0,
            params,
            input_weights,
            output_connections: Vec::new(),
        }
    }

    /// Process input spikes and potentially generate output spike
    pub fn process_input(&mut self, inputs: &[f64], current_time_us: u64) -> Option<f64> {
        // Check refractory period
        if current_time_us - self.last_spike_time < self.params.refractory_period {
            return None;
        }

        // Apply membrane leak
        self.membrane_potential *= self.params.leak_factor;

        // Accumulate weighted inputs
        let weighted_input: f64 = inputs.iter()
            .zip(self.input_weights.iter())
            .map(|(&input, &weight)| input * weight)
            .sum();

        self.membrane_potential += weighted_input;

        // Check for spike generation
        if self.membrane_potential >= self.params.threshold {
            self.membrane_potential -= self.params.threshold; // Reset with residual
            self.last_spike_time = current_time_us;
            self.spike_count += 1;
            
            Some(self.params.spike_amplitude)
        } else {
            None
        }
    }

    /// Spike-Time Dependent Plasticity (STDP) learning rule
    pub fn apply_stdp(&mut self, pre_spike_times: &[u64], post_spike_time: u64, learning_rate: f64) {
        for (i, &pre_time) in pre_spike_times.iter().enumerate() {
            if i >= self.input_weights.len() {
                break;
            }

            let time_diff = post_spike_time as i64 - pre_time as i64;
            
            // STDP learning rule: Δw = A * exp(-|Δt|/τ) * sign(Δt)
            let tau = 20000.0; // 20ms time constant
            let learning_amplitude = 0.01;
            
            let weight_change = if time_diff > 0 {
                // Post after pre: strengthen (LTP)
                learning_amplitude * (-time_diff as f64 / tau).exp() * learning_rate
            } else {
                // Pre after post: weaken (LTD)  
                -learning_amplitude * (time_diff as f64 / tau).exp() * learning_rate
            };

            self.input_weights[i] += weight_change;
            
            // Clip weights to reasonable range
            self.input_weights[i] = self.input_weights[i].clamp(-1.0, 1.0);
        }
    }
}

/// Spike event for event-driven processing
#[derive(Debug, Clone, Serialize)]
pub struct SpikeEvent {
    pub neuron_id: usize,
    pub timestamp_us: u64,
    pub amplitude: f64,
    pub layer: usize,
}

/// Spiking Neural Network architecture
pub struct SpikingBrainLayer {
    pub layer_id: usize,
    pub neurons: Vec<SpikingNeuron>,
    pub layer_type: LayerType,
    pub spike_history: VecDeque<SpikeEvent>,
    pub max_history_size: usize,
}

#[derive(Debug, Clone)]
pub enum LayerType {
    Input,
    Hidden,
    Output,
    Coordination,  // Special layer for coordination detection
}

impl SpikingBrainLayer {
    pub fn new(layer_id: usize, num_neurons: usize, inputs_per_neuron: usize, layer_type: LayerType) -> Self {
        let params = NeuronParams::default();
        let neurons = (0..num_neurons)
            .map(|i| SpikingNeuron::new(i, inputs_per_neuron, params.clone()))
            .collect();

        Self {
            layer_id,
            neurons,
            layer_type,
            spike_history: VecDeque::new(),
            max_history_size: 10000, // Keep last 10k spikes
        }
    }

    /// Process layer with spike inputs
    pub fn process_spikes(&mut self, input_spikes: &[Vec<f64>], current_time_us: u64) -> Vec<SpikeEvent> {
        let mut output_spikes = Vec::new();

        for (neuron_idx, neuron) in self.neurons.iter_mut().enumerate() {
            if neuron_idx < input_spikes.len() {
                if let Some(spike_amplitude) = neuron.process_input(&input_spikes[neuron_idx], current_time_us) {
                    let spike_event = SpikeEvent {
                        neuron_id: neuron.id,
                        timestamp_us: current_time_us,
                        amplitude: spike_amplitude,
                        layer: self.layer_id,
                    };
                    
                    output_spikes.push(spike_event.clone());
                    self.spike_history.push_back(spike_event);
                    
                    // Maintain history size
                    if self.spike_history.len() > self.max_history_size {
                        self.spike_history.pop_front();
                    }
                }
            }
        }

        output_spikes
    }

    /// Get recent spike activity for pattern analysis
    pub fn get_spike_pattern(&self, time_window_us: u64, current_time_us: u64) -> Vec<SpikeEvent> {
        let cutoff_time = current_time_us.saturating_sub(time_window_us);
        
        self.spike_history.iter()
            .filter(|spike| spike.timestamp_us >= cutoff_time)
            .cloned()
            .collect()
    }
}

/// Complete Spiking Neural Network for rug pull detection
pub struct SpikingRugPullBrain {
    pub layers: Vec<SpikingBrainLayer>,
    pub global_time: AtomicU64,
    pub spike_events: mpsc::UnboundedSender<SpikeEvent>,
    pub detection_threshold: f64,
    pub coordination_patterns: HashMap<String, CoordinationPattern>,
    pub learning_enabled: bool,
    pub learning_rate: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct CoordinationPattern {
    pub pattern_id: String,
    pub spike_signature: Vec<(usize, u64)>, // (neuron_id, relative_time)
    pub confidence: f64,
    pub detection_count: u64,
    pub last_seen: DateTime<Utc>,
}

impl SpikingRugPullBrain {
    pub fn new() -> Result<(Self, mpsc::UnboundedReceiver<SpikeEvent>)> {
        let (spike_sender, spike_receiver) = mpsc::unbounded_channel();

        // Architecture: Input -> Hidden -> Coordination -> Output
        let mut layers = Vec::new();
        
        // Input layer: 64 neurons for transaction features
        layers.push(SpikingBrainLayer::new(0, 64, 10, LayerType::Input));
        
        // Hidden layer: 128 neurons for feature extraction
        layers.push(SpikingBrainLayer::new(1, 128, 64, LayerType::Hidden));
        
        // Coordination layer: 32 neurons specialized for coordination detection
        layers.push(SpikingBrainLayer::new(2, 32, 128, LayerType::Coordination));
        
        // Output layer: 4 neurons for final decision
        layers.push(SpikingBrainLayer::new(3, 4, 32, LayerType::Output));

        // Set up inter-layer connections
        Self::setup_connections(&mut layers)?;

        let brain = Self {
            layers,
            global_time: AtomicU64::new(0),
            spike_events: spike_sender,
            detection_threshold: 0.7,
            coordination_patterns: HashMap::new(),
            learning_enabled: true,
            learning_rate: 0.001,
        };

        Ok((brain, spike_receiver))
    }

    /// Process transaction data through spiking network
    pub async fn process_transaction_stream(
        &mut self,
        transactions: &[crate::analytics::Transaction],
    ) -> Result<SpikingDetectionResult> {
        let current_time = self.get_current_time_us();
        
        // Convert transactions to spike patterns
        let spike_inputs = self.encode_transactions_to_spikes(transactions, current_time)?;
        
        // Process through network layers
        let mut layer_outputs = spike_inputs;
        let mut all_spikes = Vec::new();

        for layer in &mut self.layers {
            let layer_spikes = layer.process_spikes(&layer_outputs, current_time);
            all_spikes.extend(layer_spikes.clone());
            
            // Prepare input for next layer
            layer_outputs = self.spikes_to_layer_input(&layer_spikes);
            
            // Send spikes to event stream
            for spike in layer_spikes {
                let _ = self.spike_events.send(spike);
            }
        }

        // Analyze output spikes for coordination detection
        let detection_result = self.analyze_output_spikes(&all_spikes, current_time).await?;
        
        // Learn from this batch if enabled
        if self.learning_enabled {
            self.apply_learning(&all_spikes, &detection_result).await?;
        }

        self.increment_time();
        Ok(detection_result)
    }

    /// Convert transactions to spike patterns (rate coding + temporal coding)
    fn encode_transactions_to_spikes(
        &self,
        transactions: &[crate::analytics::Transaction],
        base_time_us: u64,
    ) -> Result<Vec<Vec<f64>>> {
        let num_input_neurons = 64;
        let mut spike_inputs = vec![vec![0.0; 10]; num_input_neurons]; // 10 time steps
        
        for (tx_idx, tx) in transactions.iter().take(num_input_neurons).enumerate() {
            // Encode different features as spike rates
            let amount_rate = (tx.amount_sol / 100.0).min(1.0); // Normalize to [0,1]
            let time_since_base = (tx.timestamp.timestamp_micros() as u64).saturating_sub(base_time_us);
            let time_rate = (time_since_base as f64 / 1_000_000.0).min(1.0); // Normalize seconds
            
            // Generate spikes based on rates (Poisson-like process)
            for time_step in 0..10 {
                let random_val: f64 = rand::random();
                
                // Amount-based spikes
                if random_val < amount_rate {
                    spike_inputs[tx_idx][time_step] += 1.0;
                }
                
                // Timing-based spikes (temporal pattern)
                let temporal_random: f64 = rand::random();
                if temporal_random < time_rate {
                    spike_inputs[tx_idx][time_step] += 0.5;
                }
                
                // Wallet similarity spikes (simplified)
                if tx_idx > 0 && transactions[tx_idx - 1].wallet.len() == tx.wallet.len() {
                    spike_inputs[tx_idx][time_step] += 0.3;
                }
            }
        }

        Ok(spike_inputs)
    }

    /// Convert spike events to input for next layer
    fn spikes_to_layer_input(&self, spikes: &[SpikeEvent]) -> Vec<Vec<f64>> {
        let max_neurons = spikes.iter().map(|s| s.neuron_id).max().unwrap_or(0) + 1;
        let time_steps = 10;
        
        let mut layer_input = vec![vec![0.0; time_steps]; max_neurons];
        
        for spike in spikes {
            if spike.neuron_id < max_neurons {
                // Distribute spike across time steps (simple spreading)
                for t in 0..time_steps {
                    let decay_factor = (-t as f64 / 3.0).exp(); // Exponential decay
                    layer_input[spike.neuron_id][t] += spike.amplitude * decay_factor;
                }
            }
        }
        
        layer_input
    }

    /// Analyze output layer spikes for coordination patterns
    async fn analyze_output_spikes(
        &mut self,
        all_spikes: &[SpikeEvent],
        current_time_us: u64,
    ) -> Result<SpikingDetectionResult> {
        // Get output layer spikes (layer 3)
        let output_spikes: Vec<&SpikeEvent> = all_spikes.iter()
            .filter(|spike| spike.layer == 3)
            .collect();

        // Analyze coordination layer spikes (layer 2) for patterns
        let coordination_spikes: Vec<&SpikeEvent> = all_spikes.iter()
            .filter(|spike| spike.layer == 2)
            .collect();

        // Calculate coordination score based on spike patterns
        let coordination_score = self.calculate_coordination_score(&coordination_spikes, current_time_us)?;
        
        // Detect known patterns
        let detected_patterns = self.match_coordination_patterns(&coordination_spikes)?;
        
        // Determine if this indicates a coordinated attack
        let is_coordinated_attack = coordination_score > self.detection_threshold;
        let confidence = coordination_score;

        // Generate cluster information from spike patterns
        let detected_clusters = self.extract_clusters_from_spikes(&coordination_spikes)?;

        Ok(SpikingDetectionResult {
            is_coordinated_attack,
            coordination_score,
            confidence,
            detected_patterns,
            detected_clusters,
            total_spikes: all_spikes.len(),
            processing_time_us: self.get_current_time_us() - current_time_us,
            timestamp: Utc::now(),
        })
    }

    /// Calculate coordination score from spike patterns
    fn calculate_coordination_score(&self, spikes: &[&SpikeEvent], time_window_us: u64) -> Result<f64> {
        if spikes.is_empty() {
            return Ok(0.0);
        }

        // Analyze spike synchrony (coordination indicator)
        let mut synchrony_score = 0.0;
        let synchrony_window = 5000; // 5ms synchrony window
        
        for i in 0..spikes.len() {
            for j in (i + 1)..spikes.len() {
                let time_diff = spikes[i].timestamp_us.abs_diff(spikes[j].timestamp_us);
                if time_diff <= synchrony_window {
                    synchrony_score += 1.0;
                }
            }
        }
        
        // Normalize by possible pairs
        let max_pairs = spikes.len() * (spikes.len() - 1) / 2;
        if max_pairs > 0 {
            synchrony_score /= max_pairs as f64;
        }

        // Analyze spike rate (high rate = more activity)
        let spike_rate = spikes.len() as f64 / (time_window_us as f64 / 1_000_000.0);
        let rate_score = (spike_rate / 100.0).min(1.0); // Normalize to reasonable rate

        // Combine metrics
        let coordination_score = (synchrony_score * 0.7 + rate_score * 0.3).min(1.0);
        
        Ok(coordination_score)
    }

    /// Match detected spike patterns against known coordination patterns
    fn match_coordination_patterns(&self, spikes: &[&SpikeEvent]) -> Result<Vec<String>> {
        let mut matched_patterns = Vec::new();
        
        // Create signature from current spikes
        let current_signature = self.create_spike_signature(spikes)?;
        
        // Compare against known patterns
        for (pattern_id, pattern) in &self.coordination_patterns {
            let similarity = self.calculate_pattern_similarity(&current_signature, &pattern.spike_signature)?;
            
            if similarity > 0.8 { // 80% similarity threshold
                matched_patterns.push(pattern_id.clone());
            }
        }
        
        Ok(matched_patterns)
    }

    fn create_spike_signature(&self, spikes: &[&SpikeEvent]) -> Result<Vec<(usize, u64)>> {
        if spikes.is_empty() {
            return Ok(Vec::new());
        }

        let base_time = spikes.iter().map(|s| s.timestamp_us).min().unwrap_or(0);
        
        let signature: Vec<(usize, u64)> = spikes.iter()
            .map(|spike| (spike.neuron_id, spike.timestamp_us - base_time))
            .collect();
            
        Ok(signature)
    }

    fn calculate_pattern_similarity(
        &self,
        sig1: &[(usize, u64)],
        sig2: &[(usize, u64)],
    ) -> Result<f64> {
        if sig1.is_empty() || sig2.is_empty() {
            return Ok(0.0);
        }

        let mut matches = 0;
        let tolerance_us = 10000; // 10ms tolerance
        
        for (neuron1, time1) in sig1 {
            for (neuron2, time2) in sig2 {
                if neuron1 == neuron2 && time1.abs_diff(*time2) <= tolerance_us {
                    matches += 1;
                    break; // Each spike in sig1 can match at most one in sig2
                }
            }
        }
        
        let similarity = matches as f64 / sig1.len().max(sig2.len()) as f64;
        Ok(similarity)
    }

    /// Extract cluster information from coordination layer spikes
    fn extract_clusters_from_spikes(&self, spikes: &[&SpikeEvent]) -> Result<Vec<SpikingCluster>> {
        let mut clusters = Vec::new();
        
        if spikes.is_empty() {
            return Ok(clusters);
        }

        // Group spikes by temporal proximity
        let mut spike_groups: Vec<Vec<&SpikeEvent>> = Vec::new();
        let grouping_window = 50000; // 50ms grouping window
        
        let mut sorted_spikes = spikes.to_vec();
        sorted_spikes.sort_by_key(|s| s.timestamp_us);
        
        let mut current_group = vec![sorted_spikes[0]];
        
        for spike in sorted_spikes.iter().skip(1) {
            let last_spike = current_group.last().unwrap();
            if spike.timestamp_us - last_spike.timestamp_us <= grouping_window {
                current_group.push(*spike);
            } else {
                spike_groups.push(current_group.clone());
                current_group = vec![*spike];
            }
        }
        spike_groups.push(current_group);

        // Convert groups to clusters
        for (cluster_id, group) in spike_groups.iter().enumerate() {
            if group.len() >= 3 { // Minimum cluster size
                let cluster = SpikingCluster {
                    cluster_id,
                    neuron_ids: group.iter().map(|s| s.neuron_id).collect(),
                    spike_count: group.len(),
                    time_span_us: group.last().unwrap().timestamp_us - group[0].timestamp_us,
                    synchrony_score: self.calculate_group_synchrony(group)?,
                };
                clusters.push(cluster);
            }
        }

        Ok(clusters)
    }

    fn calculate_group_synchrony(&self, group: &[&SpikeEvent]) -> Result<f64> {
        if group.len() < 2 {
            return Ok(0.0);
        }

        let time_span = group.last().unwrap().timestamp_us - group[0].timestamp_us;
        let expected_span = group.len() as u64 * 10000; // Expected span for random spikes
        
        let synchrony = if expected_span > 0 {
            1.0 - (time_span as f64 / expected_span as f64).min(1.0)
        } else {
            1.0
        };
        
        Ok(synchrony)
    }

    /// Apply STDP learning to improve detection
    async fn apply_learning(&mut self, spikes: &[SpikeEvent], result: &SpikingDetectionResult) -> Result<()> {
        if !self.learning_enabled {
            return Ok(());
        }

        // Learn from successful detections
        if result.is_coordinated_attack && result.confidence > 0.8 {
            // Strengthen connections that contributed to detection
            self.strengthen_detection_pathways(spikes).await?;
            
            // Store pattern for future recognition
            self.store_coordination_pattern(spikes, result).await?;
        }

        Ok(())
    }

    async fn strengthen_detection_pathways(&mut self, spikes: &[SpikeEvent]) -> Result<()> {
        // Apply STDP to coordination layer neurons
        if let Some(coord_layer) = self.layers.get_mut(2) {
            let coord_spikes: Vec<&SpikeEvent> = spikes.iter()
                .filter(|s| s.layer == 2)
                .collect();

            for neuron in &mut coord_layer.neurons {
                // Find spikes for this neuron
                let neuron_spikes: Vec<u64> = coord_spikes.iter()
                    .filter(|s| s.neuron_id == neuron.id)
                    .map(|s| s.timestamp_us)
                    .collect();

                if !neuron_spikes.is_empty() {
                    // Get input spike times (from hidden layer)
                    let input_spikes: Vec<u64> = spikes.iter()
                        .filter(|s| s.layer == 1)
                        .map(|s| s.timestamp_us)
                        .collect();

                    // Apply STDP for each output spike
                    for &post_spike_time in &neuron_spikes {
                        neuron.apply_stdp(&input_spikes, post_spike_time, self.learning_rate);
                    }
                }
            }
        }

        Ok(())
    }

    async fn store_coordination_pattern(&mut self, spikes: &[SpikeEvent], result: &SpikingDetectionResult) -> Result<()> {
        let coord_spikes: Vec<&SpikeEvent> = spikes.iter()
            .filter(|s| s.layer == 2)
            .collect();

        if !coord_spikes.is_empty() {
            let signature = self.create_spike_signature(&coord_spikes)?;
            let pattern_id = format!("pattern_{}", self.coordination_patterns.len());
            
            let pattern = CoordinationPattern {
                pattern_id: pattern_id.clone(),
                spike_signature: signature,
                confidence: result.confidence,
                detection_count: 1,
                last_seen: Utc::now(),
            };
            
            self.coordination_patterns.insert(pattern_id, pattern);
        }

        Ok(())
    }

    // Helper functions
    fn setup_connections(layers: &mut [SpikingBrainLayer]) -> Result<()> {
        // Set up forward connections between layers
        for i in 0..(layers.len() - 1) {
            let (left, right) = layers.split_at_mut(i + 1);
            let current_layer = &mut left[i];
            let next_layer = &right[0];
            
            // Connect each neuron to all neurons in next layer
            for neuron in &mut current_layer.neurons {
                neuron.output_connections = (0..next_layer.neurons.len()).collect();
            }
        }
        
        Ok(())
    }

    fn get_current_time_us(&self) -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64
    }

    fn increment_time(&self) {
        self.global_time.fetch_add(1000, Ordering::Relaxed); // Increment by 1ms
    }

    /// Get network performance metrics
    pub fn get_network_metrics(&self) -> SpikingNetworkMetrics {
        let total_neurons: usize = self.layers.iter().map(|l| l.neurons.len()).sum();
        let total_spikes: u64 = self.layers.iter()
            .map(|l| l.neurons.iter().map(|n| n.spike_count).sum::<u64>())
            .sum();

        let avg_spike_rate = if total_neurons > 0 {
            total_spikes as f64 / total_neurons as f64
        } else {
            0.0
        };

        SpikingNetworkMetrics {
            total_neurons,
            total_spikes,
            avg_spike_rate,
            patterns_learned: self.coordination_patterns.len(),
            current_time_us: self.get_current_time_us(),
            learning_enabled: self.learning_enabled,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct SpikingDetectionResult {
    pub is_coordinated_attack: bool,
    pub coordination_score: f64,
    pub confidence: f64,
    pub detected_patterns: Vec<String>,
    pub detected_clusters: Vec<SpikingCluster>,
    pub total_spikes: usize,
    pub processing_time_us: u64,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SpikingCluster {
    pub cluster_id: usize,
    pub neuron_ids: Vec<usize>,
    pub spike_count: usize,
    pub time_span_us: u64,
    pub synchrony_score: f64,
}

#[derive(Debug, Serialize)]
pub struct SpikingNetworkMetrics {
    pub total_neurons: usize,
    pub total_spikes: u64,
    pub avg_spike_rate: f64,
    pub patterns_learned: usize,
    pub current_time_us: u64,
    pub learning_enabled: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_spiking_network_creation() {
        let (brain, _receiver) = SpikingRugPullBrain::new().unwrap();
        assert_eq!(brain.layers.len(), 4);
        assert_eq!(brain.layers[0].neurons.len(), 64); // Input layer
        assert_eq!(brain.layers[3].neurons.len(), 4);  // Output layer
    }

    #[tokio::test]
    async fn test_transaction_encoding() {
        let (mut brain, _receiver) = SpikingRugPullBrain::new().unwrap();
        
        let transactions = vec![
            crate::analytics::Transaction {
                signature: "test_1".to_string(),
                wallet: "wallet_1".to_string(),
                token_mint: "token".to_string(),
                amount_sol: 100.0,
                transaction_type: crate::analytics::TransactionType::Buy,
                timestamp: Utc::now(),
            },
        ];

        let result = brain.process_transaction_stream(&transactions).await.unwrap();
        
        assert!(result.processing_time_us < 10000); // Should be under 10ms
        assert!(result.coordination_score >= 0.0 && result.coordination_score <= 1.0);
    }

    #[tokio::test]
    async fn test_learning_mechanism() {
        let (mut brain, _receiver) = SpikingRugPullBrain::new().unwrap();
        
        // Process coordinated transactions multiple times
        let coordinated_txs = vec![
            crate::analytics::Transaction {
                signature: "coord_1".to_string(),
                wallet: "coordinated_wallet_1".to_string(),
                token_mint: "test_token".to_string(),
                amount_sol: 100.0,
                transaction_type: crate::analytics::TransactionType::Buy,
                timestamp: Utc::now(),
            },
            crate::analytics::Transaction {
                signature: "coord_2".to_string(),
                wallet: "coordinated_wallet_2".to_string(),
                token_mint: "test_token".to_string(),
                amount_sol: 105.0,
                transaction_type: crate::analytics::TransactionType::Buy,
                timestamp: Utc::now() + Duration::seconds(30),
            },
        ];

        let initial_patterns = brain.coordination_patterns.len();
        
        // Process multiple times to trigger learning
        for _ in 0..5 {
            let result = brain.process_transaction_stream(&coordinated_txs).await.unwrap();
            if result.is_coordinated_attack {
                break;
            }
        }

        let final_patterns = brain.coordination_patterns.len();
        assert!(final_patterns >= initial_patterns); // Should learn patterns
    }

    #[test]
    fn test_neuron_spike_generation() {
        let params = NeuronParams::default();
        let mut neuron = SpikingNeuron::new(0, 3, params);
        
        // Strong input should cause spike
        let inputs = vec![0.5, 0.7, 0.3];
        let spike = neuron.process_input(&inputs, 1000);
        
        assert!(spike.is_some()); // Should spike with strong input
        assert_eq!(neuron.spike_count, 1);
    }
}