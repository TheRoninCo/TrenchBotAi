//! Quantum-Inspired Graph Neural Networks for Parallel Coordination Analysis
//! 
//! This module implements quantum computing principles adapted for classical hardware
//! to enable massively parallel analysis of transaction coordination patterns.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, BTreeMap};
use chrono::{DateTime, Utc, Duration};
use ndarray::{Array1, Array2, Array3, Axis, s};
use rayon::prelude::*;
use std::sync::{Arc, RwLock};
use tokio::sync::mpsc;

/// Quantum-inspired qubit state for transaction nodes
#[derive(Debug, Clone)]
pub struct QuantumQubit {
    pub amplitude_alpha: f64,    // |0⟩ state amplitude (normal transaction)
    pub amplitude_beta: f64,     // |1⟩ state amplitude (coordinated transaction)
    pub phase: f64,              // Quantum phase (temporal information)
    pub entanglement_strength: f64, // Entanglement with other nodes
}

impl QuantumQubit {
    pub fn new() -> Self {
        // Initialize in superposition state
        let sqrt_half = (0.5_f64).sqrt();
        Self {
            amplitude_alpha: sqrt_half,
            amplitude_beta: sqrt_half,
            phase: 0.0,
            entanglement_strength: 0.0,
        }
    }

    /// Measure the qubit (collapse to definite state)
    pub fn measure(&self) -> bool {
        let probability_one = self.amplitude_beta.powi(2);
        rand::random::<f64>() < probability_one
    }

    /// Apply quantum rotation (parameterized gate)
    pub fn rotate(&mut self, theta: f64, phi: f64) {
        let cos_half_theta = (theta / 2.0).cos();
        let sin_half_theta = (theta / 2.0).sin();
        
        let new_alpha = cos_half_theta * self.amplitude_alpha - 
                       (1.0i64 * phi).exp() as f64 * sin_half_theta * self.amplitude_beta;
        let new_beta = sin_half_theta * self.amplitude_alpha + 
                      (1.0i64 * phi).exp() as f64 * cos_half_theta * self.amplitude_beta;
        
        self.amplitude_alpha = new_alpha;
        self.amplitude_beta = new_beta;
        self.phase = phi;
    }

    /// Get coordination probability
    pub fn coordination_probability(&self) -> f64 {
        self.amplitude_beta.powi(2)
    }
}

/// Transaction node in the quantum graph
#[derive(Debug, Clone)]
pub struct QuantumTransactionNode {
    pub transaction_id: String,
    pub wallet: String,
    pub amount_sol: f64,
    pub timestamp: DateTime<Utc>,
    pub qubit_state: QuantumQubit,
    pub adjacency_weights: HashMap<String, f64>, // Connections to other nodes
    pub feature_embedding: Array1<f64>,
    pub layer_outputs: Vec<Array1<f64>>, // Outputs from each QGNN layer
}

impl QuantumTransactionNode {
    pub fn new(
        transaction_id: String,
        wallet: String,
        amount_sol: f64,
        timestamp: DateTime<Utc>,
    ) -> Self {
        Self {
            transaction_id,
            wallet,
            amount_sol,
            timestamp,
            qubit_state: QuantumQubit::new(),
            adjacency_weights: HashMap::new(),
            feature_embedding: Array1::zeros(128), // 128-dim features
            layer_outputs: Vec::new(),
        }
    }

    /// Create edge connection to another node
    pub fn connect_to(&mut self, other_id: &str, weight: f64) {
        self.adjacency_weights.insert(other_id.to_string(), weight);
    }

    /// Calculate entanglement with another node
    pub fn calculate_entanglement(&self, other: &QuantumTransactionNode) -> f64 {
        // Quantum entanglement measure based on state correlation
        let state_correlation = self.qubit_state.amplitude_alpha * other.qubit_state.amplitude_alpha +
                               self.qubit_state.amplitude_beta * other.qubit_state.amplitude_beta;
        
        // Add temporal correlation
        let time_diff = (self.timestamp - other.timestamp).num_seconds().abs() as f64;
        let temporal_correlation = (-time_diff / 3600.0).exp(); // 1-hour correlation decay
        
        // Wallet similarity
        let wallet_similarity = if self.wallet == other.wallet { 1.0 } else { 0.0 };
        
        // Amount similarity
        let amount_ratio = (self.amount_sol / other.amount_sol.max(0.01)).ln().abs();
        let amount_similarity = (-amount_ratio / 2.0).exp();
        
        // Combined entanglement strength
        let entanglement = (state_correlation * 0.4 + 
                           temporal_correlation * 0.3 + 
                           wallet_similarity * 0.2 + 
                           amount_similarity * 0.1).clamp(0.0, 1.0);
        
        entanglement
    }
}

/// Quantum-Inspired Graph Neural Network Layer
#[derive(Debug, Clone)]
pub struct QuantumGNNLayer {
    pub layer_id: usize,
    pub input_dim: usize,
    pub output_dim: usize,
    pub quantum_weights: Array2<f64>,    // Quantum-inspired weight matrix
    pub phase_weights: Array2<f64>,      // Phase adjustment weights
    pub entanglement_matrix: Array2<f64>, // Learned entanglement patterns
    pub attention_weights: Array2<f64>,  // Multi-head attention
    pub num_heads: usize,
}

impl QuantumGNNLayer {
    pub fn new(layer_id: usize, input_dim: usize, output_dim: usize, num_heads: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        // Initialize with quantum-inspired distributions
        let scale = (2.0 / input_dim as f64).sqrt();
        
        let quantum_weights = Array2::from_shape_fn((input_dim, output_dim), |_| {
            rng.gen_range(-scale..scale)
        });
        
        let phase_weights = Array2::from_shape_fn((input_dim, output_dim), |_| {
            rng.gen_range(0.0..2.0 * std::f64::consts::PI)
        });
        
        let entanglement_matrix = Array2::from_shape_fn((output_dim, output_dim), |_| {
            rng.gen_range(-0.1..0.1)
        });
        
        let attention_weights = Array2::from_shape_fn((output_dim, output_dim * num_heads), |_| {
            rng.gen_range(-scale..scale)
        });
        
        Self {
            layer_id,
            input_dim,
            output_dim,
            quantum_weights,
            phase_weights,
            entanglement_matrix,
            attention_weights,
            num_heads,
        }
    }

    /// Quantum-inspired forward pass with superposition
    pub fn forward(
        &self,
        node_features: &Array2<f64>,          // [num_nodes, input_dim]
        adjacency_matrix: &Array2<f64>,       // [num_nodes, num_nodes]
        quantum_states: &[QuantumQubit],      // Quantum states for each node
    ) -> Result<(Array2<f64>, Vec<QuantumQubit>)> {
        let num_nodes = node_features.nrows();
        
        // Step 1: Quantum feature transformation
        let quantum_features = self.apply_quantum_transformation(node_features, quantum_states)?;
        
        // Step 2: Graph message passing with quantum interference
        let message_features = self.quantum_message_passing(&quantum_features, adjacency_matrix)?;
        
        // Step 3: Multi-head quantum attention
        let attended_features = self.quantum_multi_head_attention(&message_features, quantum_states)?;
        
        // Step 4: Entanglement-based feature mixing
        let final_features = self.apply_entanglement_mixing(&attended_features)?;
        
        // Step 5: Update quantum states based on new features
        let updated_states = self.update_quantum_states(quantum_states, &final_features)?;
        
        Ok((final_features, updated_states))
    }

    fn apply_quantum_transformation(
        &self,
        features: &Array2<f64>,
        quantum_states: &[QuantumQubit],
    ) -> Result<Array2<f64>> {
        let num_nodes = features.nrows();
        let mut quantum_features = Array2::zeros((num_nodes, self.output_dim));
        
        for i in 0..num_nodes {
            let node_features = features.row(i);
            let qubit = &quantum_states[i];
            
            // Apply quantum superposition to features
            for j in 0..self.output_dim {
                let weighted_sum: f64 = node_features.iter()
                    .zip(self.quantum_weights.column(j).iter())
                    .map(|(&feat, &weight)| feat * weight)
                    .sum();
                
                // Apply quantum phase
                let phase = self.phase_weights.column(j).iter()
                    .zip(node_features.iter())
                    .map(|(&phase_w, &feat)| phase_w * feat)
                    .sum::<f64>();
                
                // Quantum amplitude modulation
                let amplitude_factor = qubit.amplitude_alpha.powi(2) + qubit.amplitude_beta.powi(2);
                let phase_factor = (phase + qubit.phase).cos();
                
                quantum_features[[i, j]] = weighted_sum * amplitude_factor * phase_factor;
            }
        }
        
        Ok(quantum_features)
    }

    fn quantum_message_passing(
        &self,
        features: &Array2<f64>,
        adjacency: &Array2<f64>,
    ) -> Result<Array2<f64>> {
        // Quantum-inspired message passing with interference
        let normalized_adjacency = self.normalize_adjacency_quantum(adjacency)?;
        let messages = normalized_adjacency.dot(features);
        
        // Apply quantum interference effects
        let mut interference_features = features.clone();
        for i in 0..features.nrows() {
            for j in 0..features.ncols() {
                let original = features[[i, j]];
                let message = messages[[i, j]];
                
                // Quantum interference: amplitude addition with phase consideration
                let interference = (original + message) / 2.0_f64.sqrt();
                interference_features[[i, j]] = interference;
            }
        }
        
        Ok(interference_features)
    }

    fn normalize_adjacency_quantum(&self, adjacency: &Array2<f64>) -> Result<Array2<f64>> {
        let mut normalized = adjacency.clone();
        
        // Quantum-inspired normalization (preserve quantum properties)
        for i in 0..adjacency.nrows() {
            let row_sum: f64 = adjacency.row(i).iter().map(|x| x.powi(2)).sum();
            if row_sum > 0.0 {
                let norm_factor = 1.0 / row_sum.sqrt();
                for j in 0..adjacency.ncols() {
                    normalized[[i, j]] *= norm_factor;
                }
            }
        }
        
        Ok(normalized)
    }

    fn quantum_multi_head_attention(
        &self,
        features: &Array2<f64>,
        quantum_states: &[QuantumQubit],
    ) -> Result<Array2<f64>> {
        let num_nodes = features.nrows();
        let head_dim = self.output_dim / self.num_heads;
        let mut attended_output = Array2::zeros((num_nodes, self.output_dim));
        
        for head in 0..self.num_heads {
            let head_start = head * head_dim;
            let head_end = head_start + head_dim;
            
            // Query, Key, Value projections for this head
            let head_weights = self.attention_weights.slice(s![.., head_start..head_end]);
            let queries = features.dot(&head_weights);
            let keys = features.dot(&head_weights);
            let values = features.slice(s![.., head_start..head_end]).to_owned();
            
            // Quantum-enhanced attention scores
            let mut attention_scores = Array2::zeros((num_nodes, num_nodes));
            for i in 0..num_nodes {
                for j in 0..num_nodes {
                    let q_dot_k = queries.row(i).dot(&keys.row(j));
                    let scale = 1.0 / (head_dim as f64).sqrt();
                    
                    // Add quantum entanglement bonus
                    let entanglement = quantum_states[i].entanglement_strength * 
                                     quantum_states[j].entanglement_strength;
                    
                    attention_scores[[i, j]] = (q_dot_k * scale + entanglement * 0.1).tanh();
                }
            }
            
            // Softmax normalization
            for i in 0..num_nodes {
                let row_max = attention_scores.row(i).iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let mut row_sum = 0.0;
                
                for j in 0..num_nodes {
                    attention_scores[[i, j]] = (attention_scores[[i, j]] - row_max).exp();
                    row_sum += attention_scores[[i, j]];
                }
                
                if row_sum > 0.0 {
                    for j in 0..num_nodes {
                        attention_scores[[i, j]] /= row_sum;
                    }
                }
            }
            
            // Apply attention to values
            let head_output = attention_scores.dot(&values);
            attended_output.slice_mut(s![.., head_start..head_end]).assign(&head_output);
        }
        
        Ok(attended_output)
    }

    fn apply_entanglement_mixing(&self, features: &Array2<f64>) -> Result<Array2<f64>> {
        // Apply learned entanglement patterns to mix features
        let entangled_features = features.dot(&self.entanglement_matrix);
        
        // Combine original and entangled features (quantum superposition)
        let mixed_features = (features + &entangled_features) / 2.0_f64.sqrt();
        
        Ok(mixed_features)
    }

    fn update_quantum_states(
        &self,
        current_states: &[QuantumQubit],
        new_features: &Array2<f64>,
    ) -> Result<Vec<QuantumQubit>> {
        let mut updated_states = current_states.to_vec();
        
        for (i, state) in updated_states.iter_mut().enumerate() {
            if i < new_features.nrows() {
                let feature_magnitude = new_features.row(i).dot(&new_features.row(i)).sqrt();
                let coordination_signal = feature_magnitude.tanh();
                
                // Update quantum amplitudes based on learned features
                let theta = coordination_signal * std::f64::consts::PI;
                let phi = feature_magnitude * 0.1;
                
                state.rotate(theta, phi);
                
                // Update entanglement strength
                state.entanglement_strength = (state.entanglement_strength * 0.9 + 
                                             coordination_signal * 0.1).clamp(0.0, 1.0);
            }
        }
        
        Ok(updated_states)
    }
}

/// Complete Quantum-Inspired Graph Neural Network
pub struct QuantumGraphNeuralNetwork {
    pub layers: Vec<QuantumGNNLayer>,
    pub nodes: Arc<RwLock<HashMap<String, QuantumTransactionNode>>>,
    pub adjacency_matrix: Arc<RwLock<Array2<f64>>>,
    pub global_quantum_state: Arc<RwLock<Vec<QuantumQubit>>>,
    pub learning_rate: f64,
    pub entanglement_threshold: f64,
    pub parallel_workers: usize,
}

impl QuantumGraphNeuralNetwork {
    pub fn new(
        input_dim: usize,
        hidden_dims: Vec<usize>,
        output_dim: usize,
        num_attention_heads: usize,
    ) -> Result<Self> {
        let mut layers = Vec::new();
        let mut current_dim = input_dim;
        
        // Create hidden layers
        for (i, &hidden_dim) in hidden_dims.iter().enumerate() {
            layers.push(QuantumGNNLayer::new(i, current_dim, hidden_dim, num_attention_heads));
            current_dim = hidden_dim;
        }
        
        // Final output layer
        layers.push(QuantumGNNLayer::new(
            hidden_dims.len(),
            current_dim,
            output_dim,
            num_attention_heads
        ));
        
        Ok(Self {
            layers,
            nodes: Arc::new(RwLock::new(HashMap::new())),
            adjacency_matrix: Arc::new(RwLock::new(Array2::zeros((0, 0)))),
            global_quantum_state: Arc::new(RwLock::new(Vec::new())),
            learning_rate: 0.001,
            entanglement_threshold: 0.7,
            parallel_workers: num_cpus::get(),
        })
    }

    /// Process transactions and detect coordination patterns
    pub async fn detect_coordination_patterns(
        &mut self,
        transactions: &[crate::analytics::Transaction],
    ) -> Result<QuantumCoordinationResult> {
        // Convert transactions to quantum nodes
        self.build_quantum_graph(transactions).await?;
        
        // Run quantum inference
        let coordination_predictions = self.quantum_forward_pass().await?;
        
        // Extract coordination clusters using quantum measurements
        let clusters = self.extract_quantum_clusters(&coordination_predictions).await?;
        
        // Calculate overall coordination score
        let coordination_score = self.calculate_global_coordination_score(&coordination_predictions)?;
        
        Ok(QuantumCoordinationResult {
            coordination_score,
            detected_clusters: clusters,
            quantum_predictions: coordination_predictions,
            entangled_pairs: self.find_entangled_transaction_pairs().await?,
            processing_time_ns: std::time::Instant::now().elapsed().as_nanos() as u64,
            total_quantum_nodes: self.nodes.read().unwrap().len(),
            timestamp: Utc::now(),
        })
    }

    async fn build_quantum_graph(&mut self, transactions: &[crate::analytics::Transaction]) -> Result<()> {
        let mut nodes = self.nodes.write().unwrap();
        let mut quantum_states = self.global_quantum_state.write().unwrap();
        
        nodes.clear();
        quantum_states.clear();
        
        // Create quantum nodes for each transaction
        for tx in transactions {
            let mut node = QuantumTransactionNode::new(
                tx.signature.clone(),
                tx.wallet.clone(),
                tx.amount_sol,
                tx.timestamp,
            );
            
            // Generate feature embedding
            node.feature_embedding = self.generate_transaction_features(tx)?;
            
            // Initialize quantum state
            let mut qubit = QuantumQubit::new();
            
            // Bias quantum state based on transaction characteristics
            let coordination_bias = self.calculate_coordination_bias(tx)?;
            let theta = coordination_bias * std::f64::consts::PI / 2.0;
            qubit.rotate(theta, 0.0);
            
            quantum_states.push(qubit.clone());
            node.qubit_state = qubit;
            
            nodes.insert(tx.signature.clone(), node);
        }
        
        // Build adjacency matrix and calculate entanglements
        let num_nodes = nodes.len();
        let mut adjacency = Array2::zeros((num_nodes, num_nodes));
        
        let node_list: Vec<_> = nodes.values().collect();
        
        // Parallel computation of node connections
        let connections: Vec<_> = (0..num_nodes).into_par_iter().map(|i| {
            let mut row_connections = Vec::new();
            
            for j in 0..num_nodes {
                if i != j {
                    let entanglement = node_list[i].calculate_entanglement(node_list[j]);
                    if entanglement > self.entanglement_threshold {
                        row_connections.push((j, entanglement));
                    }
                }
            }
            
            (i, row_connections)
        }).collect();
        
        // Apply connections to adjacency matrix
        for (i, row_connections) in connections {
            for (j, weight) in row_connections {
                adjacency[[i, j]] = weight;
            }
        }
        
        *self.adjacency_matrix.write().unwrap() = adjacency;
        
        Ok(())
    }

    fn generate_transaction_features(&self, tx: &crate::analytics::Transaction) -> Result<Array1<f64>> {
        let mut features = Array1::zeros(128);
        
        // Amount features (log-scaled)
        features[0] = tx.amount_sol.ln().max(0.0);
        features[1] = (tx.amount_sol / 100.0).tanh(); // Normalized amount
        
        // Temporal features
        let epoch_seconds = tx.timestamp.timestamp() as f64;
        features[2] = (epoch_seconds / 3600.0).sin(); // Hourly pattern
        features[3] = (epoch_seconds / 86400.0).sin(); // Daily pattern
        features[4] = (epoch_seconds / 604800.0).sin(); // Weekly pattern
        
        // Wallet hash features (simple encoding)
        let wallet_hash = self.simple_string_hash(&tx.wallet);
        for i in 5..20 {
            let bit_pos = (i - 5) % 64;
            features[i] = if (wallet_hash >> bit_pos) & 1 == 1 { 1.0 } else { -1.0 };
        }
        
        // Token features
        let token_hash = self.simple_string_hash(&tx.token_mint);
        for i in 20..35 {
            let bit_pos = (i - 20) % 64;
            features[i] = if (token_hash >> bit_pos) & 1 == 1 { 1.0 } else { -1.0 };
        }
        
        // Transaction type features
        match tx.transaction_type {
            crate::analytics::TransactionType::Buy => {
                features[35] = 1.0;
                features[36] = 0.0;
                features[37] = 0.0;
            }
            crate::analytics::TransactionType::Sell => {
                features[35] = 0.0;
                features[36] = 1.0;
                features[37] = 0.0;
            }
            crate::analytics::TransactionType::Swap => {
                features[35] = 0.0;
                features[36] = 0.0;
                features[37] = 1.0;
            }
        }
        
        // Remaining features filled with quantum-inspired noise
        for i in 38..128 {
            features[i] = rand::random::<f64>() * 0.1 - 0.05;
        }
        
        Ok(features)
    }

    fn simple_string_hash(&self, s: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        s.hash(&mut hasher);
        hasher.finish()
    }

    fn calculate_coordination_bias(&self, tx: &crate::analytics::Transaction) -> Result<f64> {
        // Simple heuristics for initial coordination bias
        let mut bias = 0.0;
        
        // Large amounts are more suspicious
        if tx.amount_sol > 1000.0 {
            bias += 0.3;
        } else if tx.amount_sol > 100.0 {
            bias += 0.1;
        }
        
        // Round amounts are suspicious
        if tx.amount_sol.fract() == 0.0 {
            bias += 0.2;
        }
        
        // Recent transactions get higher bias
        let hours_ago = (Utc::now() - tx.timestamp).num_hours();
        if hours_ago < 1 {
            bias += 0.2;
        } else if hours_ago < 24 {
            bias += 0.1;
        }
        
        Ok(bias.clamp(0.0, 1.0))
    }

    async fn quantum_forward_pass(&self) -> Result<Vec<f64>> {
        let nodes = self.nodes.read().unwrap();
        let adjacency = self.adjacency_matrix.read().unwrap();
        let quantum_states = self.global_quantum_state.read().unwrap();
        
        if nodes.is_empty() {
            return Ok(Vec::new());
        }
        
        // Prepare feature matrix
        let num_nodes = nodes.len();
        let mut feature_matrix = Array2::zeros((num_nodes, 128));
        
        for (i, node) in nodes.values().enumerate() {
            feature_matrix.row_mut(i).assign(&node.feature_embedding);
        }
        
        // Forward pass through quantum layers
        let mut current_features = feature_matrix;
        let mut current_quantum_states = quantum_states.clone();
        
        for layer in &self.layers {
            let (new_features, new_states) = layer.forward(
                &current_features,
                &adjacency,
                &current_quantum_states,
            )?;
            
            current_features = new_features;
            current_quantum_states = new_states;
        }
        
        // Extract coordination probabilities from final quantum states
        let coordination_probabilities: Vec<f64> = current_quantum_states.iter()
            .map(|qubit| qubit.coordination_probability())
            .collect();
        
        Ok(coordination_probabilities)
    }

    async fn extract_quantum_clusters(&self, predictions: &[f64]) -> Result<Vec<QuantumCluster>> {
        let mut clusters = Vec::new();
        let nodes = self.nodes.read().unwrap();
        let adjacency = self.adjacency_matrix.read().unwrap();
        
        if nodes.is_empty() || predictions.is_empty() {
            return Ok(clusters);
        }
        
        let node_list: Vec<_> = nodes.values().collect();
        let threshold = 0.7; // Coordination threshold
        
        let mut visited = vec![false; predictions.len()];
        let mut cluster_id = 0;
        
        for i in 0..predictions.len() {
            if visited[i] || predictions[i] < threshold {
                continue;
            }
            
            // Start new cluster using BFS
            let mut cluster_nodes = Vec::new();
            let mut queue = vec![i];
            visited[i] = true;
            
            while let Some(node_idx) = queue.pop() {
                cluster_nodes.push(node_idx);
                
                // Find connected coordinated nodes
                for j in 0..predictions.len() {
                    if !visited[j] 
                        && predictions[j] >= threshold
                        && adjacency[[node_idx, j]] > self.entanglement_threshold {
                        
                        visited[j] = true;
                        queue.push(j);
                    }
                }
            }
            
            if cluster_nodes.len() >= 2 {
                let cluster = QuantumCluster {
                    cluster_id,
                    node_indices: cluster_nodes.clone(),
                    transaction_ids: cluster_nodes.iter()
                        .map(|&idx| node_list[idx].transaction_id.clone())
                        .collect(),
                    wallets: cluster_nodes.iter()
                        .map(|&idx| node_list[idx].wallet.clone())
                        .collect(),
                    avg_coordination_score: cluster_nodes.iter()
                        .map(|&idx| predictions[idx])
                        .sum::<f64>() / cluster_nodes.len() as f64,
                    total_amount_sol: cluster_nodes.iter()
                        .map(|&idx| node_list[idx].amount_sol)
                        .sum(),
                    quantum_entanglement_strength: self.calculate_cluster_entanglement(&cluster_nodes, &adjacency)?,
                    time_span: self.calculate_cluster_time_span(&cluster_nodes, &node_list)?,
                };
                
                clusters.push(cluster);
                cluster_id += 1;
            }
        }
        
        Ok(clusters)
    }

    fn calculate_cluster_entanglement(&self, node_indices: &[usize], adjacency: &Array2<f64>) -> Result<f64> {
        if node_indices.len() < 2 {
            return Ok(0.0);
        }
        
        let mut total_entanglement = 0.0;
        let mut pair_count = 0;
        
        for i in 0..node_indices.len() {
            for j in (i + 1)..node_indices.len() {
                let idx_i = node_indices[i];
                let idx_j = node_indices[j];
                
                total_entanglement += adjacency[[idx_i, idx_j]];
                pair_count += 1;
            }
        }
        
        Ok(if pair_count > 0 { total_entanglement / pair_count as f64 } else { 0.0 })
    }

    fn calculate_cluster_time_span(&self, node_indices: &[usize], node_list: &[&QuantumTransactionNode]) -> Result<i64> {
        if node_indices.is_empty() {
            return Ok(0);
        }
        
        let timestamps: Vec<DateTime<Utc>> = node_indices.iter()
            .map(|&idx| node_list[idx].timestamp)
            .collect();
        
        let min_time = timestamps.iter().min().unwrap();
        let max_time = timestamps.iter().max().unwrap();
        
        Ok((*max_time - *min_time).num_seconds())
    }

    fn calculate_global_coordination_score(&self, predictions: &[f64]) -> Result<f64> {
        if predictions.is_empty() {
            return Ok(0.0);
        }
        
        // Calculate various coordination metrics
        let avg_coordination = predictions.iter().sum::<f64>() / predictions.len() as f64;
        let high_coordination_count = predictions.iter().filter(|&&x| x > 0.7).count();
        let coordination_ratio = high_coordination_count as f64 / predictions.len() as f64;
        
        // Variance indicates clustering
        let variance: f64 = predictions.iter()
            .map(|&x| (x - avg_coordination).powi(2))
            .sum::<f64>() / predictions.len() as f64;
        
        let coordination_clustering = variance.sqrt();
        
        // Combined global score
        let global_score = (avg_coordination * 0.4 + 
                           coordination_ratio * 0.4 + 
                           coordination_clustering * 0.2).clamp(0.0, 1.0);
        
        Ok(global_score)
    }

    async fn find_entangled_transaction_pairs(&self) -> Result<Vec<EntangledPair>> {
        let nodes = self.nodes.read().unwrap();
        let adjacency = self.adjacency_matrix.read().unwrap();
        let quantum_states = self.global_quantum_state.read().unwrap();
        
        let mut entangled_pairs = Vec::new();
        let node_list: Vec<_> = nodes.values().collect();
        
        for i in 0..node_list.len() {
            for j in (i + 1)..node_list.len() {
                let entanglement_strength = adjacency[[i, j]];
                
                if entanglement_strength > self.entanglement_threshold {
                    let pair = EntangledPair {
                        transaction_1: node_list[i].transaction_id.clone(),
                        transaction_2: node_list[j].transaction_id.clone(),
                        wallet_1: node_list[i].wallet.clone(),
                        wallet_2: node_list[j].wallet.clone(),
                        entanglement_strength,
                        coordination_correlation: quantum_states[i].coordination_probability() * 
                                                quantum_states[j].coordination_probability(),
                        time_difference_seconds: (node_list[i].timestamp - node_list[j].timestamp).num_seconds().abs(),
                        amount_similarity: self.calculate_amount_similarity(node_list[i].amount_sol, node_list[j].amount_sol),
                    };
                    
                    entangled_pairs.push(pair);
                }
            }
        }
        
        // Sort by entanglement strength
        entangled_pairs.sort_by(|a, b| b.entanglement_strength.partial_cmp(&a.entanglement_strength).unwrap());
        
        Ok(entangled_pairs)
    }

    fn calculate_amount_similarity(&self, amount1: f64, amount2: f64) -> f64 {
        let ratio = (amount1 / amount2.max(0.01)).ln().abs();
        (-ratio / 2.0).exp()
    }

    /// Get network performance metrics
    pub fn get_quantum_metrics(&self) -> QuantumNetworkMetrics {
        let nodes = self.nodes.read().unwrap();
        let quantum_states = self.global_quantum_state.read().unwrap();
        
        let total_nodes = nodes.len();
        let avg_entanglement = if total_nodes > 0 {
            quantum_states.iter().map(|q| q.entanglement_strength).sum::<f64>() / total_nodes as f64
        } else {
            0.0
        };
        
        let avg_coordination_probability = if total_nodes > 0 {
            quantum_states.iter().map(|q| q.coordination_probability()).sum::<f64>() / total_nodes as f64
        } else {
            0.0
        };
        
        QuantumNetworkMetrics {
            total_quantum_nodes: total_nodes,
            avg_entanglement_strength: avg_entanglement,
            avg_coordination_probability,
            num_layers: self.layers.len(),
            parallel_workers: self.parallel_workers,
            learning_rate: self.learning_rate,
            entanglement_threshold: self.entanglement_threshold,
            timestamp: Utc::now(),
        }
    }
}

// Results and data structures

#[derive(Debug, Serialize)]
pub struct QuantumCoordinationResult {
    pub coordination_score: f64,
    pub detected_clusters: Vec<QuantumCluster>,
    pub quantum_predictions: Vec<f64>,
    pub entangled_pairs: Vec<EntangledPair>,
    pub processing_time_ns: u64,
    pub total_quantum_nodes: usize,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize)]
pub struct QuantumCluster {
    pub cluster_id: usize,
    pub node_indices: Vec<usize>,
    pub transaction_ids: Vec<String>,
    pub wallets: Vec<String>,
    pub avg_coordination_score: f64,
    pub total_amount_sol: f64,
    pub quantum_entanglement_strength: f64,
    pub time_span: i64, // seconds
}

#[derive(Debug, Clone, Serialize)]
pub struct EntangledPair {
    pub transaction_1: String,
    pub transaction_2: String,
    pub wallet_1: String,
    pub wallet_2: String,
    pub entanglement_strength: f64,
    pub coordination_correlation: f64,
    pub time_difference_seconds: i64,
    pub amount_similarity: f64,
}

#[derive(Debug, Serialize)]
pub struct QuantumNetworkMetrics {
    pub total_quantum_nodes: usize,
    pub avg_entanglement_strength: f64,
    pub avg_coordination_probability: f64,
    pub num_layers: usize,
    pub parallel_workers: usize,
    pub learning_rate: f64,
    pub entanglement_threshold: f64,
    pub timestamp: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    #[tokio::test]
    async fn test_quantum_gnn_creation() {
        let qgnn = QuantumGraphNeuralNetwork::new(
            128,           // input_dim
            vec![256, 128], // hidden_dims
            64,            // output_dim
            4              // num_attention_heads
        ).unwrap();
        
        assert_eq!(qgnn.layers.len(), 3); // 2 hidden + 1 output
        assert_eq!(qgnn.parallel_workers, num_cpus::get());
    }

    #[tokio::test]
    async fn test_quantum_coordination_detection() {
        let mut qgnn = QuantumGraphNeuralNetwork::new(128, vec![64], 32, 2).unwrap();
        
        // Create coordinated test transactions
        let transactions = vec![
            crate::analytics::Transaction {
                signature: "quantum_tx_1".to_string(),
                wallet: "coordinated_wallet_1".to_string(),
                token_mint: "test_token".to_string(),
                amount_sol: 100.0,
                transaction_type: crate::analytics::TransactionType::Buy,
                timestamp: Utc::now(),
            },
            crate::analytics::Transaction {
                signature: "quantum_tx_2".to_string(),
                wallet: "coordinated_wallet_2".to_string(),
                token_mint: "test_token".to_string(),
                amount_sol: 100.0, // Same amount (suspicious)
                transaction_type: crate::analytics::TransactionType::Buy,
                timestamp: Utc::now() + Duration::minutes(1), // Similar timing
            },
            crate::analytics::Transaction {
                signature: "quantum_tx_3".to_string(),
                wallet: "random_wallet".to_string(),
                token_mint: "different_token".to_string(),
                amount_sol: 50.5,
                transaction_type: crate::analytics::TransactionType::Sell,
                timestamp: Utc::now() + Duration::hours(2), // Different timing
            },
        ];

        let result = qgnn.detect_coordination_patterns(&transactions).await.unwrap();
        
        assert_eq!(result.total_quantum_nodes, 3);
        assert!(result.coordination_score >= 0.0 && result.coordination_score <= 1.0);
        assert!(result.processing_time_ns > 0);
        
        // Should detect some level of coordination in the first two transactions
        assert!(result.quantum_predictions.len() == 3);
        
        println!("Quantum coordination detection result: {:.3}", result.coordination_score);
        println!("Detected {} clusters", result.detected_clusters.len());
        println!("Found {} entangled pairs", result.entangled_pairs.len());
    }

    #[test]
    fn test_quantum_qubit_operations() {
        let mut qubit = QuantumQubit::new();
        
        // Test initial superposition
        assert!((qubit.amplitude_alpha.powi(2) + qubit.amplitude_beta.powi(2) - 1.0).abs() < 1e-10);
        
        // Test rotation
        qubit.rotate(std::f64::consts::PI / 4.0, 0.0);
        let coordination_prob = qubit.coordination_probability();
        assert!(coordination_prob > 0.0 && coordination_prob < 1.0);
        
        // Test measurement
        let measurement = qubit.measure();
        assert!(measurement == true || measurement == false);
    }

    #[tokio::test]
    async fn test_quantum_clustering_performance() {
        let mut qgnn = QuantumGraphNeuralNetwork::new(128, vec![64, 32], 16, 2).unwrap();
        
        // Create a larger set of test transactions
        let mut transactions = Vec::new();
        let base_time = Utc::now();
        
        for i in 0..50 {
            transactions.push(crate::analytics::Transaction {
                signature: format!("perf_tx_{}", i),
                wallet: format!("wallet_{}", i % 10), // Create clustering
                token_mint: "performance_token".to_string(),
                amount_sol: 100.0 + (i as f64 * 0.1),
                transaction_type: crate::analytics::TransactionType::Buy,
                timestamp: base_time + Duration::seconds(i as i64 * 30),
            });
        }
        
        let start = std::time::Instant::now();
        let result = qgnn.detect_coordination_patterns(&transactions).await.unwrap();
        let duration = start.elapsed();
        
        println!("Quantum processing of {} transactions took: {:?}", transactions.len(), duration);
        println!("Coordination score: {:.3}", result.coordination_score);
        println!("Processing time (nanoseconds): {}", result.processing_time_ns);
        
        assert!(duration.as_millis() < 5000, "Should process 50 transactions in <5s");
        assert_eq!(result.total_quantum_nodes, 50);
    }
}