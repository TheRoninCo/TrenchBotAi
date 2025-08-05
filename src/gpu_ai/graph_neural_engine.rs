//! Real-time Graph Neural Networks for Transaction Analysis
//! State-of-the-art GNN models for blockchain transaction graphs

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use chrono::{DateTime, Utc};
use ndarray::{Array1, Array2, Array3, Axis};
use rayon::prelude::*;

#[cfg(feature = "gpu")]
use tch::{Tensor, Device, Kind, nn};

/// Real-time transaction graph for GNN processing
#[derive(Debug, Clone)]
pub struct TransactionGraph {
    pub nodes: HashMap<String, GraphNode>,
    pub edges: Vec<GraphEdge>,
    pub adjacency_matrix: Array2<f64>,
    pub node_features: Array2<f64>,
    pub edge_features: Array2<f64>,
    pub temporal_edges: VecDeque<TemporalEdge>,
    pub subgraphs: Vec<Subgraph>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: String,
    pub node_type: NodeType,
    pub features: Vec<f64>,
    pub timestamp: DateTime<Utc>,
    pub neighbors: HashSet<String>,
    pub centrality_scores: CentralityScores,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeType {
    Wallet,
    Token,
    Pool,
    DEX,
    Contract,
}

#[derive(Debug, Clone)]
pub struct GraphEdge {
    pub source: String,
    pub target: String,
    pub edge_type: EdgeType,
    pub weight: f64,
    pub features: Vec<f64>,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub enum EdgeType {
    Transfer,
    Swap,
    Liquidity,
    Approval,
    Call,
}

#[derive(Debug, Clone)]
struct TemporalEdge {
    edge: GraphEdge,
    expiry: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CentralityScores {
    pub degree: f64,
    pub betweenness: f64,
    pub closeness: f64,
    pub pagerank: f64,
    pub eigenvector: f64,
}

#[derive(Debug, Clone)]
struct Subgraph {
    nodes: HashSet<String>,
    edges: Vec<GraphEdge>,
    pattern_type: GraphPattern,
    confidence: f64,
}

#[derive(Debug, Clone)]
enum GraphPattern {
    Sandwich,      // Sandwich attack pattern
    Arbitrage,     // Arbitrage opportunity
    Pump,          // Pump pattern
    Wash,          // Wash trading
    FlashLoan,     // Flash loan pattern
    MEVBot,        // MEV bot behavior
}

/// Graph Attention Network for transaction analysis
pub struct GraphAttentionNetwork {
    #[cfg(feature = "gpu")]
    device: Device,
    config: GATConfig,
    attention_layers: Vec<GATLayer>,
    node_embedding: Array2<f64>,
    edge_embedding: Array2<f64>,
}

#[derive(Debug, Clone)]
pub struct GATConfig {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub dropout: f64,
    pub attention_dropout: f64,
    pub use_edge_features: bool,
}

#[derive(Debug, Clone)]
struct GATLayer {
    attention_weights: Array3<f64>, // [heads, input_dim, hidden_dim]
    linear_weights: Array2<f64>,
    bias: Array1<f64>,
    attention_bias: Array1<f64>,
}

/// Graph Convolutional Network for pattern detection
pub struct GraphConvolutionalNetwork {
    #[cfg(feature = "gpu")]
    device: Device,
    layers: Vec<GCNLayer>,
    aggregation_type: AggregationType,
    normalization: GraphNormalization,
}

#[derive(Debug, Clone)]
struct GCNLayer {
    weight_matrix: Array2<f64>,
    bias: Array1<f64>,
    activation: ActivationType,
}

#[derive(Debug, Clone)]
enum AggregationType {
    Mean,
    Sum,
    Max,
    Attention,
}

#[derive(Debug, Clone)]
enum ActivationType {
    ReLU,
    Tanh,
    Sigmoid,
    LeakyReLU,
    GELU,
}

#[derive(Debug, Clone)]
enum GraphNormalization {
    BatchNorm,
    LayerNorm,
    GraphNorm,
    None,
}

/// Temporal Graph Network for time-series analysis
pub struct TemporalGraphNetwork {
    #[cfg(feature = "gpu")]
    device: Device,
    temporal_layers: Vec<TemporalLayer>,
    memory_bank: TemporalMemory,
    time_encoding: TimeEncoding,
}

#[derive(Debug, Clone)]
struct TemporalLayer {
    gru_weights: Array2<f64>,
    attention_weights: Array2<f64>,
    temporal_weights: Array2<f64>,
}

#[derive(Debug, Clone)]
struct TemporalMemory {
    memory_states: HashMap<String, Array1<f64>>,
    memory_capacity: usize,
    decay_rate: f64,
}

#[derive(Debug, Clone)]
enum TimeEncoding {
    Positional,
    Learnable,
    Fourier,
    Relative,
}

impl TransactionGraph {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            adjacency_matrix: Array2::zeros((0, 0)),
            node_features: Array2::zeros((0, 0)),
            edge_features: Array2::zeros((0, 0)),
            temporal_edges: VecDeque::new(),
            subgraphs: Vec::new(),
        }
    }

    /// Add transaction to the graph in real-time
    pub async fn add_transaction(&mut self, tx: &crate::analytics::Transaction) -> Result<()> {
        // Create nodes for wallet and token if they don't exist
        let wallet_node = GraphNode {
            id: tx.wallet.clone(),
            node_type: NodeType::Wallet,
            features: self.extract_wallet_features(&tx.wallet).await?,
            timestamp: tx.timestamp,
            neighbors: HashSet::new(),
            centrality_scores: CentralityScores::default(),
        };

        let token_node = GraphNode {
            id: tx.token_mint.clone(),
            node_type: NodeType::Token,
            features: self.extract_token_features(&tx.token_mint).await?,
            timestamp: tx.timestamp,
            neighbors: HashSet::new(),
            centrality_scores: CentralityScores::default(),
        };

        // Insert nodes
        self.nodes.entry(tx.wallet.clone()).or_insert(wallet_node);
        self.nodes.entry(tx.token_mint.clone()).or_insert(token_node);

        // Create edge
        let edge = GraphEdge {
            source: tx.wallet.clone(),
            target: tx.token_mint.clone(),
            edge_type: match tx.transaction_type {
                crate::analytics::TransactionType::Buy => EdgeType::Transfer,
                crate::analytics::TransactionType::Sell => EdgeType::Transfer,
                crate::analytics::TransactionType::Swap => EdgeType::Swap,
            },
            weight: tx.amount_sol,
            features: vec![tx.amount_sol, self.calculate_edge_features(tx).await?],
            timestamp: tx.timestamp,
        };

        // Add temporal edge with expiry
        let temporal_edge = TemporalEdge {
            edge: edge.clone(),
            expiry: tx.timestamp + chrono::Duration::hours(24), // Edges expire after 24h
        };

        self.edges.push(edge);
        self.temporal_edges.push_back(temporal_edge);

        // Update neighbor relationships
        if let Some(wallet_node) = self.nodes.get_mut(&tx.wallet) {
            wallet_node.neighbors.insert(tx.token_mint.clone());
        }
        if let Some(token_node) = self.nodes.get_mut(&tx.token_mint) {
            token_node.neighbors.insert(tx.wallet.clone());
        }

        // Cleanup expired edges
        self.cleanup_expired_edges().await?;

        // Update graph matrices
        self.update_adjacency_matrix().await?;
        self.update_feature_matrices().await?;

        // Detect patterns in real-time
        self.detect_graph_patterns().await?;

        Ok(())
    }

    /// Real-time graph pattern detection
    pub async fn detect_graph_patterns(&mut self) -> Result<Vec<GraphPattern>> {
        let mut detected_patterns = Vec::new();

        // Detect sandwich attacks
        if let Some(sandwich_patterns) = self.detect_sandwich_patterns().await? {
            detected_patterns.extend(sandwich_patterns);
        }

        // Detect arbitrage patterns
        if let Some(arbitrage_patterns) = self.detect_arbitrage_patterns().await? {
            detected_patterns.extend(arbitrage_patterns);
        }

        // Detect MEV bot patterns
        if let Some(mev_patterns) = self.detect_mev_bot_patterns().await? {
            detected_patterns.extend(mev_patterns);
        }

        // Detect wash trading
        if let Some(wash_patterns) = self.detect_wash_trading_patterns().await? {
            detected_patterns.extend(wash_patterns);
        }

        Ok(detected_patterns)
    }

    async fn detect_sandwich_patterns(&self) -> Result<Option<Vec<GraphPattern>>> {
        let mut patterns = Vec::new();

        // Look for sandwich attack pattern: A -> B -> A within short timeframe
        for wallet in self.nodes.keys() {
            let wallet_edges: Vec<_> = self.edges.iter()
                .filter(|e| e.source == *wallet || e.target == *wallet)
                .collect();

            // Sort by timestamp
            let mut sorted_edges = wallet_edges.clone();
            sorted_edges.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));

            // Look for sandwich pattern in recent edges
            for window in sorted_edges.windows(3) {
                if self.is_sandwich_pattern(window) {
                    patterns.push(GraphPattern::Sandwich);
                }
            }
        }

        if patterns.is_empty() {
            Ok(None)
        } else {
            Ok(Some(patterns))
        }
    }

    async fn detect_arbitrage_patterns(&self) -> Result<Option<Vec<GraphPattern>>> {
        let mut patterns = Vec::new();

        // Detect triangular arbitrage patterns
        for token_a in self.nodes.keys().filter(|id| {
            self.nodes.get(*id).map(|n| matches!(n.node_type, NodeType::Token)).unwrap_or(false)
        }) {
            for token_b in self.nodes.keys().filter(|id| *id != token_a && {
                self.nodes.get(*id).map(|n| matches!(n.node_type, NodeType::Token)).unwrap_or(false)
            }) {
                for token_c in self.nodes.keys().filter(|id| *id != token_a && *id != token_b && {
                    self.nodes.get(*id).map(|n| matches!(n.node_type, NodeType::Token)).unwrap_or(false)
                }) {
                    if self.has_arbitrage_cycle(token_a, token_b, token_c) {
                        patterns.push(GraphPattern::Arbitrage);
                    }
                }
            }
        }

        if patterns.is_empty() {
            Ok(None)
        } else {
            Ok(Some(patterns))
        }
    }

    async fn detect_mev_bot_patterns(&self) -> Result<Option<Vec<GraphPattern>>> {
        let mut patterns = Vec::new();

        // Detect high-frequency trading patterns characteristic of MEV bots
        for (wallet_id, node) in &self.nodes {
            if matches!(node.node_type, NodeType::Wallet) {
                let wallet_transactions = self.get_wallet_recent_transactions(wallet_id, chrono::Duration::minutes(10));
                
                // MEV bot indicators
                let high_frequency = wallet_transactions.len() > 10;
                let high_success_rate = self.calculate_success_rate(&wallet_transactions) > 0.8;
                let consistent_timing = self.has_consistent_timing_patterns(&wallet_transactions);

                if high_frequency && high_success_rate && consistent_timing {
                    patterns.push(GraphPattern::MEVBot);
                }
            }
        }

        if patterns.is_empty() {
            Ok(None)
        } else {
            Ok(Some(patterns))
        }
    }

    async fn detect_wash_trading_patterns(&self) -> Result<Option<Vec<GraphPattern>>> {
        let mut patterns = Vec::new();

        // Detect circular trading patterns (wash trading)
        for (wallet_id, _) in &self.nodes {
            if self.has_circular_trading_pattern(wallet_id) {
                patterns.push(GraphPattern::Wash);
            }
        }

        if patterns.is_empty() {
            Ok(None)
        } else {
            Ok(Some(patterns))
        }
    }

    // Helper methods
    async fn extract_wallet_features(&self, wallet: &str) -> Result<Vec<f64>> {
        // Extract wallet-specific features
        let transaction_count = self.edges.iter()
            .filter(|e| e.source == wallet || e.target == wallet)
            .count() as f64;

        let total_volume = self.edges.iter()
            .filter(|e| e.source == wallet || e.target == wallet)
            .map(|e| e.weight)
            .sum::<f64>();

        let unique_tokens = self.edges.iter()
            .filter(|e| e.source == wallet)
            .map(|e| &e.target)
            .collect::<HashSet<_>>()
            .len() as f64;

        Ok(vec![transaction_count, total_volume, unique_tokens])
    }

    async fn extract_token_features(&self, token: &str) -> Result<Vec<f64>> {
        // Extract token-specific features
        let holder_count = self.edges.iter()
            .filter(|e| e.target == token)
            .map(|e| &e.source)
            .collect::<HashSet<_>>()
            .len() as f64;

        let total_volume = self.edges.iter()
            .filter(|e| e.target == token || e.source == token)
            .map(|e| e.weight)
            .sum::<f64>();

        let recent_activity = self.edges.iter()
            .filter(|e| (e.target == token || e.source == token) && 
                e.timestamp > Utc::now() - chrono::Duration::hours(1))
            .count() as f64;

        Ok(vec![holder_count, total_volume, recent_activity])
    }

    async fn calculate_edge_features(&self, tx: &crate::analytics::Transaction) -> Result<f64> {
        // Calculate edge-specific features
        let time_since_last = self.edges.iter()
            .filter(|e| e.source == tx.wallet)
            .map(|e| (tx.timestamp - e.timestamp).num_seconds() as f64)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);

        Ok(time_since_last)
    }

    async fn cleanup_expired_edges(&mut self) -> Result<()> {
        let now = Utc::now();
        self.temporal_edges.retain(|te| te.expiry > now);
        
        // Remove expired edges from main edge list
        let valid_edge_ids: HashSet<_> = self.temporal_edges.iter()
            .map(|te| (&te.edge.source, &te.edge.target, te.edge.timestamp))
            .collect();

        self.edges.retain(|e| valid_edge_ids.contains(&(&e.source, &e.target, e.timestamp)));

        Ok(())
    }

    async fn update_adjacency_matrix(&mut self) -> Result<()> {
        let node_ids: Vec<_> = self.nodes.keys().cloned().collect();
        let n = node_ids.len();
        let mut adj_matrix = Array2::zeros((n, n));

        for edge in &self.edges {
            if let (Some(i), Some(j)) = (
                node_ids.iter().position(|id| id == &edge.source),
                node_ids.iter().position(|id| id == &edge.target)
            ) {
                adj_matrix[[i, j]] = edge.weight;
            }
        }

        self.adjacency_matrix = adj_matrix;
        Ok(())
    }

    async fn update_feature_matrices(&mut self) -> Result<()> {
        let node_ids: Vec<_> = self.nodes.keys().cloned().collect();
        let n = node_ids.len();
        
        if n == 0 {
            return Ok(());
        }

        let feature_dim = self.nodes.values().next().unwrap().features.len();
        let mut node_features = Array2::zeros((n, feature_dim));

        for (i, node_id) in node_ids.iter().enumerate() {
            if let Some(node) = self.nodes.get(node_id) {
                for (j, &feature) in node.features.iter().enumerate() {
                    node_features[[i, j]] = feature;
                }
            }
        }

        self.node_features = node_features;

        // Update edge features
        if !self.edges.is_empty() {
            let edge_feature_dim = self.edges[0].features.len();
            let mut edge_features = Array2::zeros((self.edges.len(), edge_feature_dim));

            for (i, edge) in self.edges.iter().enumerate() {
                for (j, &feature) in edge.features.iter().enumerate() {
                    edge_features[[i, j]] = feature;
                }
            }

            self.edge_features = edge_features;
        }

        Ok(())
    }

    fn is_sandwich_pattern(&self, window: &[&GraphEdge]) -> bool {
        if window.len() != 3 {
            return false;
        }

        // Check if it's a sandwich: A->B, B->C, C->A pattern within short timeframe
        let time_diff = (window[2].timestamp - window[0].timestamp).num_seconds();
        let is_quick = time_diff < 60; // Within 1 minute

        let forms_cycle = window[0].target == window[1].source &&
                         window[1].target == window[2].source &&
                         window[2].target == window[0].source;

        is_quick && forms_cycle
    }

    fn has_arbitrage_cycle(&self, token_a: &str, token_b: &str, token_c: &str) -> bool {
        // Check if there's a profitable cycle A->B->C->A
        let path_ab = self.find_direct_path(token_a, token_b);
        let path_bc = self.find_direct_path(token_b, token_c);
        let path_ca = self.find_direct_path(token_c, token_a);

        path_ab.is_some() && path_bc.is_some() && path_ca.is_some()
    }

    fn find_direct_path(&self, from: &str, to: &str) -> Option<f64> {
        // Find direct trading path between two tokens
        self.edges.iter()
            .find(|e| e.source == from && e.target == to)
            .map(|e| e.weight)
    }

    fn get_wallet_recent_transactions(&self, wallet: &str, duration: chrono::Duration) -> Vec<&GraphEdge> {
        let cutoff = Utc::now() - duration;
        self.edges.iter()
            .filter(|e| (e.source == wallet || e.target == wallet) && e.timestamp > cutoff)
            .collect()
    }

    fn calculate_success_rate(&self, transactions: &[&GraphEdge]) -> f64 {
        if transactions.is_empty() {
            return 0.0;
        }

        // Simplified success rate calculation
        // In practice, would need profit/loss data
        let profitable_txs = transactions.len() as f64 * 0.7; // Mock 70% success rate
        profitable_txs / transactions.len() as f64
    }

    fn has_consistent_timing_patterns(&self, transactions: &[&GraphEdge]) -> bool {
        if transactions.len() < 3 {
            return false;
        }

        // Check if transactions happen at consistent intervals (bot-like behavior)
        let mut intervals = Vec::new();
        for window in transactions.windows(2) {
            let interval = (window[1].timestamp - window[0].timestamp).num_seconds();
            intervals.push(interval);
        }

        // Calculate coefficient of variation
        let mean = intervals.iter().sum::<i64>() as f64 / intervals.len() as f64;
        let variance = intervals.iter()
            .map(|&x| (x as f64 - mean).powi(2))
            .sum::<f64>() / intervals.len() as f64;
        let cv = variance.sqrt() / mean;

        cv < 0.3 // Low variation indicates consistent timing
    }

    fn has_circular_trading_pattern(&self, wallet: &str) -> bool {
        // Detect if wallet engages in circular trading (A->B->A patterns)
        let wallet_edges: Vec<_> = self.edges.iter()
            .filter(|e| e.source == wallet)
            .collect();

        for edge1 in &wallet_edges {
            for edge2 in &wallet_edges {
                if edge1.target == edge2.source && edge2.target == edge1.source {
                    // Found A->B and B->A pattern
                    let time_diff = (edge2.timestamp - edge1.timestamp).num_seconds().abs();
                    if time_diff < 3600 { // Within 1 hour
                        return true;
                    }
                }
            }
        }

        false
    }
}

impl Default for CentralityScores {
    fn default() -> Self {
        Self {
            degree: 0.0,
            betweenness: 0.0,
            closeness: 0.0,
            pagerank: 0.0,
            eigenvector: 0.0,
        }
    }
}

impl GraphAttentionNetwork {
    pub fn new(config: GATConfig) -> Result<Self> {
        #[cfg(feature = "gpu")]
        let device = Device::cuda_if_available();

        let attention_layers = Self::initialize_gat_layers(&config);
        let node_embedding = Array2::random((1000, config.input_dim), ndarray::random::StandardNormal);
        let edge_embedding = Array2::random((5000, config.input_dim), ndarray::random::StandardNormal);

        Ok(Self {
            #[cfg(feature = "gpu")]
            device,
            config,
            attention_layers,
            node_embedding,
            edge_embedding,
        })
    }

    /// Forward pass through Graph Attention Network
    pub async fn forward(&self, graph: &TransactionGraph) -> Result<Array2<f64>> {
        let mut node_features = graph.node_features.clone();

        // Apply GAT layers sequentially
        for (layer_idx, layer) in self.attention_layers.iter().enumerate() {
            node_features = self.gat_layer_forward(&node_features, &graph.adjacency_matrix, layer).await?;
            
            // Apply dropout and activation
            if layer_idx < self.attention_layers.len() - 1 {
                node_features = node_features.mapv(|x| x.max(0.0)); // ReLU
            }
        }

        Ok(node_features)
    }

    async fn gat_layer_forward(&self, node_features: &Array2<f64>, adj_matrix: &Array2<f64>, layer: &GATLayer) -> Result<Array2<f64>> {
        let n_nodes = node_features.nrows();
        let output_dim = layer.linear_weights.ncols();
        let mut output = Array2::zeros((n_nodes, output_dim));

        // Multi-head attention
        for head in 0..self.config.num_heads {
            let head_output = self.single_head_attention(node_features, adj_matrix, layer, head).await?;
            
            // Concatenate or average heads
            if head == 0 {
                output = head_output;
            } else {
                output = output + head_output;
            }
        }

        // Average across heads
        output = output / self.config.num_heads as f64;

        Ok(output)
    }

    async fn single_head_attention(&self, node_features: &Array2<f64>, adj_matrix: &Array2<f64>, layer: &GATLayer, head: usize) -> Result<Array2<f64>> {
        let n_nodes = node_features.nrows();
        
        // Linear transformation
        let transformed = node_features.dot(&layer.linear_weights);
        
        // Compute attention scores
        let mut attention_scores = Array2::zeros((n_nodes, n_nodes));
        
        for i in 0..n_nodes {
            for j in 0..n_nodes {
                if adj_matrix[[i, j]] != 0.0 {
                    // Attention mechanism: e_ij = a^T [W*h_i || W*h_j]
                    let concat_features = self.concatenate_features(&transformed.row(i), &transformed.row(j));
                    let attention_weights = &layer.attention_weights.index_axis(Axis(0), head).t();
                    let score = concat_features.dot(attention_weights);
                    attention_scores[[i, j]] = score;
                }
            }
        }

        // Apply softmax to attention scores
        let attention_weights = self.softmax(&attention_scores);

        // Aggregate neighbor features
        let output = attention_weights.dot(&transformed);

        Ok(output)
    }

    fn concatenate_features(&self, feat1: &ndarray::ArrayView1<f64>, feat2: &ndarray::ArrayView1<f64>) -> Array1<f64> {
        let mut result = Array1::zeros(feat1.len() + feat2.len());
        result.slice_mut(s![..feat1.len()]).assign(feat1);
        result.slice_mut(s![feat1.len()..]).assign(feat2);
        result
    }

    fn softmax(&self, input: &Array2<f64>) -> Array2<f64> {
        let mut output = Array2::zeros(input.raw_dim());
        
        for i in 0..input.nrows() {
            let row = input.row(i);
            let max_val = row.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            
            let exp_row: Vec<f64> = row.iter().map(|&x| (x - max_val).exp()).collect();
            let sum: f64 = exp_row.iter().sum();
            
            for (j, &exp_val) in exp_row.iter().enumerate() {
                output[[i, j]] = if sum > 0.0 { exp_val / sum } else { 0.0 };
            }
        }

        output
    }

    fn initialize_gat_layers(config: &GATConfig) -> Vec<GATLayer> {
        (0..config.num_layers)
            .map(|_| GATLayer {
                attention_weights: Array3::random(
                    (config.num_heads, config.hidden_dim * 2, 1), 
                    ndarray::random::StandardNormal
                ),
                linear_weights: Array2::random(
                    (config.input_dim, config.hidden_dim), 
                    ndarray::random::StandardNormal
                ),
                bias: Array1::random(config.hidden_dim, ndarray::random::StandardNormal),
                attention_bias: Array1::random(1, ndarray::random::StandardNormal),
            })
            .collect()
    }
}

impl Default for GATConfig {
    fn default() -> Self {
        Self {
            input_dim: 64,
            hidden_dim: 128,
            output_dim: 32,
            num_heads: 4,
            num_layers: 3,
            dropout: 0.1,
            attention_dropout: 0.1,
            use_edge_features: true,
        }
    }
}