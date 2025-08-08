//! Circuit Breakers and Emergency Shutdown System
//! 
//! Critical safety mechanisms to prevent catastrophic losses during trading operations.
//! Implements multiple layers of protection with automatic and manual overrides.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc, Duration};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, atomic::{AtomicBool, AtomicU64, Ordering}};
use tokio::sync::{RwLock, broadcast, mpsc};
use tracing::{warn, error, info};

/// Circuit breaker states following standard patterns
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CircuitState {
    Closed,    // Normal operation
    Open,      // Breaker tripped - blocking operations
    HalfOpen,  // Testing if system has recovered
}

/// Types of circuit breakers for different failure scenarios
#[derive(Debug, Clone, Serialize, Deserialize, Hash, PartialEq, Eq)]
pub enum CircuitType {
    TotalLoss,           // Maximum loss threshold exceeded
    ConsecutiveLosses,   // Too many losses in a row
    RapidLoss,          // Loss velocity too high
    NetworkFailure,     // RPC/network issues
    InvalidData,        // Data quality problems
    PositionSize,       // Position too large
    MarketVolatility,   // Extreme market conditions
    ManualOverride,     // Human-triggered emergency stop
}

/// Circuit breaker configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitConfig {
    // Loss-based triggers
    pub max_daily_loss_sol: f64,           // Maximum loss per day (SOL)
    pub max_total_loss_pct: f64,           // Maximum loss as % of capital
    pub consecutive_loss_limit: u32,        // Max consecutive losses
    pub rapid_loss_threshold_sol: f64,      // Loss amount triggering rapid loss check
    pub rapid_loss_window_minutes: i64,     // Time window for rapid loss check
    
    // Position and risk limits
    pub max_position_size_pct: f64,        // Max position as % of capital
    pub max_concurrent_positions: usize,    // Max simultaneous positions
    pub correlation_limit: f64,             // Max correlation between positions
    
    // Network and data quality
    pub max_rpc_failures: u32,             // RPC failure threshold
    pub max_data_staleness_seconds: i64,   // Max acceptable data age
    pub min_liquidity_sol: f64,            // Minimum liquidity required
    
    // Recovery settings
    pub recovery_test_interval_minutes: i64, // How often to test recovery
    pub recovery_test_position_size_pct: f64, // Position size for recovery test
    pub auto_recovery_enabled: bool,         // Allow automatic recovery
}

impl Default for CircuitConfig {
    fn default() -> Self {
        Self {
            // Conservative loss limits
            max_daily_loss_sol: 100.0,           // 100 SOL max daily loss
            max_total_loss_pct: 0.20,            // 20% max total loss
            consecutive_loss_limit: 5,            // 5 consecutive losses max
            rapid_loss_threshold_sol: 50.0,      // 50 SOL in short time
            rapid_loss_window_minutes: 15,       // 15 minute window
            
            // Position limits
            max_position_size_pct: 0.10,         // 10% max position size
            max_concurrent_positions: 3,          // Max 3 positions
            correlation_limit: 0.7,              // 70% max correlation
            
            // Network requirements
            max_rpc_failures: 3,                 // 3 RPC failures max
            max_data_staleness_seconds: 30,      // 30 second data freshness
            min_liquidity_sol: 100.0,            // 100 SOL minimum liquidity
            
            // Recovery settings
            recovery_test_interval_minutes: 30,   // Test every 30 minutes
            recovery_test_position_size_pct: 0.01, // 1% test position
            auto_recovery_enabled: true,          // Enable auto-recovery
        }
    }
}

/// Individual circuit breaker
#[derive(Debug)]
pub struct CircuitBreaker {
    pub circuit_type: CircuitType,
    pub state: Arc<AtomicU64>, // Using AtomicU64 to store CircuitState as u64
    pub config: CircuitConfig,
    pub failure_count: Arc<AtomicU64>,
    pub last_failure: Arc<RwLock<Option<DateTime<Utc>>>>,
    pub last_recovery_test: Arc<RwLock<Option<DateTime<Utc>>>>,
}

impl CircuitBreaker {
    pub fn new(circuit_type: CircuitType, config: CircuitConfig) -> Self {
        Self {
            circuit_type,
            state: Arc::new(AtomicU64::new(0)), // 0 = Closed
            config,
            failure_count: Arc::new(AtomicU64::new(0)),
            last_failure: Arc::new(RwLock::new(None)),
            last_recovery_test: Arc::new(RwLock::new(None)),
        }
    }

    pub fn get_state(&self) -> CircuitState {
        match self.state.load(Ordering::Acquire) {
            0 => CircuitState::Closed,
            1 => CircuitState::Open,
            2 => CircuitState::HalfOpen,
            _ => CircuitState::Open, // Default to safe state
        }
    }

    pub fn set_state(&self, new_state: CircuitState) {
        let state_value = match new_state {
            CircuitState::Closed => 0,
            CircuitState::Open => 1,
            CircuitState::HalfOpen => 2,
        };
        self.state.store(state_value, Ordering::Release);
    }

    /// Check if operation should be allowed
    pub async fn can_proceed(&self) -> bool {
        match self.get_state() {
            CircuitState::Closed => true,
            CircuitState::Open => {
                // Check if enough time has passed for recovery test
                if self.config.auto_recovery_enabled {
                    let last_test = self.last_recovery_test.read().await;
                    if let Some(last_test_time) = *last_test {
                        let time_since_test = Utc::now() - last_test_time;
                        if time_since_test > Duration::minutes(self.config.recovery_test_interval_minutes) {
                            drop(last_test);
                            self.set_state(CircuitState::HalfOpen);
                            return true; // Allow one test operation
                        }
                    } else {
                        // No previous test - allow immediate test
                        self.set_state(CircuitState::HalfOpen);
                        return true;
                    }
                }
                false
            },
            CircuitState::HalfOpen => true, // Allow one test operation
        }
    }

    /// Record a successful operation
    pub async fn record_success(&self) {
        match self.get_state() {
            CircuitState::HalfOpen => {
                // Recovery test succeeded - return to normal
                self.set_state(CircuitState::Closed);
                self.failure_count.store(0, Ordering::Release);
                info!("Circuit breaker {:?} recovered - returning to normal operation", self.circuit_type);
            },
            CircuitState::Closed => {
                // Reset failure count on success during normal operation
                self.failure_count.store(0, Ordering::Release);
            },
            _ => {} // No action needed for Open state
        }
    }

    /// Record a failure and potentially trip the breaker
    pub async fn record_failure(&self, reason: String) -> bool {
        let failure_count = self.failure_count.fetch_add(1, Ordering::AcqRel) + 1;
        
        // Update last failure time
        {
            let mut last_failure = self.last_failure.write().await;
            *last_failure = Some(Utc::now());
        }

        // Check if we should trip the breaker
        let should_trip = match self.circuit_type {
            CircuitType::ConsecutiveLosses => failure_count >= self.config.consecutive_loss_limit as u64,
            CircuitType::NetworkFailure => failure_count >= self.config.max_rpc_failures as u64,
            _ => failure_count >= 3, // Default threshold
        };

        if should_trip {
            self.set_state(CircuitState::Open);
            error!("Circuit breaker {:?} TRIPPED: {} (failure count: {})", 
                   self.circuit_type, reason, failure_count);
            return true;
        }

        warn!("Circuit breaker {:?} failure recorded: {} (count: {})", 
              self.circuit_type, reason, failure_count);
        false
    }

    /// Manually trip the circuit breaker
    pub async fn trip_manually(&self, reason: String) {
        self.set_state(CircuitState::Open);
        error!("Circuit breaker {:?} MANUALLY TRIPPED: {}", self.circuit_type, reason);
    }

    /// Manually reset the circuit breaker
    pub async fn reset_manually(&self, reason: String) {
        self.set_state(CircuitState::Closed);
        self.failure_count.store(0, Ordering::Release);
        info!("Circuit breaker {:?} MANUALLY RESET: {}", self.circuit_type, reason);
    }
}

/// Emergency shutdown events
#[derive(Debug, Clone, Serialize)]
pub struct EmergencyEvent {
    pub event_type: EmergencyEventType,
    pub timestamp: DateTime<Utc>,
    pub reason: String,
    pub triggered_by: String,
    pub affected_systems: Vec<String>,
    pub severity: EmergencySeverity,
}

#[derive(Debug, Clone, Serialize)]
pub enum EmergencyEventType {
    CircuitBreakerTripped,
    ManualShutdown,
    CatastrophicLoss,
    SystemFailure,
    MarketCrash,
    RecoveryAttempt,
}

#[derive(Debug, Clone, Serialize, PartialEq, PartialOrd)]
pub enum EmergencySeverity {
    Low,        // Minor issue, operations can continue
    Medium,     // Significant issue, some operations blocked  
    High,       // Major issue, most operations blocked
    Critical,   // System-wide shutdown required
}

/// Main emergency management system
pub struct EmergencyManager {
    pub circuit_breakers: HashMap<CircuitType, Arc<CircuitBreaker>>,
    pub config: CircuitConfig,
    pub global_shutdown: Arc<AtomicBool>,
    pub emergency_events: Arc<RwLock<VecDeque<EmergencyEvent>>>,
    pub event_broadcaster: broadcast::Sender<EmergencyEvent>,
    pub manual_override: Arc<AtomicBool>,
    
    // Performance tracking for circuit breaker logic
    pub daily_pnl: Arc<RwLock<f64>>,
    pub consecutive_losses: Arc<AtomicU64>,
    pub recent_losses: Arc<RwLock<VecDeque<(DateTime<Utc>, f64)>>>,
}

impl EmergencyManager {
    pub fn new(config: CircuitConfig) -> (Self, broadcast::Receiver<EmergencyEvent>) {
        let (event_sender, event_receiver) = broadcast::channel(100);
        
        // Create circuit breakers for each type
        let mut circuit_breakers = HashMap::new();
        for circuit_type in [
            CircuitType::TotalLoss,
            CircuitType::ConsecutiveLosses, 
            CircuitType::RapidLoss,
            CircuitType::NetworkFailure,
            CircuitType::InvalidData,
            CircuitType::PositionSize,
            CircuitType::MarketVolatility,
            CircuitType::ManualOverride,
        ] {
            circuit_breakers.insert(
                circuit_type.clone(), 
                Arc::new(CircuitBreaker::new(circuit_type, config.clone()))
            );
        }

        let manager = Self {
            circuit_breakers,
            config,
            global_shutdown: Arc::new(AtomicBool::new(false)),
            emergency_events: Arc::new(RwLock::new(VecDeque::with_capacity(1000))),
            event_broadcaster: event_sender,
            manual_override: Arc::new(AtomicBool::new(false)),
            daily_pnl: Arc::new(RwLock::new(0.0)),
            consecutive_losses: Arc::new(AtomicU64::new(0)),
            recent_losses: Arc::new(RwLock::new(VecDeque::new())),
        };

        (manager, event_receiver)
    }

    /// Check if any operation can proceed (master check)
    pub async fn can_proceed(&self) -> Result<bool> {
        // Global shutdown check
        if self.global_shutdown.load(Ordering::Acquire) {
            return Ok(false);
        }

        // Manual override check
        if self.manual_override.load(Ordering::Acquire) {
            return Ok(false);
        }

        // Check all critical circuit breakers
        for (circuit_type, breaker) in &self.circuit_breakers {
            match circuit_type {
                CircuitType::TotalLoss | 
                CircuitType::ConsecutiveLosses | 
                CircuitType::ManualOverride => {
                    if !breaker.can_proceed().await {
                        warn!("Operation blocked by circuit breaker: {:?}", circuit_type);
                        return Ok(false);
                    }
                },
                _ => {} // Non-critical breakers don't block operations
            }
        }

        Ok(true)
    }

    /// Record a trade result and check for circuit breaker conditions
    pub async fn record_trade_result(&self, profit_loss: f64, operation_id: &str) -> Result<()> {
        // Update daily P&L
        {
            let mut daily_pnl = self.daily_pnl.write().await;
            *daily_pnl += profit_loss;
        }

        if profit_loss < 0.0 {
            // Record loss
            let loss_amount = -profit_loss;
            
            // Update consecutive losses
            let consecutive_losses = self.consecutive_losses.fetch_add(1, Ordering::AcqRel) + 1;
            
            // Add to recent losses for rapid loss detection
            {
                let mut recent_losses = self.recent_losses.write().await;
                recent_losses.push_back((Utc::now(), loss_amount));
                
                // Clean old entries (keep last hour)
                let cutoff = Utc::now() - Duration::hours(1);
                while let Some((timestamp, _)) = recent_losses.front() {
                    if *timestamp < cutoff {
                        recent_losses.pop_front();
                    } else {
                        break;
                    }
                }
            }

            // Check circuit breaker conditions
            self.check_loss_conditions(loss_amount, consecutive_losses, operation_id).await?;
            
        } else {
            // Reset consecutive losses on profit
            self.consecutive_losses.store(0, Ordering::Release);
            
            // Record success in circuit breakers
            for breaker in self.circuit_breakers.values() {
                breaker.record_success().await;
            }
        }

        Ok(())
    }

    async fn check_loss_conditions(&self, loss_amount: f64, consecutive_losses: u64, operation_id: &str) -> Result<()> {
        // Check total daily loss
        let daily_pnl = self.daily_pnl.read().await;
        if -*daily_pnl >= self.config.max_daily_loss_sol {
            self.trigger_emergency(
                EmergencyEventType::CatastrophicLoss,
                format!("Daily loss limit exceeded: {:.2} SOL", -*daily_pnl),
                operation_id.to_string(),
                EmergencySeverity::Critical,
            ).await?;
            
            if let Some(breaker) = self.circuit_breakers.get(&CircuitType::TotalLoss) {
                breaker.record_failure(format!("Daily loss limit exceeded: {:.2} SOL", -*daily_pnl)).await;
            }
        }

        // Check consecutive losses
        if consecutive_losses >= self.config.consecutive_loss_limit as u64 {
            if let Some(breaker) = self.circuit_breakers.get(&CircuitType::ConsecutiveLosses) {
                breaker.record_failure(format!("Consecutive loss limit reached: {}", consecutive_losses)).await;
            }
        }

        // Check rapid loss
        let recent_losses = self.recent_losses.read().await;
        let rapid_window = Utc::now() - Duration::minutes(self.config.rapid_loss_window_minutes);
        let rapid_loss: f64 = recent_losses.iter()
            .filter(|(timestamp, _)| *timestamp > rapid_window)
            .map(|(_, amount)| *amount)
            .sum();
            
        if rapid_loss >= self.config.rapid_loss_threshold_sol {
            if let Some(breaker) = self.circuit_breakers.get(&CircuitType::RapidLoss) {
                breaker.record_failure(format!("Rapid loss detected: {:.2} SOL in {} minutes", 
                                               rapid_loss, self.config.rapid_loss_window_minutes)).await;
            }
        }

        Ok(())
    }

    /// Trigger emergency event
    pub async fn trigger_emergency(
        &self,
        event_type: EmergencyEventType,
        reason: String,
        triggered_by: String,
        severity: EmergencySeverity,
    ) -> Result<()> {
        let event = EmergencyEvent {
            event_type: event_type.clone(),
            timestamp: Utc::now(),
            reason: reason.clone(),
            triggered_by,
            affected_systems: vec!["trading".to_string(), "risk_management".to_string()],
            severity: severity.clone(),
        };

        // Store event
        {
            let mut events = self.emergency_events.write().await;
            events.push_back(event.clone());
            if events.len() > 1000 {
                events.pop_front();
            }
        }

        // Broadcast event
        if let Err(e) = self.event_broadcaster.send(event.clone()) {
            error!("Failed to broadcast emergency event: {}", e);
        }

        // Handle based on severity
        match severity {
            EmergencySeverity::Critical => {
                self.emergency_shutdown(reason).await?;
            },
            EmergencySeverity::High => {
                warn!("HIGH SEVERITY EMERGENCY: {} - Implementing restrictions", reason);
                // Could implement partial restrictions here
            },
            _ => {
                info!("Emergency event recorded: {}", reason);
            }
        }

        Ok(())
    }

    /// Emergency shutdown - stops all operations immediately
    pub async fn emergency_shutdown(&self, reason: String) -> Result<()> {
        error!("🚨 EMERGENCY SHUTDOWN ACTIVATED: {}", reason);
        
        // Set global shutdown flag
        self.global_shutdown.store(true, Ordering::Release);
        
        // Trip all circuit breakers
        for (circuit_type, breaker) in &self.circuit_breakers {
            breaker.trip_manually(format!("Emergency shutdown: {}", reason)).await;
        }

        // Trigger emergency event
        self.trigger_emergency(
            EmergencyEventType::ManualShutdown,
            format!("Emergency shutdown: {}", reason),
            "emergency_manager".to_string(),
            EmergencySeverity::Critical,
        ).await?;

        Ok(())
    }

    /// Manual override to stop all operations
    pub async fn manual_override(&self, reason: String, enabled: bool) -> Result<()> {
        self.manual_override.store(enabled, Ordering::Release);
        
        if enabled {
            error!("🛑 MANUAL OVERRIDE ACTIVATED: {}", reason);
            if let Some(breaker) = self.circuit_breakers.get(&CircuitType::ManualOverride) {
                breaker.trip_manually(reason.clone()).await;
            }
        } else {
            info!("✅ MANUAL OVERRIDE DEACTIVATED: {}", reason);
            if let Some(breaker) = self.circuit_breakers.get(&CircuitType::ManualOverride) {
                breaker.reset_manually(reason.clone()).await;
            }
        }

        Ok(())
    }

    /// Get system status summary
    pub async fn get_system_status(&self) -> SystemStatus {
        let mut breaker_states = HashMap::new();
        for (circuit_type, breaker) in &self.circuit_breakers {
            breaker_states.insert(circuit_type.clone(), breaker.get_state());
        }

        let daily_pnl = *self.daily_pnl.read().await;
        let consecutive_losses = self.consecutive_losses.load(Ordering::Acquire);
        let recent_events_count = self.emergency_events.read().await.len();

        SystemStatus {
            global_shutdown: self.global_shutdown.load(Ordering::Acquire),
            manual_override: self.manual_override.load(Ordering::Acquire),
            circuit_breaker_states: breaker_states,
            daily_pnl,
            consecutive_losses,
            recent_events_count,
            can_trade: self.can_proceed().await.unwrap_or(false),
            timestamp: Utc::now(),
        }
    }

    /// Reset all systems (use with extreme caution)
    pub async fn reset_all_systems(&self, authorization_code: &str, reason: String) -> Result<()> {
        // Simple authorization check (in production, use proper auth)
        if authorization_code != "EMERGENCY_RESET_TRENCHBOT_2024" {
            return Err(anyhow::anyhow!("Invalid authorization code"));
        }

        warn!("🔄 RESETTING ALL EMERGENCY SYSTEMS: {}", reason);

        // Reset global flags
        self.global_shutdown.store(false, Ordering::Release);
        self.manual_override.store(false, Ordering::Release);

        // Reset all circuit breakers
        for breaker in self.circuit_breakers.values() {
            breaker.reset_manually(format!("System reset: {}", reason)).await;
        }

        // Reset counters
        self.consecutive_losses.store(0, Ordering::Release);
        *self.daily_pnl.write().await = 0.0;
        self.recent_losses.write().await.clear();

        info!("✅ All emergency systems have been reset");
        Ok(())
    }
}

#[derive(Debug, Serialize)]
pub struct SystemStatus {
    pub global_shutdown: bool,
    pub manual_override: bool,
    pub circuit_breaker_states: HashMap<CircuitType, CircuitState>,
    pub daily_pnl: f64,
    pub consecutive_losses: u64,
    pub recent_events_count: usize,
    pub can_trade: bool,
    pub timestamp: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, Duration as TokioDuration};

    #[tokio::test]
    async fn test_circuit_breaker_basic_operation() {
        let config = CircuitConfig::default();
        let breaker = CircuitBreaker::new(CircuitType::ConsecutiveLosses, config);
        
        // Should start in closed state
        assert_eq!(breaker.get_state(), CircuitState::Closed);
        assert!(breaker.can_proceed().await);
        
        // Record some failures
        for i in 0..4 {
            let tripped = breaker.record_failure(format!("Test failure {}", i)).await;
            assert!(!tripped); // Should not trip until threshold
        }
        
        // This should trip the breaker
        let tripped = breaker.record_failure("Final failure".to_string()).await;
        assert!(tripped);
        assert_eq!(breaker.get_state(), CircuitState::Open);
        assert!(!breaker.can_proceed().await);
    }

    #[tokio::test]
    async fn test_emergency_manager_loss_tracking() {
        let config = CircuitConfig {
            max_daily_loss_sol: 100.0,
            consecutive_loss_limit: 3,
            ..Default::default()
        };
        
        let (mut manager, _receiver) = EmergencyManager::new(config);
        
        // Record some losses
        manager.record_trade_result(-25.0, "test_1").await.unwrap();
        manager.record_trade_result(-30.0, "test_2").await.unwrap();
        
        let status = manager.get_system_status().await;
        assert_eq!(status.daily_pnl, -55.0);
        assert_eq!(status.consecutive_losses, 2);
        
        // Should still be able to trade
        assert!(status.can_trade);
        
        // One more loss to trigger consecutive loss breaker
        manager.record_trade_result(-10.0, "test_3").await.unwrap();
        
        let status = manager.get_system_status().await;
        assert_eq!(status.consecutive_losses, 3);
        
        // Check if consecutive loss breaker is tripped
        if let Some(CircuitState::Open) = status.circuit_breaker_states.get(&CircuitType::ConsecutiveLosses) {
            // Breaker should be tripped
        } else {
            panic!("Consecutive loss breaker should be open");
        }
    }

    #[tokio::test]
    async fn test_emergency_shutdown() {
        let config = CircuitConfig::default();
        let (manager, mut receiver) = EmergencyManager::new(config);
        
        // Should start in normal state
        assert!(manager.can_proceed().await.unwrap());
        
        // Trigger emergency shutdown
        manager.emergency_shutdown("Test shutdown".to_string()).await.unwrap();
        
        // Should block all operations
        assert!(!manager.can_proceed().await.unwrap());
        
        // Should receive emergency event
        let event = receiver.recv().await.unwrap();
        assert!(matches!(event.event_type, EmergencyEventType::ManualShutdown));
        assert_eq!(event.severity, EmergencySeverity::Critical);
    }

    #[tokio::test]
    async fn test_manual_override() {
        let config = CircuitConfig::default();
        let (manager, _receiver) = EmergencyManager::new(config);
        
        assert!(manager.can_proceed().await.unwrap());
        
        // Enable manual override
        manager.manual_override("Testing manual override".to_string(), true).await.unwrap();
        assert!(!manager.can_proceed().await.unwrap());
        
        // Disable manual override
        manager.manual_override("Test complete".to_string(), false).await.unwrap();
        assert!(manager.can_proceed().await.unwrap());
    }

    #[tokio::test]
    async fn test_system_reset() {
        let config = CircuitConfig::default();
        let (manager, _receiver) = EmergencyManager::new(config);
        
        // Trigger shutdown
        manager.emergency_shutdown("Test".to_string()).await.unwrap();
        assert!(!manager.can_proceed().await.unwrap());
        
        // Reset with correct authorization
        manager.reset_all_systems(
            "EMERGENCY_RESET_TRENCHBOT_2024", 
            "Test reset".to_string()
        ).await.unwrap();
        
        assert!(manager.can_proceed().await.unwrap());
        
        // Test invalid authorization
        let result = manager.reset_all_systems("wrong_code", "Test".to_string()).await;
        assert!(result.is_err());
    }
}