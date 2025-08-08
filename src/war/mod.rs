pub mod engine;
pub mod munitions;  
pub mod receipts;
pub mod tracking;
pub mod runescape_rankings;
pub mod warfare_themes;
pub mod circuit_breakers;
pub mod position_management;
pub mod ultra_low_latency;
pub mod integration_tests;

pub use engine::RecruitmentRebate;
pub use tracking::{Recruit, RecruitStatus};
pub use receipts::RebateReceipt;
pub use runescape_rankings::{
    RuneScapeRankingSystem, CombatLevel, CombatSkill, MonsterKill, 
    SpecialAttack, RareDrop, PlayerStats, KillEntry
};
pub use warfare_themes::{
    WarfareTheme, ThemeConfig, ThemeableWarfareSystem, RankTier, TargetType
};
pub use circuit_breakers::{
    EmergencyManager, CircuitBreaker, CircuitConfig, CircuitType, CircuitState, 
    EmergencyEvent, EmergencyEventType, EmergencySeverity, SystemStatus
};
pub use position_management::{
    PositionManager, RiskConfig, PositionSizing, PositionInfo, MarketConditions,
    PositionValidation, RiskReport, LiquidityConditions, CorrelationRegime
};
pub use ultra_low_latency::{
    UltraLowLatencyEngine, UltraFastMatchingEngine, SIMDPriceEngine, 
    OptimizedDecisionEngine, LatencyProfiler, PerformanceReport, Trade
};

/// Main entry point for war module operations
pub fn process_rebates() {
    // TODO: Implement main rebate processing logic
}