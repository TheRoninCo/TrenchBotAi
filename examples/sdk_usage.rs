use std::collections::HashMap;
use std::sync::Arc;
use tokio;
use tracing::{info, Level};
use trenchbot_dex::{TrenchConfig, FireMode, SafetyProtocol, WarChest};
use trenchbot_dex::sdk::{
    TrenchSdkManager, TrenchToolExecutor, PermissionLevel,
    ToolRequest, ToolConfirmationResult
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .init();

    info!("Starting TrenchBot SDK Example");

    // Create a sample TrenchBot configuration
    let trench_config = TrenchConfig {
        fire_control: FireMode::Warm, // Simulation mode
        safety: SafetyProtocol {
            max_daily_loss: 1000.0,
            position_cap: 10000.0,
            gas_abort_threshold: 100.0,
        },
        warchest: WarChest {
            total_rounds: 50000.0,
            round_size: 100.0,
            reserve_ammo: 5000.0,
        },
        allocation: Default::default(),
        killswitch: Default::default(),
    };

    // Initialize the SDK manager
    let sdk_manager = Arc::new(TrenchSdkManager::new(&trench_config).await?);
    info!("SDK Manager initialized");

    // Create a tool executor
    let executor = TrenchToolExecutor::new(sdk_manager.clone());

    // Example 1: Execute a small trade (should be allowed)
    info!("=== Example 1: Small Trade ===");
    match executor.execute_trade("buy", 100.0, "SOL").await {
        Ok(result) => info!("Trade successful: {}", result),
        Err(e) => info!("Trade failed: {}", e),
    }

    // Example 2: Execute a large trade (should prompt for confirmation)
    info!("=== Example 2: Large Trade ===");
    match executor.execute_trade("sell", 5000.0, "USDC").await {
        Ok(result) => info!("Large trade successful: {}", result),
        Err(e) => info!("Large trade blocked: {}", e),
    }

    // Example 3: Execute a transfer (should prompt)
    info!("=== Example 3: Transfer ===");
    match executor.execute_transfer("9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM", 500.0).await {
        Ok(result) => info!("Transfer successful: {}", result),
        Err(e) => info!("Transfer blocked: {}", e),
    }

    // Example 4: Direct tool confirmation testing
    info!("=== Example 4: Direct Tool Confirmation ===");
    
    // Test a query operation (should be allowed)
    let query_request = ToolRequest {
        tool_name: "query".to_string(),
        parameters: HashMap::new(),
        context: "data_analysis".to_string(),
        timestamp: chrono::Utc::now(),
    };
    
    match sdk_manager.confirmation_manager.confirm_tool_usage(&query_request).await {
        ToolConfirmationResult::Allow => info!("Query tool: ALLOWED"),
        ToolConfirmationResult::Deny(reason) => info!("Query tool: DENIED - {}", reason),
        ToolConfirmationResult::Prompt(message) => info!("Query tool: PROMPT - {}", message),
    }

    // Test a dangerous operation (should be denied)
    let delete_request = ToolRequest {
        tool_name: "delete".to_string(),
        parameters: HashMap::new(),
        context: "cleanup".to_string(),
        timestamp: chrono::Utc::now(),
    };
    
    match sdk_manager.confirmation_manager.confirm_tool_usage(&delete_request).await {
        ToolConfirmationResult::Allow => info!("Delete tool: ALLOWED"),
        ToolConfirmationResult::Deny(reason) => info!("Delete tool: DENIED - {}", reason),
        ToolConfirmationResult::Prompt(message) => info!("Delete tool: PROMPT - {}", message),
    }

    // Example 5: Dynamic permission updates
    info!("=== Example 5: Dynamic Permission Updates ===");
    
    // Update permission for a specific tool
    sdk_manager.update_permission(
        "analyze".to_string(),
        PermissionLevel::Always,
        "Analysis operations are always safe".to_string()
    ).await;
    
    let analyze_request = ToolRequest {
        tool_name: "analyze".to_string(),
        parameters: HashMap::new(),
        context: "market_analysis".to_string(),
        timestamp: chrono::Utc::now(),
    };
    
    match sdk_manager.confirmation_manager.confirm_tool_usage(&analyze_request).await {
        ToolConfirmationResult::Allow => info!("Analyze tool: ALLOWED (updated permission)"),
        ToolConfirmationResult::Deny(reason) => info!("Analyze tool: DENIED - {}", reason),
        ToolConfirmationResult::Prompt(message) => info!("Analyze tool: PROMPT - {}", message),
    }

    // Example 6: Show current configuration
    info!("=== Example 6: Current Configuration ===");
    let config = sdk_manager.get_config();
    info!("Default permission: {:?}", config.default_permission);
    info!("Handlers enabled - Default: {}, Trench: {}, Security: {}", 
          config.handlers.enable_default,
          config.handlers.enable_trench,
          config.handlers.enable_security);
    info!("Blocked tools: {:?}", config.handlers.blocked_tools);
    info!("Max transaction amount: {}", config.security.max_transaction_amount);

    info!("SDK Example completed successfully");
    Ok(())
}

// Helper function to simulate user confirmation
async fn simulate_user_confirmation(message: &str) -> bool {
    info!("USER PROMPT: {}", message);
    info!("Simulating user response: APPROVED");
    true // In a real implementation, this would wait for user input
}