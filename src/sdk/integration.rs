use std::collections::HashMap;
use std::sync::Arc;
use tracing::{info, error};
use crate::{TrenchConfig, CombatError};
use crate::sdk::{
    ToolConfirmationManager, DefaultToolHandler, TrenchToolHandler, SecurityHandler,
    SdkConfig, ToolRequest, ToolConfirmationResult, PermissionLevel
};

/// TrenchBot SDK integration wrapper
pub struct TrenchSdkManager {
    pub confirmation_manager: ToolConfirmationManager,
    pub config: SdkConfig,
}

impl TrenchSdkManager {
    /// Initialize SDK with TrenchBot configuration
    pub async fn new(trench_config: &TrenchConfig) -> Result<Self, CombatError> {
        // Load SDK configuration
        let sdk_config = SdkConfig::load().unwrap_or_else(|_| {
            info!("Using default SDK configuration");
            SdkConfig::default()
        });

        // Create confirmation manager with appropriate default permission
        let default_permission = match trench_config.fire_control {
            crate::FireMode::Cold => PermissionLevel::Always,    // Recon mode - allow all
            crate::FireMode::Warm => PermissionLevel::Prompt,    // Simulation - prompt
            crate::FireMode::Hot | crate::FireMode::Inferno => PermissionLevel::Never, // Combat - block by default
        };

        let mut manager = ToolConfirmationManager::new(default_permission);

        // Add handlers based on configuration
        if sdk_config.handlers.enable_default {
            let default_handler = DefaultToolHandler::new("TrenchBot-Default".to_string());
            manager.add_callback(Box::new(default_handler));
        }

        if sdk_config.handlers.enable_trench {
            let trench_handler = TrenchToolHandler::new(sdk_config.handlers.trench_safety_enabled)
                .with_blocked_tools(sdk_config.handlers.blocked_tools.clone())
                .with_rate_limit(sdk_config.handlers.rate_limit_per_minute);
            manager.add_callback(Box::new(trench_handler));
        }

        if sdk_config.handlers.enable_security {
            let security_handler = SecurityHandler::new(sdk_config.handlers.security_require_all_confirmation);
            manager.add_callback(Box::new(security_handler));
        }

        // Apply permission set
        let permission_set = sdk_config.to_permission_set();
        for (tool_name, permission) in permission_set.tools {
            manager.set_tool_permission(tool_name, permission).await;
        }

        Ok(Self {
            confirmation_manager: manager,
            config: sdk_config,
        })
    }

    /// Execute a tool with confirmation
    pub async fn execute_tool<F, Fut, T>(&self, 
        tool_name: &str, 
        parameters: HashMap<String, serde_json::Value>,
        context: &str,
        tool_fn: F
    ) -> Result<T, CombatError> 
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<T, CombatError>>,
    {
        // Create tool request
        let request = ToolRequest {
            tool_name: tool_name.to_string(),
            parameters,
            context: context.to_string(),
            timestamp: chrono::Utc::now(),
        };

        // Check if tool usage is allowed
        match self.confirmation_manager.confirm_tool_usage(&request).await {
            ToolConfirmationResult::Allow => {
                info!(tool = %tool_name, "Tool usage confirmed");
            }
            ToolConfirmationResult::Deny(reason) => {
                error!(tool = %tool_name, reason = %reason, "Tool usage denied");
                return Err(CombatError::SafetyLock(format!("Tool '{}' denied: {}", tool_name, reason)));
            }
            ToolConfirmationResult::Prompt(message) => {
                // In a real implementation, this would prompt the user
                // For now, we'll log and allow based on context
                info!(tool = %tool_name, message = %message, "Tool requires confirmation");
                
                // Auto-approve for simulation contexts, deny for production
                if context.contains("simulation") || context.contains("test") {
                    info!("Auto-approving tool for simulation context");
                } else {
                    error!("Tool requires manual confirmation in production context");
                    return Err(CombatError::SafetyLock(format!("Tool '{}' requires confirmation: {}", tool_name, message)));
                }
            }
        }

        // Execute the tool
        let result = tool_fn().await;
        let success = result.is_ok();
        let result_msg = if success { 
            Some("Success".to_string()) 
        } else { 
            Some(format!("Error: {:?}", result.as_ref().err())) 
        };

        // Notify handlers of execution
        self.confirmation_manager.notify_tool_executed(&request, success, result_msg).await;

        result
    }

    /// Update tool permission dynamically
    pub async fn update_permission(&self, tool_name: String, level: PermissionLevel, description: String) {
        let permission = crate::sdk::ToolPermission::new(level, description);
        self.confirmation_manager.set_tool_permission(tool_name, permission).await;
    }

    /// Get current configuration
    pub fn get_config(&self) -> &SdkConfig {
        &self.config
    }
}

/// Example usage wrapper for common TrenchBot operations
pub struct TrenchToolExecutor {
    sdk_manager: Arc<TrenchSdkManager>,
}

impl TrenchToolExecutor {
    pub fn new(sdk_manager: Arc<TrenchSdkManager>) -> Self {
        Self { sdk_manager }
    }

    /// Execute a trading operation with confirmation
    pub async fn execute_trade(&self, 
        trade_type: &str, 
        amount: f64, 
        token: &str
    ) -> Result<String, CombatError> {
        let mut params = HashMap::new();
        params.insert("amount".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(amount).unwrap()));
        params.insert("token".to_string(), serde_json::Value::String(token.to_string()));
        params.insert("trade_type".to_string(), serde_json::Value::String(trade_type.to_string()));

        self.sdk_manager.execute_tool(
            "trade",
            params,
            "trading_engine",
            || async {
                // Simulate trade execution
                info!(trade_type = %trade_type, amount = %amount, token = %token, "Executing trade");
                Ok(format!("Trade executed: {} {} {}", trade_type, amount, token))
            }
        ).await
    }

    /// Execute a transfer with confirmation
    pub async fn execute_transfer(&self, 
        to_address: &str, 
        amount: f64
    ) -> Result<String, CombatError> {
        let mut params = HashMap::new();
        params.insert("to_address".to_string(), serde_json::Value::String(to_address.to_string()));
        params.insert("amount".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(amount).unwrap()));

        self.sdk_manager.execute_tool(
            "transfer",
            params,
            "wallet_manager",
            || async {
                // Simulate transfer execution
                info!(to = %to_address, amount = %amount, "Executing transfer");
                Ok(format!("Transfer executed: {} to {}", amount, to_address))
            }
        ).await
    }
}