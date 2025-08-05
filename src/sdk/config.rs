use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use figment::{Figment, providers::{Toml, Env, Json, Format}};
use crate::sdk::permissions::{PermissionLevel, ToolPermission, PermissionSet};
use crate::CombatError;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SdkConfig {
    pub default_permission: PermissionLevel,
    pub tools: HashMap<String, ToolPermissionConfig>,
    pub handlers: HandlerConfig,
    pub security: SecurityConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolPermissionConfig {
    pub level: PermissionLevel,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandlerConfig {
    pub enable_default: bool,
    pub enable_trench: bool,
    pub enable_security: bool,
    pub trench_safety_enabled: bool,
    pub security_require_all_confirmation: bool,
    pub blocked_tools: Vec<String>,
    pub rate_limit_per_minute: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    pub log_all_tool_usage: bool,
    pub require_confirmation_for_financial: bool,
    pub require_confirmation_for_destructive: bool,
    pub max_transaction_amount: f64,
    pub production_confirmation_required: bool,
}

impl Default for SdkConfig {
    fn default() -> Self {
        let mut tools = HashMap::new();
        
        // Default tool permissions
        tools.insert("transfer".to_string(), ToolPermissionConfig {
            level: PermissionLevel::Prompt,
            description: "Financial transfer operations".to_string(),
        });
        
        tools.insert("withdraw".to_string(), ToolPermissionConfig {
            level: PermissionLevel::Prompt,
            description: "Withdrawal operations".to_string(),
        });
        
        tools.insert("trade".to_string(), ToolPermissionConfig {
            level: PermissionLevel::Prompt,
            description: "Trading operations".to_string(),
        });
        
        tools.insert("delete".to_string(), ToolPermissionConfig {
            level: PermissionLevel::Never,
            description: "Destructive delete operations".to_string(),
        });

        Self {
            default_permission: PermissionLevel::Prompt,
            tools,
            handlers: HandlerConfig {
                enable_default: true,
                enable_trench: true,
                enable_security: false,
                trench_safety_enabled: true,
                security_require_all_confirmation: false,
                blocked_tools: vec![
                    "rm".to_string(),
                    "delete".to_string(),
                    "format".to_string(),
                    "drop_table".to_string(),
                ],
                rate_limit_per_minute: 60,
            },
            security: SecurityConfig {
                log_all_tool_usage: true,
                require_confirmation_for_financial: true,
                require_confirmation_for_destructive: true,
                max_transaction_amount: 1000.0,
                production_confirmation_required: true,
            },
        }
    }
}

impl SdkConfig {
    /// Load configuration from files and environment
    pub fn load() -> Result<Self, CombatError> {
        Figment::new()
            .merge(Toml::file("sdk.toml").nested())
            .merge(Json::file("sdk.json").nested())
            .merge(Env::prefixed("TRENCH_SDK_"))
            .extract()
            .map_err(|e| CombatError::ConfigError(format!("SDK config error: {}", e)))
    }

    /// Convert to PermissionSet for use with ToolConfirmationManager
    pub fn to_permission_set(&self) -> PermissionSet {
        let mut permission_set = PermissionSet::new(self.default_permission.clone());
        
        for (tool_name, config) in &self.tools {
            let permission = ToolPermission::new(
                config.level.clone(),
                config.description.clone(),
            );
            permission_set.add_tool_permission(tool_name.clone(), permission);
        }
        
        permission_set
    }

    /// Save configuration to file
    pub fn save_to_file(&self, path: &str) -> Result<(), CombatError> {
        let toml_content = toml::to_string_pretty(self)
            .map_err(|e| CombatError::ConfigError(format!("Failed to serialize config: {}", e)))?;
        
        std::fs::write(path, toml_content)
            .map_err(|e| CombatError::ConfigError(format!("Failed to write config file: {}", e)))?;
        
        Ok(())
    }

    /// Update tool permission
    pub fn update_tool_permission(&mut self, tool_name: String, level: PermissionLevel, description: String) {
        self.tools.insert(tool_name, ToolPermissionConfig { level, description });
    }

    /// Remove tool permission (will use default)
    pub fn remove_tool_permission(&mut self, tool_name: &str) {
        self.tools.remove(tool_name);
    }

    /// Get effective permission level for a tool
    pub fn get_effective_permission(&self, tool_name: &str) -> PermissionLevel {
        self.tools
            .get(tool_name)
            .map(|config| config.level.clone())
            .unwrap_or_else(|| self.default_permission.clone())
    }
}