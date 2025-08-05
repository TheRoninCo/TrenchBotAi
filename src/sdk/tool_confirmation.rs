use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::sync::RwLock;
use crate::sdk::permissions::{ToolPermission, PermissionLevel};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolRequest {
    pub tool_name: String,
    pub parameters: HashMap<String, serde_json::Value>,
    pub context: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ToolConfirmationResult {
    Allow,
    Deny(String),
    Prompt(String),
}

/// Core trait for tool confirmation callbacks
#[async_trait]
pub trait CanUseToolCallback: Send + Sync {
    /// Called before a tool is executed to determine if it should be allowed
    async fn can_use_tool(&self, request: &ToolRequest) -> ToolConfirmationResult;
    
    /// Called after a tool is executed with the result
    async fn tool_executed(&self, request: &ToolRequest, success: bool, result: Option<String>);
}

/// Manager for tool confirmation with multiple callbacks and permissions
pub struct ToolConfirmationManager {
    callbacks: Vec<Box<dyn CanUseToolCallback>>,
    permissions: RwLock<HashMap<String, ToolPermission>>,
    default_permission: PermissionLevel,
}

impl ToolConfirmationManager {
    pub fn new(default_permission: PermissionLevel) -> Self {
        Self {
            callbacks: Vec::new(),
            permissions: RwLock::new(HashMap::new()),
            default_permission,
        }
    }

    /// Add a callback to the confirmation chain
    pub fn add_callback(&mut self, callback: Box<dyn CanUseToolCallback>) {
        self.callbacks.push(callback);
    }

    /// Set permission for a specific tool
    pub async fn set_tool_permission(&self, tool_name: String, permission: ToolPermission) {
        let mut perms = self.permissions.write().await;
        perms.insert(tool_name, permission);
    }

    /// Check if a tool can be used by consulting all callbacks and permissions
    pub async fn confirm_tool_usage(&self, request: &ToolRequest) -> ToolConfirmationResult {
        // First check static permissions
        let perms = self.permissions.read().await;
        if let Some(permission) = perms.get(&request.tool_name) {
            match permission.level {
                PermissionLevel::Always => return ToolConfirmationResult::Allow,
                PermissionLevel::Never => return ToolConfirmationResult::Deny("Tool blocked by permission".to_string()),
                PermissionLevel::Prompt => {} // Continue to callbacks
            }
        } else {
            // Use default permission if no specific permission set
            match self.default_permission {
                PermissionLevel::Always => return ToolConfirmationResult::Allow,
                PermissionLevel::Never => return ToolConfirmationResult::Deny("Tool blocked by default permission".to_string()),
                PermissionLevel::Prompt => {} // Continue to callbacks
            }
        }
        drop(perms);

        // If we reach here, we need to consult callbacks
        for callback in &self.callbacks {
            match callback.can_use_tool(request).await {
                ToolConfirmationResult::Allow => continue, // Check next callback
                ToolConfirmationResult::Deny(reason) => return ToolConfirmationResult::Deny(reason),
                ToolConfirmationResult::Prompt(message) => return ToolConfirmationResult::Prompt(message),
            }
        }

        // If all callbacks allow, then allow
        ToolConfirmationResult::Allow
    }

    /// Notify all callbacks that a tool was executed
    pub async fn notify_tool_executed(&self, request: &ToolRequest, success: bool, result: Option<String>) {
        for callback in &self.callbacks {
            callback.tool_executed(request, success, result.clone()).await;
        }
    }
}