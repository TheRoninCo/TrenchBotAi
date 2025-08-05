use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PermissionLevel {
    Always,  // Always allow without confirmation
    Never,   // Never allow
    Prompt,  // Prompt for confirmation
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolPermission {
    pub level: PermissionLevel,
    pub description: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

impl ToolPermission {
    pub fn new(level: PermissionLevel, description: String) -> Self {
        let now = chrono::Utc::now();
        Self {
            level,
            description,
            created_at: now,
            updated_at: now,
        }
    }

    pub fn always(description: String) -> Self {
        Self::new(PermissionLevel::Always, description)
    }

    pub fn never(description: String) -> Self {
        Self::new(PermissionLevel::Never, description)
    }

    pub fn prompt(description: String) -> Self {
        Self::new(PermissionLevel::Prompt, description)
    }

    pub fn update_level(&mut self, level: PermissionLevel) {
        self.level = level;
        self.updated_at = chrono::Utc::now();
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PermissionSet {
    pub tools: HashMap<String, ToolPermission>,
    pub default_level: PermissionLevel,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

impl PermissionSet {
    pub fn new(default_level: PermissionLevel) -> Self {
        Self {
            tools: HashMap::new(),
            default_level,
            created_at: chrono::Utc::now(),
        }
    }

    pub fn add_tool_permission(&mut self, tool_name: String, permission: ToolPermission) {
        self.tools.insert(tool_name, permission);
    }

    pub fn get_tool_permission(&self, tool_name: &str) -> Option<&ToolPermission> {
        self.tools.get(tool_name)
    }

    pub fn remove_tool_permission(&mut self, tool_name: &str) -> Option<ToolPermission> {
        self.tools.remove(tool_name)
    }

    /// Get effective permission level for a tool (uses default if not explicitly set)
    pub fn get_effective_level(&self, tool_name: &str) -> PermissionLevel {
        self.tools
            .get(tool_name)
            .map(|p| p.level.clone())
            .unwrap_or_else(|| self.default_level.clone())
    }
}

impl Default for PermissionSet {
    fn default() -> Self {
        Self::new(PermissionLevel::Prompt)
    }
}