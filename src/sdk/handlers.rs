use async_trait::async_trait;
use std::collections::HashMap;
use tracing::{info, warn, error};
use crate::sdk::tool_confirmation::{CanUseToolCallback, ToolRequest, ToolConfirmationResult};

/// Default tool handler that logs all tool usage
pub struct DefaultToolHandler {
    pub name: String,
}

impl DefaultToolHandler {
    pub fn new(name: String) -> Self {
        Self { name }
    }
}

#[async_trait]
impl CanUseToolCallback for DefaultToolHandler {
    async fn can_use_tool(&self, request: &ToolRequest) -> ToolConfirmationResult {
        info!(
            handler = %self.name,
            tool = %request.tool_name,
            context = %request.context,
            "Tool usage request received"
        );
        
        // Default behavior: allow all tools but log them
        ToolConfirmationResult::Allow
    }

    async fn tool_executed(&self, request: &ToolRequest, success: bool, result: Option<String>) {
        if success {
            info!(
                handler = %self.name,
                tool = %request.tool_name,
                "Tool executed successfully"
            );
        } else {
            warn!(
                handler = %self.name,
                tool = %request.tool_name,
                result = ?result,
                "Tool execution failed"
            );
        }
    }
}

/// TrenchBot-specific tool handler with safety protocols
pub struct TrenchToolHandler {
    pub safety_enabled: bool,
    pub blocked_tools: Vec<String>,
    pub max_executions_per_minute: u32,
    execution_count: HashMap<String, (chrono::DateTime<chrono::Utc>, u32)>,
}

impl TrenchToolHandler {
    pub fn new(safety_enabled: bool) -> Self {
        Self {
            safety_enabled,
            blocked_tools: vec![
                "rm".to_string(),
                "delete".to_string(),
                "format".to_string(),
                "drop_table".to_string(),
            ],
            max_executions_per_minute: 60,
            execution_count: HashMap::new(),
        }
    }

    pub fn with_blocked_tools(mut self, tools: Vec<String>) -> Self {
        self.blocked_tools = tools;
        self
    }

    pub fn with_rate_limit(mut self, max_per_minute: u32) -> Self {
        self.max_executions_per_minute = max_per_minute;
        self
    }

    fn is_rate_limited(&mut self, tool_name: &str) -> bool {
        let now = chrono::Utc::now();
        let minute_ago = now - chrono::Duration::minutes(1);

        match self.execution_count.get_mut(tool_name) {
            Some((last_reset, count)) => {
                if *last_reset < minute_ago {
                    // Reset counter
                    *last_reset = now;
                    *count = 1;
                    false
                } else if *count >= self.max_executions_per_minute {
                    true
                } else {
                    *count += 1;
                    false
                }
            }
            None => {
                self.execution_count.insert(tool_name.to_string(), (now, 1));
                false
            }
        }
    }

    fn is_dangerous_tool(&self, tool_name: &str) -> bool {
        self.blocked_tools.iter().any(|blocked| tool_name.contains(blocked))
    }
}

#[async_trait]
impl CanUseToolCallback for TrenchToolHandler {
    async fn can_use_tool(&self, request: &ToolRequest) -> ToolConfirmationResult {
        // Safety protocol check
        if self.safety_enabled {
            if self.is_dangerous_tool(&request.tool_name) {
                error!(
                    tool = %request.tool_name,
                    "SAFETY PROTOCOL: Dangerous tool blocked"
                );
                return ToolConfirmationResult::Deny(
                    format!("Safety protocol blocked dangerous tool: {}", request.tool_name)
                );
            }
        }

        // Rate limiting check (we need &mut self for this, but async_trait doesn't allow it)
        // This would need to be implemented with Arc<Mutex<>> or similar for production use
        
        // Context-based analysis
        if request.context.contains("production") || request.context.contains("live") {
            warn!(
                tool = %request.tool_name,
                context = %request.context,
                "Production environment tool usage requires confirmation"
            );
            return ToolConfirmationResult::Prompt(
                format!("Tool '{}' requested in production context. Confirm usage?", request.tool_name)
            );
        }

        // Financial operation check
        if request.tool_name.contains("transfer") || 
           request.tool_name.contains("withdraw") || 
           request.tool_name.contains("trade") {
            
            // Check if amount is specified and if it's significant
            if let Some(amount) = request.parameters.get("amount") {
                if let Some(amount_val) = amount.as_f64() {
                    if amount_val > 1000.0 { // Significant amount threshold
                        return ToolConfirmationResult::Prompt(
                            format!("Large financial operation: {} with amount {}. Confirm?", 
                                   request.tool_name, amount_val)
                        );
                    }
                }
            }
        }

        info!(
            tool = %request.tool_name,
            context = %request.context,
            "TrenchBot tool usage approved"
        );

        ToolConfirmationResult::Allow
    }

    async fn tool_executed(&self, request: &ToolRequest, success: bool, result: Option<String>) {
        if success {
            info!(
                tool = %request.tool_name,
                context = %request.context,
                "TrenchBot tool executed successfully"
            );
        } else {
            error!(
                tool = %request.tool_name,
                context = %request.context,
                result = ?result,
                "TrenchBot tool execution failed"
            );
        }
    }
}

/// Security-focused handler for high-stakes operations
pub struct SecurityHandler {
    pub require_confirmation_for_all: bool,
}

impl SecurityHandler {
    pub fn new(require_confirmation_for_all: bool) -> Self {
        Self {
            require_confirmation_for_all,
        }
    }
}

#[async_trait]
impl CanUseToolCallback for SecurityHandler {
    async fn can_use_tool(&self, request: &ToolRequest) -> ToolConfirmationResult {
        if self.require_confirmation_for_all {
            return ToolConfirmationResult::Prompt(
                format!("Security handler requires confirmation for tool: {}", request.tool_name)
            );
        }

        // Always allow if not in strict mode
        ToolConfirmationResult::Allow
    }

    async fn tool_executed(&self, request: &ToolRequest, success: bool, _result: Option<String>) {
        // Security logging
        info!(
            tool = %request.tool_name,
            success = success,
            timestamp = %chrono::Utc::now(),
            "Security audit log: tool execution"
        );
    }
}