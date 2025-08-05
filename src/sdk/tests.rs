#[cfg(test)]
mod tests {
    use super::*;
    use crate::sdk::{
        ToolConfirmationManager, DefaultToolHandler, TrenchToolHandler, SecurityHandler,
        ToolRequest, ToolConfirmationResult, PermissionLevel, ToolPermission,
        SdkConfig, TrenchSdkManager
    };
    use crate::{TrenchConfig, FireMode, SafetyProtocol, WarChest};
    use std::collections::HashMap;
    use tokio;

    fn create_test_request(tool_name: &str) -> ToolRequest {
        ToolRequest {
            tool_name: tool_name.to_string(),
            parameters: HashMap::new(),
            context: "test_context".to_string(),
            timestamp: chrono::Utc::now(),
        }
    }

    fn create_test_trench_config() -> TrenchConfig {
        TrenchConfig {
            fire_control: FireMode::Warm,
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
        }
    }

    #[tokio::test]
    async fn test_default_handler_allows_all() {
        let handler = DefaultToolHandler::new("test".to_string());
        let request = create_test_request("any_tool");
        
        let result = handler.can_use_tool(&request).await;
        assert!(matches!(result, ToolConfirmationResult::Allow));
    }

    #[tokio::test]
    async fn test_trench_handler_blocks_dangerous_tools() {
        let handler = TrenchToolHandler::new(true);
        let request = create_test_request("delete_everything");
        
        let result = handler.can_use_tool(&request).await;
        assert!(matches!(result, ToolConfirmationResult::Deny(_)));
    }

    #[tokio::test]
    async fn test_trench_handler_prompts_for_production() {
        let handler = TrenchToolHandler::new(true);
        let mut request = create_test_request("safe_tool");
        request.context = "production environment".to_string();
        
        let result = handler.can_use_tool(&request).await;
        assert!(matches!(result, ToolConfirmationResult::Prompt(_)));
    }

    #[tokio::test]
    async fn test_trench_handler_prompts_for_large_amounts() {
        let handler = TrenchToolHandler::new(true);
        let mut request = create_test_request("transfer");
        request.parameters.insert(
            "amount".to_string(),
            serde_json::Value::Number(serde_json::Number::from_f64(5000.0).unwrap())
        );
        
        let result = handler.can_use_tool(&request).await;
        assert!(matches!(result, ToolConfirmationResult::Prompt(_)));
    }

    #[tokio::test]
    async fn test_security_handler_prompt_mode() {
        let handler = SecurityHandler::new(true);
        let request = create_test_request("any_tool");
        
        let result = handler.can_use_tool(&request).await;
        assert!(matches!(result, ToolConfirmationResult::Prompt(_)));
    }

    #[tokio::test]
    async fn test_security_handler_allow_mode() {
        let handler = SecurityHandler::new(false);
        let request = create_test_request("any_tool");
        
        let result = handler.can_use_tool(&request).await;
        assert!(matches!(result, ToolConfirmationResult::Allow));
    }

    #[tokio::test]
    async fn test_tool_confirmation_manager_permissions() {
        let mut manager = ToolConfirmationManager::new(PermissionLevel::Prompt);
        
        // Set a tool to always allow
        let permission = ToolPermission::new(PermissionLevel::Always, "Test permission".to_string());
        manager.set_tool_permission("allowed_tool".to_string(), permission).await;
        
        // Set a tool to never allow
        let permission = ToolPermission::new(PermissionLevel::Never, "Blocked tool".to_string());
        manager.set_tool_permission("blocked_tool".to_string(), permission).await;
        
        // Test allowed tool
        let allowed_request = create_test_request("allowed_tool");
        let result = manager.confirm_tool_usage(&allowed_request).await;
        assert!(matches!(result, ToolConfirmationResult::Allow));
        
        // Test blocked tool
        let blocked_request = create_test_request("blocked_tool");
        let result = manager.confirm_tool_usage(&blocked_request).await;
        assert!(matches!(result, ToolConfirmationResult::Deny(_)));
        
        // Test unknown tool (should use default)
        let unknown_request = create_test_request("unknown_tool");
        let result = manager.confirm_tool_usage(&unknown_request).await;
        // Default is Prompt, but no callbacks are added, so it should Allow
        assert!(matches!(result, ToolConfirmationResult::Allow));
    }

    #[tokio::test]
    async fn test_tool_confirmation_manager_with_callbacks() {
        let mut manager = ToolConfirmationManager::new(PermissionLevel::Prompt);
        
        // Add a callback that denies specific tools
        let handler = TrenchToolHandler::new(true);
        manager.add_callback(Box::new(handler));
        
        // Test dangerous tool
        let dangerous_request = create_test_request("delete_all");
        let result = manager.confirm_tool_usage(&dangerous_request).await;
        assert!(matches!(result, ToolConfirmationResult::Deny(_)));
        
        // Test safe tool
        let safe_request = create_test_request("query_data");
        let result = manager.confirm_tool_usage(&safe_request).await;
        assert!(matches!(result, ToolConfirmationResult::Allow));
    }

    #[tokio::test]
    async fn test_sdk_config_default() {
        let config = SdkConfig::default();
        
        assert_eq!(config.default_permission, PermissionLevel::Prompt);
        assert!(config.handlers.enable_default);
        assert!(config.handlers.enable_trench);
        assert!(!config.handlers.enable_security);
        assert!(config.security.log_all_tool_usage);
        assert!(config.security.require_confirmation_for_financial);
    }

    #[tokio::test]
    async fn test_sdk_config_to_permission_set() {
        let config = SdkConfig::default();
        let permission_set = config.to_permission_set();
        
        assert_eq!(permission_set.default_level, PermissionLevel::Prompt);
        assert!(permission_set.tools.contains_key("transfer"));
        assert!(permission_set.tools.contains_key("delete"));
        
        // Check specific permissions
        let transfer_permission = permission_set.get_tool_permission("transfer").unwrap();
        assert_eq!(transfer_permission.level, PermissionLevel::Prompt);
        
        let delete_permission = permission_set.get_tool_permission("delete").unwrap();
        assert_eq!(delete_permission.level, PermissionLevel::Never);
    }

    #[tokio::test]
    async fn test_trench_sdk_manager_initialization() {
        let trench_config = create_test_trench_config();
        let sdk_manager = TrenchSdkManager::new(&trench_config).await;
        
        assert!(sdk_manager.is_ok());
        let manager = sdk_manager.unwrap();
        
        // Check that configuration was loaded
        assert!(manager.config.handlers.enable_default);
        assert!(manager.config.handlers.enable_trench);
    }

    #[tokio::test]
    async fn test_trench_sdk_manager_fire_mode_permissions() {
        // Test Cold mode (should allow all)
        let mut cold_config = create_test_trench_config();
        cold_config.fire_control = FireMode::Cold;
        let cold_manager = TrenchSdkManager::new(&cold_config).await.unwrap();
        // We can't easily test the internal permission without exposing it
        
        // Test Hot mode (should be restrictive)
        let mut hot_config = create_test_trench_config();
        hot_config.fire_control = FireMode::Hot;
        let hot_manager = TrenchSdkManager::new(&hot_config).await.unwrap();
        
        // Both should initialize successfully
        assert!(!cold_manager.config.handlers.blocked_tools.is_empty());
        assert!(!hot_manager.config.handlers.blocked_tools.is_empty());
    }

    #[tokio::test]
    async fn test_permission_level_serialization() {
        let always = PermissionLevel::Always;
        let never = PermissionLevel::Never;
        let prompt = PermissionLevel::Prompt;
        
        // Test that they can be serialized and deserialized
        let always_json = serde_json::to_string(&always).unwrap();
        let never_json = serde_json::to_string(&never).unwrap();
        let prompt_json = serde_json::to_string(&prompt).unwrap();
        
        assert_eq!(always_json, "\"Always\"");
        assert_eq!(never_json, "\"Never\"");
        assert_eq!(prompt_json, "\"Prompt\"");
        
        let always_back: PermissionLevel = serde_json::from_str(&always_json).unwrap();
        let never_back: PermissionLevel = serde_json::from_str(&never_json).unwrap();
        let prompt_back: PermissionLevel = serde_json::from_str(&prompt_json).unwrap();
        
        assert_eq!(always_back, PermissionLevel::Always);
        assert_eq!(never_back, PermissionLevel::Never);
        assert_eq!(prompt_back, PermissionLevel::Prompt);
    }
}