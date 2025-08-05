pub mod tool_confirmation;
pub mod permissions;
pub mod handlers;
pub mod config;
pub mod integration;
mod tests;

pub use tool_confirmation::{CanUseToolCallback, ToolConfirmationManager, ToolRequest, ToolConfirmationResult};
pub use permissions::{ToolPermission, PermissionLevel, PermissionSet};
pub use handlers::{DefaultToolHandler, TrenchToolHandler, SecurityHandler};
pub use config::{SdkConfig, ToolPermissionConfig, HandlerConfig, SecurityConfig};
pub use integration::{TrenchSdkManager, TrenchToolExecutor};