pub mod health;
pub mod auth;
pub mod dashboard;
pub mod portfolio;
pub mod trading;
pub mod copy_trading;
pub mod social;
pub mod rankings;
pub mod achievements;
pub mod notifications;
pub mod metrics;
pub mod admin;
pub mod static_files;

// Re-export all handlers
pub use health::*;
pub use auth::*;
pub use dashboard::*;
pub use portfolio::*;
pub use trading::*;
pub use copy_trading::*;
pub use social::*;
pub use rankings::*;
pub use achievements::*;
pub use notifications::*;
pub use metrics::*;
pub use admin::*;
pub use static_files::*;