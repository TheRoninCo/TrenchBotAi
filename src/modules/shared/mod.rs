// src/modules/shared/mod.rs
pub mod classifiers;
pub mod coordinator;
pub mod metrics;
pub mod sinks;
pub mod training;
pub mod health;

pub use classifiers::CombatRation;
pub use coordinator::RedisCoordinator;
pub use metrics::MevMetrics;
pub use sinks::{MongoSink, BatchedS3Sink, KafkaSink};
pub use training::{TrainingLogger, WolframExporter};
pub use health::SinkHealth;