use super::{LogSink, SinkHealth, RetryPolicy};
use crate::observability::CombatLog;
use crate::observability::LogError;
use prometheus::HistogramTimer;
use std::sync::Arc;

pub struct InstrumentedSink {
    inner: Arc<dyn LogSink>,
}

impl InstrumentedSink {
    /// Wrap an existing sink so that its calls are recorded in Prometheus metrics.
    pub fn new(inner: Arc<dyn LogSink>) -> Self {
        Self { inner }
    }
}

#[async_trait::async_trait]
impl LogSink for InstrumentedSink {
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    async fn write(&self, log: &CombatLog) -> Result<(), LogError> {
        // Start the histogram timer
        let timer: HistogramTimer = SINK_LATENCY
            .with_label_values(&[self.name()])
            .start_timer();
        
        // Increment total attempts
        LOGS_READ
            .with_label_values(&[self.name()])
            .inc();

        // Delegate to inner sink
        let result = self.inner.write(log).await;

        // Observe duration
        timer.observe_duration();

        // If error, bump failure counter
        if let Err(e) = &result {
            LOGS_FAILED
                .with_label_values(&[self.name(), e.to_string().as_str()])
                .inc();
        }

        result
    }

    async fn flush(&self) -> Result<(), LogError> {
        self.inner.flush().await
    }

    fn health_check(&self) -> SinkHealth {
        self.inner.health_check()
    }

    fn retry_policy(&self) -> RetryPolicy {
        self.inner.retry_policy()
    }

    fn diagnostics(&self) -> serde_json::Value {
        self.inner.diagnostics()
    }
}
