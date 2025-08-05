use prometheus::{
    IntCounterVec, HistogramVec, Opts, HistogramOpts, Registry,
    Encoder, TextEncoder
};
use lazy_static::lazy_static;
use std::net::SocketAddr;
use tokio::task::JoinHandle;

lazy_static! {
    // -- Core Logging Metrics --
    pub static ref LOGS_PROCESSED: IntCounterVec = IntCounterVec::new(
        Opts::new("trenchbot_logs_processed", "Total logs processed by type"),
        &["log_type"]  // combat, mev, whale, etc.
    ).expect("Failed to create LOGS_PROCESSED metric");

    // -- Performance Metrics --
    pub static ref DETECTION_LATENCY: HistogramVec = HistogramVec::new(
        HistogramOpts::new(
            "trenchbot_detection_latency_seconds",
            "Latency of MEV/whale detection"
        )
        .buckets(vec![0.01, 0.05, 0.1, 0.5, 1.0, 5.0]),
        &["detector_type"]  // mev, whale, sandwich, etc.
    ).expect("Failed to create DETECTION_LATENCY metric");

    // -- Error Metrics --
    pub static ref CRITICAL_FAILURES: IntCounterVec = IntCounterVec::new(
        Opts::new("trenchbot_critical_failures", "Mission-critical failures"),
        &["subsystem", "error_code"]  // rpc, detector, redis, etc.
    ).expect("Failed to create CRITICAL_FAILURES metric");
}

/// Registers all metrics with a custom registry
pub fn register_metrics(registry: &Registry) -> Result<(), prometheus::Error> {
    registry.register(Box::new(LOGS_PROCESSED.clone()))?;
    registry.register(Box::new(DETECTION_LATENCY.clone()))?;
    registry.register(Box::new(CRITICAL_FAILURES.clone()))?;
    Ok(())
}

/// Starts metrics HTTP server on specified port
pub fn start_metrics_server(addr: SocketAddr, registry: Registry) -> JoinHandle<()> {
    tokio::spawn(async move {
        let server = hyper::Server::bind(&addr)
            .serve(hyper::service::make_service_fn(|_| async {
                Ok::<_, hyper::Error>(hyper::service::service_fn(|_| async {
                    let encoder = TextEncoder::new();
                    let mut buffer = vec![];
                    encoder.encode(&registry.gather(), &mut buffer).unwrap();
                    Ok::<_, hyper::Error>(hyper::Response::new(buffer.into()))
                }))
            }));

        if let Err(e) = server.await {
            eprintln!("Metrics server error: {}", e);
        }
    })
}// In src/observability/metrics.rs
pub static CONFIG_RELOADS: IntCounter = register_int_counter!(
    "trenchbot_config_reloads_total",
    "Total config reload attempts"
).unwrap();