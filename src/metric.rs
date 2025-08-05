use prometheus::{IntCounter, Histogram, Registry};

pub struct MevMetrics {
    pub bundles_received: IntCounter,
    pub s3_upload_latency: Histogram,
    pub health_status: IntCounter,
}

impl MevMetrics {
    pub fn new(registry: &Registry) -> Self {
        let bundles_received = IntCounter::new(
            "mev_bundles_received_total",
            "Total bundles processed"
        ).unwrap();
        
        let s3_upload_latency = Histogram::with_opts(
            HistogramOpts::new(
                "mev_s3_upload_latency_seconds",
                "S3 batch upload times"
            ).buckets(vec![0.1, 0.5, 1.0, 2.5, 5.0])
        ).unwrap();

        registry.register(Box::new(bundles_received.clone())).unwrap();
        registry.register(Box::new(s3_upload_latency.clone())).unwrap();

        Self {
            bundles_received,
            s3_upload_latency,
        }
    }
}