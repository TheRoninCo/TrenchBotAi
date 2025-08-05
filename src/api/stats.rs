// src/api/stats.rs
use axum::Json;
use trenchie_core::killfeed::Killfeed;

pub async fn get_stats(killfeed: Killfeed) -> Json<serde_json::Value> {
    let snapshot = killfeed.snapshot();
    Json(serde_json::json!({
        "top_performers": snapshot.top_performers,
        "total_profit": snapshot.total_profit,
        "latency_p99_ns": snapshot.latency.p99,
    }))
}