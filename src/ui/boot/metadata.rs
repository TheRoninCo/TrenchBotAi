use crossterm::style::Stylize;
use sysinfo::System;
use tokio::time::Instant;

/// Prints version + either GPU info or CPU info, plus ping.
pub async fn print_metadata() -> anyhow::Result<()> {
    let version = env!("CARGO_PKG_VERSION");
    let (label, spec) = hardware_spec();
    let ping_ms = ping_helius().await;

    println!(
        "{} | {} | {}",
        format!("v{}", version).bold(),
        format!("{}: {}", label, spec).underlined(),
        format!("{}ms PING", ping_ms).italic()
    );
    Ok(())
}

fn hardware_spec() -> (&'static str, String) {
    let mut sys = System::new_all();
    sys.refresh_all();

    // Try GPUs first (requires "cuda" feature)
    if let Some(gpus) = sys.cuda_memory() {
        if !gpus.is_empty() {
            let gpu_count = gpus.len();
            let total_bytes: u64 = gpus.iter().map(|m| m.total_memory).sum();
            let total_gb = total_bytes / (1024 * 1024 * 1024);
            return (
                "GPUs",
                format!("{} × {} GB VRAM", gpu_count, total_gb),
            );
        }
    }

    // Fallback to CPU info
    let cpu_count = sys.cpus().len();
    let total_mem_gb = sys.total_memory() / 1024 / 1024;
    (
        "CPU",
        format!("{} cores, {} GB RAM", cpu_count, total_mem_gb),
    )
}

async fn ping_helius() -> u128 {
    let start = Instant::now();
    let _ = reqwest::Client::new()
        .head("https://api.helius.xyz")
        .send()
        .await;
    start.elapsed().as_millis()
}
