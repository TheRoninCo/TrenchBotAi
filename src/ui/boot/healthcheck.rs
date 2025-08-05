use crossterm::style::Stylize;
use tokio::time::{timeout, Duration};

pub async fn print_healthchecks() -> anyhow::Result<()> {
    let redis_ok = timed_check(check_redis()).await;
    let helius_ok = timed_check(check_helius()).await;
    let jup_ok   = timed_check(check_jupiter()).await;

    println!(
        "[Redis {}] [Helius {}] [JUP-API {}]",
        mark(redis_ok),
        mark(helius_ok),
        mark(jup_ok)
    );
    Ok(())
}

async fn timed_check<F>(fut: F) -> bool
where
    F: std::future::Future<Output = bool>,
{
    match timeout(Duration::from_millis(500), fut).await {
        Ok(ok) => ok,
        Err(_) => false,
    }
}

// Replace these with your real probes:
async fn check_redis() -> bool {
    // e.g., ping Redis client
    true
}
async fn check_helius() -> bool {
    true
}
async fn check_jupiter() -> bool {
    true
}

fn mark(ok: bool) -> String {
    if ok {
        "✔".green().to_string()
    } else {
        "✘".red().to_string()
    }
}
