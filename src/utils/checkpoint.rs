use std::path::Path;
use tokio::fs;

pub async fn save_checkpoint(
    data: &[u8],
    epoch: u32,
    run_id: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let path = format!("/persistent/checkpoints/{}-{}.bin", run_id, epoch);
    fs::write(Path::new(&path), data).await?;
    Ok(())
}

pub async fn load_latest_checkpoint(
    run_id: &str,
) -> Option<Vec<u8>> {
    let pattern = format!("/persistent/checkpoints/{}*.bin", run_id);
    let latest = glob::glob(&pattern)
        .ok()?
        .filter_map(|p| p.ok())
        .max()?;
    
    fs::read(latest).await.ok()
}