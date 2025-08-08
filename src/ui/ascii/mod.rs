use crate::ui::boot::{animation, healthcheck, metadata, matrix};
use crossterm::ExecutableCommand;
// Removed unused import

/// Boot styles available
pub enum BootStyle {
    Default,
    Matrix,
}

/// Entry point
pub async fn show_branding(style: BootStyle) -> anyhow::Result<()> {
    // 1) Pick animation
    match style {
        BootStyle::Matrix => matrix::run_matrix_rain().await?,
        BootStyle::Default => animation::typewriter_banner().await?,
    }

    // 2) Metadata (version, GPUs, VRAM, ping)
    metadata::print_metadata().await?;

    // 3) Health checks with timeouts
    healthcheck::print_healthchecks().await?;

    // 4) Clear after delay
    tokio::time::sleep(std::time::Duration::from_millis(1500)).await;
    stdout().execute(crossterm::terminal::Clear(
        crossterm::terminal::ClearType::All,
    ))?;
    Ok(())
}
