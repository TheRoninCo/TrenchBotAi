use crossterm::{QueueableCommand, style::Print};
use figlet_rs::FIGfont;
use std::io::{stdout, Write};
use tokio::time::{sleep, Duration};

/// Prints a FIGlet banner with a Solana purple→teal gradient, shaded blocks,
/// and a typewriter effect.
pub async fn typewriter_banner() -> anyhow::Result<()> {
    // 1) Render text
    let font = FIGfont::standard()
        .map_err(|_| anyhow::anyhow!("Failed to load FIGlet font"))?;
    let figure = font
        .convert("TRENCHBOTAI")
        .ok_or_else(|| anyhow::anyhow!("FIGlet conversion failed"))?;
    let lines: Vec<&str> = figure.lines().collect();

    // 2) Gradient endpoints (purple → teal)
    let start = (153u8,  69u8, 255u8); // #9945FF
    let end   = ( 20u8, 241u8, 149u8); // #14F195

    // 3) Precompute a color per line
    let max_i = (lines.len() - 1).max(1) as f32;
    let colors: Vec<String> = (0..lines.len())
        .map(|i| {
            let t = i as f32 / max_i;
            let r = lerp(start.0, end.0, t);
            let g = lerp(start.1, end.1, t);
            let b = lerp(start.2, end.2, t);
            format!("\x1B[38;2;{};{};{}m", r, g, b)
        })
        .collect();

    // 4) Shading characters, recycled if more lines than shades
    let shades = ['░','▒','▓','█','█'];
    let mut stdout = stdout();

    // 5) Typewriter print each line
    for (i, &line) in lines.iter().enumerate() {
        let shade = shades[i % shades.len()];
        let shaded: String = line
            .chars()
            .map(|c| if c != ' ' { shade } else { ' ' })
            .collect();

        // Set that line’s gradient color
        stdout.queue(Print(colors[i].as_str()))?;

        for ch in shaded.chars() {
            stdout.queue(Print(ch))?;
            stdout.flush()?;                // immediate display
            sleep(Duration::from_millis(3)).await;
        }

        // Reset color and newline
        println!("\x1B[0m");
    }

    Ok(())
}

/// Linear interpolation between two u8 endpoints
fn lerp(a: u8, b: u8, t: f32) -> u8 {
    ((a as f32) + t * ((b as f32) - (a as f32))).round() as u8
}
