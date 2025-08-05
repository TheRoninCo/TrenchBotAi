use crossterm::{cursor, style::Print, terminal, QueueableCommand};
use rand::{thread_rng, Rng};
use std::io::{stdout, Write};
use tokio::time::{sleep, Duration};

pub async fn run_matrix_rain() -> anyhow::Result<()> {
    let (cols, rows) = terminal::size()?;
    let mut ypos: Vec<u16> = (0..cols).map(|_| thread_rng().gen_range(0..rows)).collect();

    for _ in 0..rows * 2 {
        for (x, y) in ypos.iter_mut().enumerate() {
            stdout()
                .queue(cursor::MoveTo(x as u16, *y))?
                .queue(Print("\x1B[38;5;46m".to_string()))?  // bright green
                .unwrap();
            let c = thread_rng().gen_range(33u8..126u8) as char;
            stdout().queue(Print(c))?;
            *y = if *y >= rows { 0 } else { *y + 1 };
        }
        stdout().flush()?;
        sleep(Duration::from_millis(50)).await;
    }
    Ok(())
}
