use tokio::{process::Command, time::{interval, Duration}};
use std::path::Path;

pub struct WolframTrainer {
    script_path: String,
    output_dir: String,
    interval_hours: u32,
}

impl WolframTrainer {
    pub fn new(script_path: &str, output_dir: &str, interval_hours: u32) -> Self {
        Self {
            script_path: script_path.to_string(),
            output_dir: output_dir.to_string(),
            interval_hours,
        }
    }

    pub async fn start(&self) {
        let mut interval = interval(Duration::from_secs(self.interval_hours * 3600));
        
        loop {
            interval.tick().await;
            self.run_training().await;
        }
    }

    async fn run_training(&self) {
        let output = Command::new("wolframscript")
            .args(&["-file", &self.script_path])
            .output()
            .await
            .expect("Failed to execute WolframScript");

        if !output.status.success() {
            eprintln!("Training failed: {}", String::from_utf8_lossy(&output.stderr));
            return;
        }

        info!("New model saved to {}", self.output_dir);
    }
}