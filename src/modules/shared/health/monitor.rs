//! Sink health tracking with retries
use std::time::{Duration, Instant};
use tokio::sync::watch;

pub struct SinkHealth {
    last_ok: watch::Receiver<Instant>,
}

impl SinkHealth {
    pub fn new() -> (Self, HealthUpdater) {
        let (tx, rx) = watch::channel(Instant::now());
        (Self { last_ok: rx }, HealthUpdater(tx))
    }

    pub fn is_healthy(&self) -> bool {
        self.last_ok.borrow().elapsed() < Duration::from_secs(30)
    }
}

pub struct HealthUpdater(watch::Sender<Instant>);

impl HealthUpdater {
    pub fn ping(&self) {
        self.0.send(Instant::now()).ok();
    }
}