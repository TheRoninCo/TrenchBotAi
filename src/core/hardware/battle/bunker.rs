//! BUNKER PROTOCOL - Emergency fallback
pub struct BunkerProtocol {
    rekt_counter: AtomicU32,
    threshold: u32,
    status: AtomicBool,
}

impl BunkerProtocol {
    pub fn new(threshold: u32) -> Self {
        Self {
            rekt_counter: AtomicU32::new(0),
            threshold,
            status: AtomicBool::new(false),
        }
    }

    pub fn record_rekt(&self) -> bool {
        let rekts = self.rekt_counter.fetch_add(1, Ordering::Relaxed) + 1;
        if rekts >= self.threshold {
            self.status.store(true, Ordering::Release);
            true
        } else {
            false
        }
    }
}