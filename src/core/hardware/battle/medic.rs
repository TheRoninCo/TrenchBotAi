//! FIELD MEDICS - Error recovery squad
pub struct Medic {
    revival_attempts: AtomicU32,
}

impl Medic {
    pub fn revive<F, T>(&self, operation: F) -> Result<T>
    where F: Fn() -> Result<T> 
    {
        match operation() {
            Ok(val) => Ok(val),
            Err(e) if self.revival_attempts.fetch_add(1, Ordering::Relaxed) < 3 => {
                log::warn!("Medic reviving wounded op: {}", e);
                operation()
            }
            Err(e) => Err(e)
        }
    }
}