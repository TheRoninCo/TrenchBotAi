use std::time::Instant;
use super::*;

impl TrenchConfig {
    pub fn validate_with_metrics(&self) -> Result<(), ValidationErrors> {
        let timer = Instant::now();
        
        self.validate()?;
        
        // Module-specific validation
        self.mev.validate()?;
        self.sonar.validate()?;
        
        metrics::CONFIG_VALIDATION_TIME.observe(timer.elapsed().as_secs_f64());
        Ok(())
    }
}

impl WhaleConfig {
    pub fn validate_relationships(&self) -> Result<(), ValidationError> {
        if self.gorilla.threshold <= self.kraken.min_volume {
            return Err(ValidationError::new(
                "Gorilla threshold must be greater than Kraken min volume"
            ));
        }
        Ok(())
    }
}