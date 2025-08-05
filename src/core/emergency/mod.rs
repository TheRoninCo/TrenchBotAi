use async_trait::async_trait;
use anyhow::Result;

#[derive(Debug)]
pub enum RecoveryProcedure {
    CloseAllPositions,
    CancelPendingTransactions,
}

#[async_trait]
impl RecoveryProcedure {
    pub async fn execute(&self) -> Result<()> {
        match self {
            Self::CloseAllPositions => {
                tracing::info!("Closing all positions");
                Ok(())
            }
            Self::CancelPendingTransactions => {
                tracing::info!("Cancelling pending transactions");
                Ok(())
            }
        }
    }

    pub fn priority(&self) -> u8 {
        match self {
            Self::CloseAllPositions => 0,
            Self::CancelPendingTransactions => 1,
        }
    }
}

pub struct EmergencySystem {
    procedures: Vec<RecoveryProcedure>,
}

impl EmergencySystem {
    pub fn new() -> Self {
        Self {
            procedures: vec![
                RecoveryProcedure::CloseAllPositions,
                RecoveryProcedure::CancelPendingTransactions,
            ],
        }
    }

    pub async fn execute_recovery(&self) -> Result<()> {
        let mut procedures = self.procedures.clone();
        procedures.sort_by_key(|p| p.priority());
        
        for procedure in procedures {
            procedure.execute().await?;
        }
        Ok(())
    }
}