use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecruitmentRebate {
    pub amount: f64,
    pub percentage: f64,
    pub conditions: Vec<String>,
}

impl RecruitmentRebate {
    pub fn new(amount: f64, percentage: f64) -> Self {
        Self {
            amount,
            percentage,
            conditions: Vec::new(),
        }
    }
}