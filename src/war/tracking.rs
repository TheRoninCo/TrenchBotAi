use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recruit {
    pub id: String,
    pub joined_at: DateTime<Utc>,
    pub status: RecruitStatus,
    pub performance_metrics: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RecruitStatus {
    Active,
    Inactive,
    Promoted,
    Dismissed,
}

impl Recruit {
    pub fn new(id: String) -> Self {
        Self {
            id,
            joined_at: Utc::now(),
            status: RecruitStatus::Active,
            performance_metrics: Vec::new(),
        }
    }

    pub fn update_status(&mut self, status: RecruitStatus) {
        self.status = status;
    }
}