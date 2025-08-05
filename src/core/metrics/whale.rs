use prometheus::{IntCounterVec, register_int_counter_vec};
use lazy_static::lazy_static;

lazy_static! {
    pub static ref CLASSIFICATIONS: IntCounterVec = register_int_counter_vec!(
        "whale_classifications_total",
        "Count of whale classifications by type",
        &["personality"]
    ).unwrap();

    pub static ref PERSONALITY_CHANGES: IntCounterVec = register_int_counter_vec!(
        "whale_personality_changes_total",
        "Count of personality transitions",
        &["from", "to"]
    ).unwrap();
}

pub fn record_classification(personality: &str) {
    CLASSIFICATIONS.with_label_values(&[personality]).inc();
}

pub fn record_personality_change(from: &str, to: &str) {
    PERSONALITY_CHANGES.with_label_values(&[from, to]).inc();
}