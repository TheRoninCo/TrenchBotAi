pub struct Recruit {
    pub id: String,
    pub tier_upgraded: bool,
    pub total_commission: f64,
    pub last_activity: DateTime<Utc> // Anti-sybil
}