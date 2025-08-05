// tests/classifiers/ration_test.rs
use crate::modules::shared::classifiers::CombatRation;

#[test]
fn test_profit_classification() {
    assert_eq!(CombatRation::classify(50.0), CombatRation::KetchupPacket);
    assert_eq!(CombatRation::classify(300.0), CombatRation::ShrimpSnack);
    assert_eq!(CombatRation::classify(1_500.0), CombatRation::HappyMeal);
    assert_eq!(CombatRation::classify(10_000.0), CombatRation::FieldSteak);
    assert_eq!(CombatRation::classify(75_000.0), CombatRation::AdmiralsFeast);
    assert_eq!(CombatRation::classify(1_000_000.0), CombatRation::KrakenHarvest);
}

#[test]
fn test_label_conversion() {
    assert_eq!(CombatRation::HappyMeal.label(), "Happy Meal");
    assert_eq!(<&str>::from(CombatRation::KrakenHarvest), "Kraken Harvest");
}