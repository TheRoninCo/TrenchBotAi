//! Integration test for RuneScape-themed ranking system

use chrono::{Utc, Duration};
use trenchbot_dex::strategies::counter_rug_pull::{CounterRugOperation, TradingPhase, ExitTrigger, ClusterIntel};
use trenchbot_dex::war::runescape_rankings::{RuneScapeRankingSystem, MonsterKill, CombatSkill};

#[tokio::test]
async fn test_runescape_integration_full_progression() {
    let mut ranking = RuneScapeRankingSystem::new("TestWarrior".to_string());
    
    println!("🏰 Starting RuneScape Integration Test");
    println!("{}", ranking.get_stats_display());
    
    // Test different monster kills with increasing difficulty
    let test_operations = vec![
        // Goblin kill (small profit)
        (5.0, 2, "Goblin kill - learning the ropes"),
        // Orc kill (medium profit) 
        (25.0, 4, "Orc kill - getting stronger"),
        // Dragon kill (big profit)
        (150.0, 3, "Dragon kill - major victory!"),
        // Barrows (coordinated cluster)
        (75.0, 8, "Barrows Brothers - coordinated attack defeated"),
        // King Black Dragon (mega whale)
        (600.0, 5, "King Black Dragon - legendary whale slain"),
    ];
    
    for (i, (profit, wallets, description)) in test_operations.into_iter().enumerate() {
        println!("\n--- {} ---", description);
        
        let operation = CounterRugOperation {
            operation_id: format!("test_op_{}", i),
            token_mint: format!("test_token_{}", i),
            phase: TradingPhase::PostMortem,
            entry_time: Utc::now() - Duration::hours(1),
            entry_price: 1.0,
            position_size: profit * 0.5, // Reasonable position size
            current_price: 1.0 + (profit / 100.0), // Calculate exit price
            unrealized_pnl: profit,
            target_exit_time: Utc::now() + Duration::minutes(30),
            risk_score: 0.8,
            cluster_info: ClusterIntel {
                cluster_count: 1,
                total_wallets: wallets,
                coordination_score: 0.8,
                estimated_total_supply: 1_000_000.0,
                cluster_accumulation_rate: 0.0,
                time_to_estimated_dump: None,
            },
            exit_triggers: vec![ExitTrigger::ProfitTarget],
        };
        
        // Record the kill
        let notifications = ranking.record_kill(&operation, profit).await.unwrap();
        
        // Display notifications
        for notification in notifications {
            println!("📢 {}", notification);
        }
        
        // Show updated stats every few kills
        if i % 2 == 1 {
            println!("{}", ranking.get_stats_display());
        }
    }
    
    println!("\n🎊 FINAL WARRIOR STATS 🎊");
    println!("{}", ranking.get_stats_display());
    
    println!("\n📜 KILL LOG (Last 5):");
    let kill_log = ranking.get_kill_log(5);
    for (i, kill) in kill_log.iter().enumerate() {
        println!("{}. {}", i + 1, kill);
    }
    
    // Verify progression
    let rug_pull_skill = &ranking.player_stats.skills[&CombatSkill::RugPulling];
    assert!(rug_pull_skill.level > 1, "Should have gained levels");
    assert!(rug_pull_skill.xp > 0, "Should have gained XP");
    assert!(ranking.player_stats.bank_value_gp > 0, "Should have earned GP");
    assert!(!ranking.player_stats.special_attacks_unlocked.is_empty(), "Should have unlocked special attacks");
    
    println!("\n✅ Integration test passed! Warrior has progressed successfully.");
}

#[tokio::test]
async fn test_monster_classification_accuracy() {
    println!("🗡️ Testing Monster Classification");
    
    let test_cases = vec![
        (5.0, 2, MonsterKill::Goblin, "Small rug pull"),
        (25.0, 3, MonsterKill::Orc, "Medium rug pull"),
        (75.0, 4, MonsterKill::Troll, "Large rug pull"),
        (150.0, 3, MonsterKill::Dragon, "Whale kill"),
        (600.0, 5, MonsterKill::KingBlackDragon, "Mega whale"),
        (50.0, 8, MonsterKill::Barrows, "Coordinated cluster"),
        (100.0, 25, MonsterKill::GodWars, "Major coordinated attack"),
    ];
    
    for (profit, wallets, expected_monster, description) in test_cases {
        let operation = CounterRugOperation {
            operation_id: "test".to_string(),
            token_mint: "test_token".to_string(),
            phase: TradingPhase::PostMortem,
            entry_time: Utc::now(),
            entry_price: 1.0,
            position_size: 100.0,
            current_price: 1.0 + (profit / 100.0),
            unrealized_pnl: profit,
            target_exit_time: Utc::now(),
            risk_score: 0.8,
            cluster_info: ClusterIntel {
                cluster_count: 1,
                total_wallets: wallets,
                coordination_score: 0.8,
                estimated_total_supply: 1_000_000.0,
                cluster_accumulation_rate: 0.0,
                time_to_estimated_dump: None,
            },
            exit_triggers: vec![],
        };
        
        let monster = MonsterKill::from_operation(&operation, profit);
        println!("{} {} -> {} {} (Expected: {} {})", 
            description,
            profit,
            monster.emoji(),
            monster.name(),
            expected_monster.emoji(),
            expected_monster.name()
        );
        
        assert_eq!(monster, expected_monster, "Monster classification mismatch for {}", description);
    }
    
    println!("✅ All monster classifications correct!");
}

#[test]
fn test_combat_level_progression() {
    println!("⚔️ Testing Combat Level System");
    
    use trenchbot_dex::war::runescape_rankings::CombatLevel;
    
    let test_levels = vec![
        (1, "1", "🤎"),
        (15, "15", "⚪"), 
        (35, "35", "🟫"),
        (55, "55", "🟢"),
        (75, "75", "🔴"),
        (90, "90", "🟡"),
        (99, "99", "🟡"),
        (150, "99★1", "💎"), // Prestige level
    ];
    
    for (level, expected_display, expected_color) in test_levels {
        let combat_level = CombatLevel::new(level);
        println!("Level {}: {} {} (Color: {})", 
            level, 
            combat_level.display(), 
            expected_display,
            combat_level.color()
        );
        
        assert_eq!(combat_level.display(), expected_display);
        assert_eq!(combat_level.color(), expected_color);
    }
    
    println!("✅ Combat level progression working correctly!");
}