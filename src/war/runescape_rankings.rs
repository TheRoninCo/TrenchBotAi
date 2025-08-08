//! RuneScape-Themed Warfare Ranking System
//! 
//! Experience-based progression, combat levels, special attacks, and legendary drops
//! for successful anti-rug-pull operations. Just like the good old days of Lumbridge.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc, Duration};
use std::collections::{HashMap, BTreeMap};

use crate::analytics::RugPullAlert;
use crate::strategies::counter_rug_pull::{CounterRugOperation, WarfareStats, ExitTrigger};

/// Combat levels (1-99, then prestige levels)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct CombatLevel {
    pub level: u32,
    pub prestige: u32, // For levels beyond 99
}

impl CombatLevel {
    pub fn new(level: u32) -> Self {
        if level <= 99 {
            Self { level, prestige: 0 }
        } else {
            Self { 
                level: 99, 
                prestige: level / 99 
            }
        }
    }

    pub fn effective_level(&self) -> u32 {
        self.level + (self.prestige * 99)
    }

    pub fn display(&self) -> String {
        if self.prestige == 0 {
            format!("{}", self.level)
        } else {
            format!("{}★{}", self.level, self.prestige)
        }
    }

    pub fn color(&self) -> &'static str {
        match self.level {
            1..=9 => "🤎",      // Bronze
            10..=19 => "⚪",     // Iron  
            20..=29 => "⚫",     // Steel
            30..=39 => "🟫",     // Black
            40..=49 => "🔵",     // Mithril
            50..=59 => "🟢",     // Adamant
            60..=79 => "🔴",     // Rune
            80..=99 => "🟡",     // Dragon
            _ => "💎",          // Prestige
        }
    }
}

/// Different combat skills we track (RuneScape style)
#[derive(Debug, Clone, Serialize, Deserialize, Hash, PartialEq, Eq)]
pub enum CombatSkill {
    RugPulling,      // Main combat skill - detecting and countering rug pulls
    Whaling,         // Hunting large targets
    Coordination,    // Detecting coordinated attacks  
    Timing,          // Perfect entry/exit timing
    RiskManagement,  // Avoiding losses
    Profiteering,    // Maximizing gains
    Scouting,        // Information gathering
    Persistence,     // Long-term consistency
}

impl CombatSkill {
    pub fn emoji(&self) -> &'static str {
        match self {
            CombatSkill::RugPulling => "🗡️",
            CombatSkill::Whaling => "🔱", 
            CombatSkill::Coordination => "🛡️",
            CombatSkill::Timing => "🏹",
            CombatSkill::RiskManagement => "🧙‍♂️",
            CombatSkill::Profiteering => "💰",
            CombatSkill::Scouting => "👁️",
            CombatSkill::Persistence => "⚔️",
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            CombatSkill::RugPulling => "Rug Pulling",
            CombatSkill::Whaling => "Whaling",
            CombatSkill::Coordination => "Coordination",
            CombatSkill::Timing => "Timing",
            CombatSkill::RiskManagement => "Risk Management", 
            CombatSkill::Profiteering => "Profiteering",
            CombatSkill::Scouting => "Scouting",
            CombatSkill::Persistence => "Persistence",
        }
    }
}

/// Special attacks that can be unlocked
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpecialAttack {
    DragonDagger,    // Quick 2x damage (fast small trades)
    DragonClaws,     // 4x combo attack (multi-position)
    AGS,            // Armor Godsword - massive single hit
    DDS,            // Dragon Dagger Super Poison (toxic to scammers)
    Whip,           // Consistent DPS (steady profits)
    Dharoks,        // More damage when low HP (risky comeback)
}

impl SpecialAttack {
    pub fn emoji(&self) -> &'static str {
        match self {
            SpecialAttack::DragonDagger => "🗡️",
            SpecialAttack::DragonClaws => "🦅",
            SpecialAttack::AGS => "⚔️",
            SpecialAttack::DDS => "💚",
            SpecialAttack::Whip => "🪢",
            SpecialAttack::Dharoks => "🩸",
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            SpecialAttack::DragonDagger => "Dragon Dagger",
            SpecialAttack::DragonClaws => "Dragon Claws", 
            SpecialAttack::AGS => "Armadyl Godsword",
            SpecialAttack::DDS => "Dragon Dagger (Super)",
            SpecialAttack::Whip => "Abyssal Whip",
            SpecialAttack::Dharoks => "Dharok's Greataxe",
        }
    }

    pub fn unlock_level(&self) -> u32 {
        match self {
            SpecialAttack::DragonDagger => 20,
            SpecialAttack::DDS => 35, 
            SpecialAttack::Whip => 50,
            SpecialAttack::DragonClaws => 65,
            SpecialAttack::Dharoks => 75,
            SpecialAttack::AGS => 85,
        }
    }
}

/// Rare drops from successful kills
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RareDrop {
    pub item_name: String,
    pub emoji: String,
    pub rarity: DropRarity,
    pub value_gp: u64, // In "GP" (SOL equivalent)
    pub dropped_from: String, // Which operation
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DropRarity {
    Common,      // 1/10
    Uncommon,    // 1/50
    Rare,        // 1/200
    VeryRare,    // 1/1000
    UltraRare,   // 1/5000
    Legendary,   // 1/50000 (Third Age)
}

impl DropRarity {
    pub fn color(&self) -> &'static str {
        match self {
            DropRarity::Common => "⚪",
            DropRarity::Uncommon => "🟢", 
            DropRarity::Rare => "🔵",
            DropRarity::VeryRare => "🟣",
            DropRarity::UltraRare => "🟡",
            DropRarity::Legendary => "🔴",
        }
    }

    pub fn drop_rate(&self) -> u32 {
        match self {
            DropRarity::Common => 10,
            DropRarity::Uncommon => 50,
            DropRarity::Rare => 200,
            DropRarity::VeryRare => 1000,
            DropRarity::UltraRare => 5000,
            DropRarity::Legendary => 50000,
        }
    }
}

/// Player stats (RuneScape style skill levels)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlayerStats {
    pub combat_level: CombatLevel,
    pub total_level: u32,
    pub skills: HashMap<CombatSkill, SkillData>,
    pub special_attacks_unlocked: Vec<SpecialAttack>,
    pub bank_value_gp: u64,
    pub rare_drops: Vec<RareDrop>,
    pub total_xp: u64,
    pub quest_points: u32, // Achievements completed
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillData {
    pub level: u32,
    pub xp: u64,
    pub xp_to_next: u64,
}

impl SkillData {
    pub fn new() -> Self {
        Self {
            level: 1,
            xp: 0,
            xp_to_next: 83, // XP needed for level 2
        }
    }

    /// Calculate level from XP (RuneScape XP table)
    pub fn calculate_level(xp: u64) -> u32 {
        if xp == 0 { return 1; }
        
        let mut level = 1u32;
        let mut total_xp = 0u64;
        
        while level < 99 {
            let xp_for_next = (level as f64 * 0.25 * (1.0 + (level as f64 / 300.0).powi(2))).floor() as u64;
            if total_xp + xp_for_next > xp {
                break;
            }
            total_xp += xp_for_next;
            level += 1;
        }
        
        level
    }

    pub fn add_xp(&mut self, xp: u64) {
        self.xp += xp;
        let new_level = Self::calculate_level(self.xp);
        
        if new_level > self.level {
            self.level = new_level;
        }
        
        // Calculate XP to next level
        if self.level < 99 {
            let xp_for_current = Self::xp_for_level(self.level);
            let xp_for_next = Self::xp_for_level(self.level + 1);
            self.xp_to_next = xp_for_next - self.xp;
        } else {
            self.xp_to_next = 0;
        }
    }

    fn xp_for_level(level: u32) -> u64 {
        if level <= 1 { return 0; }
        
        let mut total = 0u64;
        for lvl in 1..level {
            total += (lvl as f64 * 0.25 * (1.0 + (lvl as f64 / 300.0).powi(2))).floor() as u64;
        }
        total
    }
}

/// Kill types with RuneScape monster themes
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MonsterKill {
    Goblin,          // Small rug pull (< 10 SOL)
    Orc,             // Medium rug pull (10-50 SOL)
    Troll,           // Large rug pull (50-100 SOL) 
    Dragon,          // Whale kill (100-500 SOL)
    KingBlackDragon, // Mega whale (500+ SOL)
    Barrows,         // Coordinated cluster (6+ wallets)
    GodWars,         // Major coordinated attack (20+ wallets)
    Jad,             // Perfect execution under pressure
    Zulrah,          // Multiple kill phases
    Vorkath,         // Solo boss (single large target)
}

impl MonsterKill {
    pub fn from_operation(operation: &CounterRugOperation, profit: f64) -> Self {
        // Determine monster type based on operation characteristics
        if operation.cluster_info.total_wallets >= 20 {
            MonsterKill::GodWars
        } else if operation.cluster_info.total_wallets >= 6 {
            MonsterKill::Barrows  
        } else if profit >= 500.0 {
            MonsterKill::KingBlackDragon
        } else if profit >= 100.0 {
            MonsterKill::Dragon
        } else if profit >= 50.0 {
            MonsterKill::Troll
        } else if profit >= 10.0 {
            MonsterKill::Orc
        } else {
            MonsterKill::Goblin
        }
    }

    pub fn emoji(&self) -> &'static str {
        match self {
            MonsterKill::Goblin => "👹",
            MonsterKill::Orc => "🧌",
            MonsterKill::Troll => "🗿",
            MonsterKill::Dragon => "🐲",
            MonsterKill::KingBlackDragon => "🖤",
            MonsterKill::Barrows => "⚰️",
            MonsterKill::GodWars => "⚡",
            MonsterKill::Jad => "🔥",
            MonsterKill::Zulrah => "🐍",
            MonsterKill::Vorkath => "💀",
        }
    }

    pub fn xp_reward(&self) -> u64 {
        match self {
            MonsterKill::Goblin => 50,
            MonsterKill::Orc => 100,
            MonsterKill::Troll => 250,
            MonsterKill::Dragon => 500,
            MonsterKill::KingBlackDragon => 1000,
            MonsterKill::Barrows => 750,
            MonsterKill::GodWars => 1500,
            MonsterKill::Jad => 2000,
            MonsterKill::Zulrah => 1250,
            MonsterKill::Vorkath => 1750,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            MonsterKill::Goblin => "Goblin",
            MonsterKill::Orc => "Orc",
            MonsterKill::Troll => "Troll",
            MonsterKill::Dragon => "Dragon",
            MonsterKill::KingBlackDragon => "King Black Dragon",
            MonsterKill::Barrows => "Barrows Brothers",
            MonsterKill::GodWars => "God Wars Dungeon",
            MonsterKill::Jad => "TzTok-Jad",
            MonsterKill::Zulrah => "Zulrah",
            MonsterKill::Vorkath => "Vorkath",
        }
    }
}

/// Main RuneScape-themed ranking system
pub struct RuneScapeRankingSystem {
    pub player_stats: PlayerStats,
    pub kill_log: Vec<KillEntry>,
    pub daily_challenges: HashMap<String, DailyChallenge>,
    pub high_scores_rank: Option<u32>,
    pub clan_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KillEntry {
    pub timestamp: DateTime<Utc>,
    pub monster: MonsterKill,
    pub loot_value: u64,
    pub xp_gained: HashMap<CombatSkill, u64>,
    pub rare_drops: Vec<RareDrop>,
    pub combat_duration: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DailyChallenge {
    pub description: String,
    pub target: u32,
    pub progress: u32,
    pub reward_xp: u64,
    pub completed: bool,
}

impl RuneScapeRankingSystem {
    pub fn new(player_name: String) -> Self {
        let mut skills = HashMap::new();
        
        // Initialize all skills at level 1
        for skill in [
            CombatSkill::RugPulling,
            CombatSkill::Whaling, 
            CombatSkill::Coordination,
            CombatSkill::Timing,
            CombatSkill::RiskManagement,
            CombatSkill::Profiteering,
            CombatSkill::Scouting,
            CombatSkill::Persistence,
        ] {
            skills.insert(skill, SkillData::new());
        }

        Self {
            player_stats: PlayerStats {
                combat_level: CombatLevel::new(3), // Start at combat level 3 like RuneScape
                total_level: 8, // 8 skills at level 1 
                skills,
                special_attacks_unlocked: vec![],
                bank_value_gp: 0,
                rare_drops: vec![],
                total_xp: 0,
                quest_points: 0,
            },
            kill_log: vec![],
            daily_challenges: HashMap::new(),
            high_scores_rank: None,
            clan_name: format!("{} Rug Slayers", player_name),
        }
    }

    /// Record a monster kill and gain XP
    pub async fn record_kill(&mut self, operation: &CounterRugOperation, profit: f64) -> Result<Vec<String>> {
        let mut notifications = Vec::new();
        
        let monster = MonsterKill::from_operation(operation, profit);
        let base_xp = monster.xp_reward();
        
        // Calculate XP gains for different skills
        let mut xp_gains = HashMap::new();
        
        // Main rug pulling XP
        xp_gains.insert(CombatSkill::RugPulling, base_xp);
        
        // Bonus XP based on operation characteristics
        if profit >= 100.0 {
            xp_gains.insert(CombatSkill::Whaling, base_xp / 2);
        }
        if operation.cluster_info.total_wallets >= 5 {
            xp_gains.insert(CombatSkill::Coordination, base_xp / 3);
        }
        if matches!(operation.exit_triggers.first(), Some(ExitTrigger::ProfitTarget)) {
            xp_gains.insert(CombatSkill::Timing, base_xp / 4);
        }
        if profit > 0.0 {
            xp_gains.insert(CombatSkill::Profiteering, (profit * 10.0) as u64);
        }

        // Apply XP gains and check for level ups
        let mut level_ups = Vec::new();
        for (skill, xp) in &xp_gains {
            let old_level = self.player_stats.skills[skill].level;
            self.player_stats.skills.get_mut(skill).unwrap().add_xp(*xp);
            let new_level = self.player_stats.skills[skill].level;
            
            if new_level > old_level {
                level_ups.push((skill.clone(), new_level));
                notifications.push(format!(
                    "🎉 {} Level Up! {} {} is now level {}!",
                    skill.emoji(),
                    skill.name(),
                    skill.emoji(), 
                    new_level
                ));
                
                // Check for special attack unlocks
                if *skill == CombatSkill::RugPulling {
                    if let Some(spec_attack) = self.check_special_attack_unlock(new_level) {
                        notifications.push(format!(
                            "⚔️ SPECIAL ATTACK UNLOCKED: {} {}!",
                            spec_attack.emoji(),
                            spec_attack.name()
                        ));
                        self.player_stats.special_attacks_unlocked.push(spec_attack);
                    }
                }
            }
        }

        // Update total XP and combat level
        self.player_stats.total_xp += xp_gains.values().sum::<u64>();
        let rug_pull_level = self.player_stats.skills[&CombatSkill::RugPulling].level;
        self.player_stats.combat_level = CombatLevel::new(rug_pull_level);

        // Check for rare drops
        let rare_drops = self.roll_for_rare_drops(&monster, profit).await?;
        
        // Bank the loot
        let loot_value = (profit * 1000.0) as u64; // Convert SOL to "GP"
        self.player_stats.bank_value_gp += loot_value;

        let kill_entry = KillEntry {
            timestamp: Utc::now(),
            monster: monster.clone(),
            loot_value,
            xp_gained: xp_gains.clone(),
            rare_drops: rare_drops.clone(),
            combat_duration: Utc::now() - operation.entry_time,
        };

        self.kill_log.push(kill_entry);

        // Generate kill notification
        let kill_msg = format!(
            "{} {} DEFEATED! {} GP loot | {} {} XP | Combat Lvl: {}",
            monster.emoji(),
            monster.name(),
            loot_value,
            xp_gains.get(&CombatSkill::RugPulling).unwrap_or(&0),
            CombatSkill::RugPulling.emoji(),
            self.player_stats.combat_level.display()
        );
        notifications.insert(0, kill_msg);

        // Add rare drop notifications
        for drop in rare_drops {
            notifications.push(format!(
                "💎 RARE DROP: {} {} {} (1/{})", 
                drop.rarity.color(),
                drop.emoji,
                drop.item_name,
                drop.rarity.drop_rate()
            ));
        }

        Ok(notifications)
    }

    fn check_special_attack_unlock(&self, level: u32) -> Option<SpecialAttack> {
        let attacks = [
            SpecialAttack::DragonDagger,
            SpecialAttack::DDS, 
            SpecialAttack::Whip,
            SpecialAttack::DragonClaws,
            SpecialAttack::Dharoks,
            SpecialAttack::AGS,
        ];

        for attack in attacks {
            if level >= attack.unlock_level() && !self.player_stats.special_attacks_unlocked.contains(&attack) {
                return Some(attack);
            }
        }
        None
    }

    async fn roll_for_rare_drops(&mut self, monster: &MonsterKill, profit: f64) -> Result<Vec<RareDrop>> {
        let mut drops = Vec::new();
        
        // Roll for different rarities (simplified random for demo)
        let roll = fastrand::u32(1..=50000);
        
        let (rarity, item_name, emoji) = if roll == 1 {
            (DropRarity::Legendary, "3rd Age Scammer Detector", "👑")
        } else if roll <= 10 {
            (DropRarity::UltraRare, "Dragon Rug Pull Slayer", "🗡️")
        } else if roll <= 100 {
            (DropRarity::VeryRare, "Whip of Profit Extraction", "🪢")
        } else if roll <= 500 {
            (DropRarity::Rare, "Rune Scammer Bane", "⚔️")
        } else if roll <= 2000 {
            (DropRarity::Uncommon, "Mithril Anti-Rug Armor", "🛡️")
        } else if roll <= 10000 {
            (DropRarity::Common, "Steel Scammer Detector", "🔍")
        } else {
            return Ok(drops); // No drop
        };

        let drop = RareDrop {
            item_name: item_name.to_string(),
            emoji: emoji.to_string(), 
            rarity,
            value_gp: (profit * 100.0) as u64,
            dropped_from: monster.name().to_string(),
            timestamp: Utc::now(),
        };

        drops.push(drop);
        Ok(drops)
    }

    /// Get player stats display (like examining a player)
    pub fn get_stats_display(&self) -> String {
        let mut stats = format!(
            "\n🏰 {} - Combat Level: {} {}\n",
            self.clan_name,
            self.player_stats.combat_level.color(),
            self.player_stats.combat_level.display()
        );

        stats.push_str("📊 SKILL LEVELS:\n");
        for (skill, data) in &self.player_stats.skills {
            stats.push_str(&format!(
                "  {} {} {}: {} ({} XP)\n",
                skill.emoji(),
                data.level,
                skill.name(),
                data.level,
                data.xp
            ));
        }

        stats.push_str(&format!(
            "\n💰 Bank Value: {} GP\n",
            self.player_stats.bank_value_gp
        ));

        stats.push_str("⚔️ SPECIAL ATTACKS:\n");
        for spec in &self.player_stats.special_attacks_unlocked {
            stats.push_str(&format!("  {} {}\n", spec.emoji(), spec.name()));
        }

        if !self.player_stats.rare_drops.is_empty() {
            stats.push_str("\n💎 RARE DROPS:\n");
            for drop in self.player_stats.rare_drops.iter().rev().take(5) {
                stats.push_str(&format!(
                    "  {} {} {} ({})\n",
                    drop.rarity.color(),
                    drop.emoji,
                    drop.item_name,
                    drop.timestamp.format("%Y-%m-%d")
                ));
            }
        }

        stats
    }

    /// Get recent kills (like a combat log)
    pub fn get_kill_log(&self, limit: usize) -> Vec<String> {
        self.kill_log
            .iter()
            .rev()
            .take(limit)
            .map(|kill| {
                format!(
                    "{} {} | {} GP | {} XP | {}",
                    kill.monster.emoji(),
                    kill.monster.name(),
                    kill.loot_value,
                    kill.xp_gained.get(&CombatSkill::RugPulling).unwrap_or(&0),
                    kill.timestamp.format("%H:%M:%S")
                )
            })
            .collect()
    }

    /// Generate daily challenges (like daily tasks)
    pub fn generate_daily_challenges(&mut self) {
        let today = Utc::now().date_naive().format("%Y-%m-%d").to_string();
        
        if self.daily_challenges.contains_key(&today) {
            return; // Already generated today's challenges
        }

        let challenges = vec![
            DailyChallenge {
                description: "Slay 3 Goblins (small rug pulls)".to_string(),
                target: 3,
                progress: 0,
                reward_xp: 500,
                completed: false,
            },
            DailyChallenge {
                description: "Defeat a Dragon (100+ SOL profit)".to_string(),
                target: 1,
                progress: 0,
                reward_xp: 1000,
                completed: false,
            },
            DailyChallenge {
                description: "Earn 10,000 GP in loot".to_string(),
                target: 10000,
                progress: 0,
                reward_xp: 750,
                completed: false,
            },
        ];

        for (i, challenge) in challenges.into_iter().enumerate() {
            self.daily_challenges.insert(format!("{}_challenge_{}", today, i), challenge);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategies::counter_rug_pull::{ClusterIntel, TradingPhase};

    #[tokio::test]
    async fn test_runescape_progression() {
        let mut ranking = RuneScapeRankingSystem::new("TestPlayer".to_string());
        
        assert_eq!(ranking.player_stats.combat_level.level, 3);
        assert_eq!(ranking.player_stats.skills[&CombatSkill::RugPulling].level, 1);
        
        // Simulate killing a dragon (big profit)
        let operation = CounterRugOperation {
            operation_id: "dragon_test".to_string(),
            token_mint: "dragon_token".to_string(),
            phase: TradingPhase::PostMortem,
            entry_time: Utc::now() - Duration::hours(1),
            entry_price: 1.0,
            position_size: 1000.0,
            current_price: 1.15,
            unrealized_pnl: 150.0,
            target_exit_time: Utc::now(),
            risk_score: 0.9,
            cluster_info: ClusterIntel {
                cluster_count: 1,
                total_wallets: 3,
                coordination_score: 0.9,
                estimated_total_supply: 1_000_000.0,
                cluster_accumulation_rate: 0.0,
                time_to_estimated_dump: None,
            },
            exit_triggers: vec![ExitTrigger::ProfitTarget],
        };

        let notifications = ranking.record_kill(&operation, 150.0).await.unwrap();
        
        assert!(!notifications.is_empty());
        assert!(notifications[0].contains("Dragon"));
        assert!(ranking.player_stats.bank_value_gp > 0);
        assert!(ranking.player_stats.total_xp > 0);
        
        println!("{}", ranking.get_stats_display());
        
        let kill_log = ranking.get_kill_log(5);
        assert!(!kill_log.is_empty());
        println!("Kill log: {:#?}", kill_log);
    }

    #[test]
    fn test_monster_classification() {
        let whale_op = CounterRugOperation {
            operation_id: "whale".to_string(),
            token_mint: "whale_token".to_string(), 
            phase: TradingPhase::PostMortem,
            entry_time: Utc::now(),
            entry_price: 1.0,
            position_size: 1000.0,
            current_price: 1.50,
            unrealized_pnl: 500.0,
            target_exit_time: Utc::now(),
            risk_score: 0.9,
            cluster_info: ClusterIntel {
                cluster_count: 1,
                total_wallets: 3,
                coordination_score: 0.9,
                estimated_total_supply: 1_000_000.0,
                cluster_accumulation_rate: 0.0,
                time_to_estimated_dump: None,
            },
            exit_triggers: vec![],
        };

        let monster = MonsterKill::from_operation(&whale_op, 500.0);
        assert_eq!(monster, MonsterKill::KingBlackDragon);
        assert_eq!(monster.xp_reward(), 1000);
        assert_eq!(monster.emoji(), "🖤");
    }
}