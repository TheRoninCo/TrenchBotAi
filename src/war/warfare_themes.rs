//! Configurable Warfare Themes
//! 
//! Support multiple themes: RuneScape MMO, Military Combat, Pirate Warfare,
//! Street Fighter, Call of Duty, etc. Users can choose their preferred style.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc, Duration};
use std::collections::HashMap;

/// Available warfare themes
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum WarfareTheme {
    RuneScape,   // MMO with levels, XP, monsters
    Military,    // Military ranks, operations, campaigns
    Pirate,      // Pirate crews, treasure, naval battles
    StreetFighter, // Fighting game combos, tournaments
    CallOfDuty,  // Modern warfare, killstreaks, prestige
    Cyberpunk,   // Hacking, netrunning, corpo warfare
    Medieval,    // Knights, castles, siege warfare
    SpaceWar,    // Galactic combat, starships, planets
}

impl WarfareTheme {
    pub fn all_themes() -> Vec<WarfareTheme> {
        vec![
            WarfareTheme::RuneScape,
            WarfareTheme::Military,
            WarfareTheme::Pirate,
            WarfareTheme::StreetFighter,
            WarfareTheme::CallOfDuty,
            WarfareTheme::Cyberpunk,
            WarfareTheme::Medieval,
            WarfareTheme::SpaceWar,
        ]
    }

    pub fn name(&self) -> &'static str {
        match self {
            WarfareTheme::RuneScape => "RuneScape",
            WarfareTheme::Military => "Military",
            WarfareTheme::Pirate => "Pirate",
            WarfareTheme::StreetFighter => "Street Fighter",
            WarfareTheme::CallOfDuty => "Call of Duty",
            WarfareTheme::Cyberpunk => "Cyberpunk",
            WarfareTheme::Medieval => "Medieval",
            WarfareTheme::SpaceWar => "Space War",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            WarfareTheme::RuneScape => "Classic MMO with levels, XP, and monsters",
            WarfareTheme::Military => "Military operations with ranks and campaigns",
            WarfareTheme::Pirate => "Pirate adventures with treasure and naval combat",
            WarfareTheme::StreetFighter => "Fighting tournaments with combos and rivals",
            WarfareTheme::CallOfDuty => "Modern warfare with killstreaks and prestige",
            WarfareTheme::Cyberpunk => "Corporate hacking and netrunning warfare",
            WarfareTheme::Medieval => "Knights and castles in medieval combat",
            WarfareTheme::SpaceWar => "Galactic warfare across star systems",
        }
    }
}

/// Rank progression for each theme
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankProgression {
    pub ranks: Vec<RankTier>,
    pub max_level: u32,
    pub prestige_available: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankTier {
    pub name: String,
    pub emoji: String,
    pub min_kills: u64,
    pub color: String,
}

/// Kill/Target types for each theme  
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetType {
    pub name: String,
    pub emoji: String,
    pub min_profit: f64,
    pub min_wallets: usize,
    pub xp_reward: u64,
    pub rarity: String,
}

/// Theme-specific messaging and flavor text
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThemeMessaging {
    pub kill_prefix: String,
    pub victory_message: String,
    pub level_up_message: String,
    pub special_unlock_message: String,
    pub stats_title: String,
}

/// Complete theme configuration
pub struct ThemeConfig {
    pub theme: WarfareTheme,
    pub rank_progression: RankProgression,
    pub target_types: Vec<TargetType>,
    pub messaging: ThemeMessaging,
    pub special_abilities: Vec<SpecialAbility>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecialAbility {
    pub name: String,
    pub emoji: String,
    pub unlock_level: u32,
    pub description: String,
}

impl ThemeConfig {
    pub fn new(theme: WarfareTheme) -> Self {
        match theme {
            WarfareTheme::RuneScape => Self::runescape_theme(),
            WarfareTheme::Military => Self::military_theme(),
            WarfareTheme::Pirate => Self::pirate_theme(),
            WarfareTheme::StreetFighter => Self::street_fighter_theme(),
            WarfareTheme::CallOfDuty => Self::call_of_duty_theme(),
            WarfareTheme::Cyberpunk => Self::cyberpunk_theme(),
            WarfareTheme::Medieval => Self::medieval_theme(),
            WarfareTheme::SpaceWar => Self::space_war_theme(),
        }
    }

    fn runescape_theme() -> Self {
        Self {
            theme: WarfareTheme::RuneScape,
            rank_progression: RankProgression {
                ranks: vec![
                    RankTier { name: "Recruit".to_string(), emoji: "🪖".to_string(), min_kills: 0, color: "🤎".to_string() },
                    RankTier { name: "Sergeant".to_string(), emoji: "⚪".to_string(), min_kills: 6, color: "⚪".to_string() },
                    RankTier { name: "Lieutenant".to_string(), emoji: "⚫".to_string(), min_kills: 16, color: "⚫".to_string() },
                    RankTier { name: "Captain".to_string(), emoji: "🟫".to_string(), min_kills: 36, color: "🟫".to_string() },
                    RankTier { name: "Major".to_string(), emoji: "🔵".to_string(), min_kills: 76, color: "🔵".to_string() },
                    RankTier { name: "Colonel".to_string(), emoji: "🟢".to_string(), min_kills: 151, color: "🟢".to_string() },
                    RankTier { name: "General".to_string(), emoji: "🔴".to_string(), min_kills: 301, color: "🔴".to_string() },
                    RankTier { name: "Warlord".to_string(), emoji: "🟡".to_string(), min_kills: 501, color: "🟡".to_string() },
                    RankTier { name: "Apex".to_string(), emoji: "💎".to_string(), min_kills: 1001, color: "💎".to_string() },
                ],
                max_level: 99,
                prestige_available: true,
            },
            target_types: vec![
                TargetType { name: "Goblin".to_string(), emoji: "👹".to_string(), min_profit: 0.0, min_wallets: 0, xp_reward: 50, rarity: "Common".to_string() },
                TargetType { name: "Orc".to_string(), emoji: "🧌".to_string(), min_profit: 10.0, min_wallets: 0, xp_reward: 100, rarity: "Common".to_string() },
                TargetType { name: "Dragon".to_string(), emoji: "🐲".to_string(), min_profit: 100.0, min_wallets: 0, xp_reward: 500, rarity: "Rare".to_string() },
                TargetType { name: "King Black Dragon".to_string(), emoji: "🖤".to_string(), min_profit: 500.0, min_wallets: 0, xp_reward: 1000, rarity: "Legendary".to_string() },
            ],
            messaging: ThemeMessaging {
                kill_prefix: "🗡️".to_string(),
                victory_message: "MONSTER SLAIN!".to_string(),
                level_up_message: "🎉 Level Up!".to_string(),
                special_unlock_message: "⚔️ SPECIAL ATTACK UNLOCKED:".to_string(),
                stats_title: "🏰 Adventurer Stats".to_string(),
            },
            special_abilities: vec![
                SpecialAbility { name: "Dragon Dagger".to_string(), emoji: "🗡️".to_string(), unlock_level: 20, description: "Quick strike for fast profits".to_string() },
                SpecialAbility { name: "Abyssal Whip".to_string(), emoji: "🪢".to_string(), unlock_level: 50, description: "Consistent high damage".to_string() },
                SpecialAbility { name: "Armadyl Godsword".to_string(), emoji: "⚔️".to_string(), unlock_level: 85, description: "Devastating special attack".to_string() },
            ],
        }
    }

    fn military_theme() -> Self {
        Self {
            theme: WarfareTheme::Military,
            rank_progression: RankProgression {
                ranks: vec![
                    RankTier { name: "Private".to_string(), emoji: "🎖️".to_string(), min_kills: 0, color: "🟫".to_string() },
                    RankTier { name: "Corporal".to_string(), emoji: "🏅".to_string(), min_kills: 5, color: "🟫".to_string() },
                    RankTier { name: "Sergeant".to_string(), emoji: "⭐".to_string(), min_kills: 15, color: "🔵".to_string() },
                    RankTier { name: "Lieutenant".to_string(), emoji: "🌟".to_string(), min_kills: 35, color: "🔵".to_string() },
                    RankTier { name: "Captain".to_string(), emoji: "💫".to_string(), min_kills: 75, color: "🟢".to_string() },
                    RankTier { name: "Major".to_string(), emoji: "🏆".to_string(), min_kills: 150, color: "🟢".to_string() },
                    RankTier { name: "Colonel".to_string(), emoji: "👑".to_string(), min_kills: 300, color: "🔴".to_string() },
                    RankTier { name: "General".to_string(), emoji: "💎".to_string(), min_kills: 500, color: "🟡".to_string() },
                    RankTier { name: "5-Star General".to_string(), emoji: "🔱".to_string(), min_kills: 1000, color: "💎".to_string() },
                ],
                max_level: 100,
                prestige_available: false,
            },
            target_types: vec![
                TargetType { name: "Insurgent".to_string(), emoji: "🎯".to_string(), min_profit: 0.0, min_wallets: 0, xp_reward: 100, rarity: "Common".to_string() },
                TargetType { name: "Squad Leader".to_string(), emoji: "🔫".to_string(), min_profit: 25.0, min_wallets: 0, xp_reward: 200, rarity: "Uncommon".to_string() },
                TargetType { name: "Tank".to_string(), emoji: "🚗".to_string(), min_profit: 100.0, min_wallets: 0, xp_reward: 500, rarity: "Rare".to_string() },
                TargetType { name: "Command Center".to_string(), emoji: "🏢".to_string(), min_profit: 500.0, min_wallets: 10, xp_reward: 1000, rarity: "Epic".to_string() },
            ],
            messaging: ThemeMessaging {
                kill_prefix: "🎯".to_string(),
                victory_message: "TARGET ELIMINATED!".to_string(),
                level_up_message: "🎖️ PROMOTED!".to_string(),
                special_unlock_message: "🔫 WEAPON UNLOCKED:".to_string(),
                stats_title: "🎖️ Service Record".to_string(),
            },
            special_abilities: vec![
                SpecialAbility { name: "Precision Strike".to_string(), emoji: "🎯".to_string(), unlock_level: 15, description: "Accurate high-value target elimination".to_string() },
                SpecialAbility { name: "Artillery Barrage".to_string(), emoji: "💥".to_string(), unlock_level: 40, description: "Area of effect damage".to_string() },
                SpecialAbility { name: "Tactical Nuke".to_string(), emoji: "☢️".to_string(), unlock_level: 75, description: "Ultimate destruction".to_string() },
            ],
        }
    }

    fn pirate_theme() -> Self {
        Self {
            theme: WarfareTheme::Pirate,
            rank_progression: RankProgression {
                ranks: vec![
                    RankTier { name: "Cabin Boy".to_string(), emoji: "👦".to_string(), min_kills: 0, color: "🟫".to_string() },
                    RankTier { name: "Sailor".to_string(), emoji: "⚓".to_string(), min_kills: 5, color: "🔵".to_string() },
                    RankTier { name: "Bosun".to_string(), emoji: "🔱".to_string(), min_kills: 15, color: "🔵".to_string() },
                    RankTier { name: "First Mate".to_string(), emoji: "🗡️".to_string(), min_kills: 35, color: "🟢".to_string() },
                    RankTier { name: "Pirate Captain".to_string(), emoji: "🏴‍☠️".to_string(), min_kills: 75, color: "🔴".to_string() },
                    RankTier { name: "Commodore".to_string(), emoji: "⚔️".to_string(), min_kills: 150, color: "🟡".to_string() },
                    RankTier { name: "Pirate Lord".to_string(), emoji: "👑".to_string(), min_kills: 300, color: "🟡".to_string() },
                    RankTier { name: "King of Pirates".to_string(), emoji: "💎".to_string(), min_kills: 1000, color: "💎".to_string() },
                ],
                max_level: 100,
                prestige_available: false,
            },
            target_types: vec![
                TargetType { name: "Merchant Ship".to_string(), emoji: "🚢".to_string(), min_profit: 0.0, min_wallets: 0, xp_reward: 75, rarity: "Common".to_string() },
                TargetType { name: "Treasure Chest".to_string(), emoji: "💰".to_string(), min_profit: 50.0, min_wallets: 0, xp_reward: 250, rarity: "Uncommon".to_string() },
                TargetType { name: "Galleon".to_string(), emoji: "⛵".to_string(), min_profit: 150.0, min_wallets: 0, xp_reward: 500, rarity: "Rare".to_string() },
                TargetType { name: "Legendary Treasure".to_string(), emoji: "🏴‍☠️".to_string(), min_profit: 500.0, min_wallets: 0, xp_reward: 1000, rarity: "Legendary".to_string() },
            ],
            messaging: ThemeMessaging {
                kill_prefix: "🏴‍☠️".to_string(),
                victory_message: "TREASURE PLUNDERED!".to_string(),
                level_up_message: "⚓ PROMOTED ABOARD!".to_string(),
                special_unlock_message: "🗡️ WEAPON MASTERED:".to_string(),
                stats_title: "🏴‍☠️ Pirate Reputation".to_string(),
            },
            special_abilities: vec![
                SpecialAbility { name: "Cutlass Combo".to_string(), emoji: "🗡️".to_string(), unlock_level: 20, description: "Swift sword attacks".to_string() },
                SpecialAbility { name: "Cannon Barrage".to_string(), emoji: "💣".to_string(), unlock_level: 45, description: "Ship-to-ship bombardment".to_string() },
                SpecialAbility { name: "Kraken Summoning".to_string(), emoji: "🐙".to_string(), unlock_level: 80, description: "Call forth the sea monster".to_string() },
            ],
        }
    }

    fn street_fighter_theme() -> Self {
        Self {
            theme: WarfareTheme::StreetFighter,
            rank_progression: RankProgression {
                ranks: vec![
                    RankTier { name: "Rookie".to_string(), emoji: "🥊".to_string(), min_kills: 0, color: "🟫".to_string() },
                    RankTier { name: "Fighter".to_string(), emoji: "👊".to_string(), min_kills: 10, color: "🔵".to_string() },
                    RankTier { name: "Combatant".to_string(), emoji: "🤛".to_string(), min_kills: 25, color: "🟢".to_string() },
                    RankTier { name: "Warrior".to_string(), emoji: "💪".to_string(), min_kills: 50, color: "🟢".to_string() },
                    RankTier { name: "Champion".to_string(), emoji: "🏆".to_string(), min_kills: 100, color: "🟡".to_string() },
                    RankTier { name: "Master".to_string(), emoji: "👑".to_string(), min_kills: 200, color: "🟡".to_string() },
                    RankTier { name: "Grandmaster".to_string(), emoji: "💎".to_string(), min_kills: 500, color: "💎".to_string() },
                    RankTier { name: "Legend".to_string(), emoji: "🔱".to_string(), min_kills: 1000, color: "🔱".to_string() },
                ],
                max_level: 50,
                prestige_available: false,
            },
            target_types: vec![
                TargetType { name: "Street Thug".to_string(), emoji: "🥊".to_string(), min_profit: 0.0, min_wallets: 0, xp_reward: 100, rarity: "Common".to_string() },
                TargetType { name: "Rival Fighter".to_string(), emoji: "👊".to_string(), min_profit: 25.0, min_wallets: 0, xp_reward: 200, rarity: "Uncommon".to_string() },
                TargetType { name: "Tournament Boss".to_string(), emoji: "💪".to_string(), min_profit: 100.0, min_wallets: 0, xp_reward: 500, rarity: "Rare".to_string() },
                TargetType { name: "Final Boss".to_string(), emoji: "👹".to_string(), min_profit: 500.0, min_wallets: 0, xp_reward: 1000, rarity: "Legendary".to_string() },
            ],
            messaging: ThemeMessaging {
                kill_prefix: "💥".to_string(),
                victory_message: "K.O.! VICTORY!".to_string(),
                level_up_message: "🥋 SKILL MASTERED!".to_string(),
                special_unlock_message: "🔥 COMBO UNLOCKED:".to_string(),
                stats_title: "🥊 Fighter Profile".to_string(),
            },
            special_abilities: vec![
                SpecialAbility { name: "Hadouken".to_string(), emoji: "🔥".to_string(), unlock_level: 15, description: "Energy projectile attack".to_string() },
                SpecialAbility { name: "Dragon Punch".to_string(), emoji: "🐲".to_string(), unlock_level: 30, description: "Rising uppercut combo".to_string() },
                SpecialAbility { name: "Ultra Combo".to_string(), emoji: "⚡".to_string(), unlock_level: 45, description: "Devastating finishing move".to_string() },
            ],
        }
    }

    fn call_of_duty_theme() -> Self {
        Self {
            theme: WarfareTheme::CallOfDuty,
            rank_progression: RankProgression {
                ranks: vec![
                    RankTier { name: "Recruit".to_string(), emoji: "🎖️".to_string(), min_kills: 0, color: "🟫".to_string() },
                    RankTier { name: "Private".to_string(), emoji: "🏅".to_string(), min_kills: 8, color: "🟫".to_string() },
                    RankTier { name: "Sergeant".to_string(), emoji: "⭐".to_string(), min_kills: 22, color: "🔵".to_string() },
                    RankTier { name: "Lieutenant".to_string(), emoji: "🌟".to_string(), min_kills: 44, color: "🔵".to_string() },
                    RankTier { name: "Captain".to_string(), emoji: "💫".to_string(), min_kills: 80, color: "🟢".to_string() },
                    RankTier { name: "Major".to_string(), emoji: "🏆".to_string(), min_kills: 144, color: "🟢".to_string() },
                    RankTier { name: "Commander".to_string(), emoji: "👑".to_string(), min_kills: 236, color: "🔴".to_string() },
                    RankTier { name: "Prestige".to_string(), emoji: "💎".to_string(), min_kills: 1000, color: "💎".to_string() },
                ],
                max_level: 55,
                prestige_available: true,
            },
            target_types: vec![
                TargetType { name: "Enemy Soldier".to_string(), emoji: "🎯".to_string(), min_profit: 0.0, min_wallets: 0, xp_reward: 100, rarity: "Common".to_string() },
                TargetType { name: "Sniper".to_string(), emoji: "🔫".to_string(), min_profit: 25.0, min_wallets: 0, xp_reward: 150, rarity: "Uncommon".to_string() },
                TargetType { name: "Attack Helicopter".to_string(), emoji: "🚁".to_string(), min_profit: 100.0, min_wallets: 0, xp_reward: 300, rarity: "Rare".to_string() },
                TargetType { name: "AC-130".to_string(), emoji: "✈️".to_string(), min_profit: 500.0, min_wallets: 15, xp_reward: 1000, rarity: "Legendary".to_string() },
            ],
            messaging: ThemeMessaging {
                kill_prefix: "💥".to_string(),
                victory_message: "ENEMY DOWN!".to_string(),
                level_up_message: "📈 RANK UP!".to_string(),
                special_unlock_message: "🔥 KILLSTREAK REWARD:".to_string(),
                stats_title: "🎖️ Combat Record".to_string(),
            },
            special_abilities: vec![
                SpecialAbility { name: "UAV".to_string(), emoji: "📡".to_string(), unlock_level: 10, description: "Reveal enemy positions".to_string() },
                SpecialAbility { name: "Predator Missile".to_string(), emoji: "🚀".to_string(), unlock_level: 25, description: "Guided missile strike".to_string() },
                SpecialAbility { name: "Tactical Nuke".to_string(), emoji: "☢️".to_string(), unlock_level: 50, description: "End the game instantly".to_string() },
            ],
        }
    }

    fn cyberpunk_theme() -> Self {
        Self {
            theme: WarfareTheme::Cyberpunk,
            rank_progression: RankProgression {
                ranks: vec![
                    RankTier { name: "Script Kiddie".to_string(), emoji: "💾".to_string(), min_kills: 0, color: "🟫".to_string() },
                    RankTier { name: "Hacker".to_string(), emoji: "💻".to_string(), min_kills: 10, color: "🔵".to_string() },
                    RankTier { name: "Netrunner".to_string(), emoji: "🔌".to_string(), min_kills: 25, color: "🟢".to_string() },
                    RankTier { name: "Console Cowboy".to_string(), emoji: "🤠".to_string(), min_kills: 50, color: "🟢".to_string() },
                    RankTier { name: "Cyber Samurai".to_string(), emoji: "⚔️".to_string(), min_kills: 100, color: "🔴".to_string() },
                    RankTier { name: "Data Thief".to_string(), emoji: "🏴‍☠️".to_string(), min_kills: 200, color: "🟡".to_string() },
                    RankTier { name: "Ghost in Shell".to_string(), emoji: "👻".to_string(), min_kills: 500, color: "💜".to_string() },
                    RankTier { name: "AI Overlord".to_string(), emoji: "🤖".to_string(), min_kills: 1000, color: "💎".to_string() },
                ],
                max_level: 100,
                prestige_available: false,
            },
            target_types: vec![
                TargetType { name: "Corp Shill".to_string(), emoji: "💼".to_string(), min_profit: 0.0, min_wallets: 0, xp_reward: 75, rarity: "Common".to_string() },
                TargetType { name: "Security IC".to_string(), emoji: "🛡️".to_string(), min_profit: 25.0, min_wallets: 0, xp_reward: 200, rarity: "Uncommon".to_string() },
                TargetType { name: "Corpo Executive".to_string(), emoji: "👔".to_string(), min_profit: 100.0, min_wallets: 0, xp_reward: 500, rarity: "Rare".to_string() },
                TargetType { name: "Megacorp Mainframe".to_string(), emoji: "🏢".to_string(), min_profit: 500.0, min_wallets: 20, xp_reward: 1000, rarity: "Legendary".to_string() },
            ],
            messaging: ThemeMessaging {
                kill_prefix: "⚡".to_string(),
                victory_message: "SYSTEM BREACHED!".to_string(),
                level_up_message: "🧠 NEURAL UPGRADE!".to_string(),
                special_unlock_message: "🔧 EXPLOIT UNLOCKED:".to_string(),
                stats_title: "🤖 Netrunner Profile".to_string(),
            },
            special_abilities: vec![
                SpecialAbility { name: "Data Spike".to_string(), emoji: "⚡".to_string(), unlock_level: 15, description: "Direct neural hack".to_string() },
                SpecialAbility { name: "Virus Injection".to_string(), emoji: "🦠".to_string(), unlock_level: 35, description: "Spread malicious code".to_string() },
                SpecialAbility { name: "System Override".to_string(), emoji: "🔓".to_string(), unlock_level: 65, description: "Total system control".to_string() },
            ],
        }
    }

    fn medieval_theme() -> Self {
        Self {
            theme: WarfareTheme::Medieval,
            rank_progression: RankProgression {
                ranks: vec![
                    RankTier { name: "Peasant".to_string(), emoji: "👨‍🌾".to_string(), min_kills: 0, color: "🟫".to_string() },
                    RankTier { name: "Squire".to_string(), emoji: "🛡️".to_string(), min_kills: 5, color: "🔵".to_string() },
                    RankTier { name: "Knight".to_string(), emoji: "⚔️".to_string(), min_kills: 20, color: "🔵".to_string() },
                    RankTier { name: "Knight-Captain".to_string(), emoji: "🗡️".to_string(), min_kills: 50, color: "🟢".to_string() },
                    RankTier { name: "Baron".to_string(), emoji: "🏰".to_string(), min_kills: 100, color: "🟢".to_string() },
                    RankTier { name: "Earl".to_string(), emoji: "👑".to_string(), min_kills: 200, color: "🔴".to_string() },
                    RankTier { name: "Duke".to_string(), emoji: "💎".to_string(), min_kills: 400, color: "🟡".to_string() },
                    RankTier { name: "King".to_string(), emoji: "🔱".to_string(), min_kills: 1000, color: "💎".to_string() },
                ],
                max_level: 100,
                prestige_available: false,
            },
            target_types: vec![
                TargetType { name: "Bandit".to_string(), emoji: "🗡️".to_string(), min_profit: 0.0, min_wallets: 0, xp_reward: 50, rarity: "Common".to_string() },
                TargetType { name: "Enemy Knight".to_string(), emoji: "⚔️".to_string(), min_profit: 50.0, min_wallets: 0, xp_reward: 200, rarity: "Uncommon".to_string() },
                TargetType { name: "Castle Siege".to_string(), emoji: "🏰".to_string(), min_profit: 200.0, min_wallets: 8, xp_reward: 500, rarity: "Rare".to_string() },
                TargetType { name: "Dragon".to_string(), emoji: "🐉".to_string(), min_profit: 500.0, min_wallets: 0, xp_reward: 1000, rarity: "Legendary".to_string() },
            ],
            messaging: ThemeMessaging {
                kill_prefix: "⚔️".to_string(),
                victory_message: "ENEMY VANQUISHED!".to_string(),
                level_up_message: "🏆 HONOR GAINED!".to_string(),
                special_unlock_message: "🗡️ WEAPON MASTERY:".to_string(),
                stats_title: "🏰 Noble Deeds".to_string(),
            },
            special_abilities: vec![
                SpecialAbility { name: "Sword Strike".to_string(), emoji: "⚔️".to_string(), unlock_level: 10, description: "Precise blade work".to_string() },
                SpecialAbility { name: "Cavalry Charge".to_string(), emoji: "🐎".to_string(), unlock_level: 30, description: "Mounted assault".to_string() },
                SpecialAbility { name: "Siege Engine".to_string(), emoji: "🏹".to_string(), unlock_level: 60, description: "Trebuchet bombardment".to_string() },
            ],
        }
    }

    fn space_war_theme() -> Self {
        Self {
            theme: WarfareTheme::SpaceWar,
            rank_progression: RankProgression {
                ranks: vec![
                    RankTier { name: "Cadet".to_string(), emoji: "🚀".to_string(), min_kills: 0, color: "🟫".to_string() },
                    RankTier { name: "Pilot".to_string(), emoji: "👨‍🚀".to_string(), min_kills: 8, color: "🔵".to_string() },
                    RankTier { name: "Lieutenant".to_string(), emoji: "🌟".to_string(), min_kills: 25, color: "🔵".to_string() },
                    RankTier { name: "Commander".to_string(), emoji: "🛸".to_string(), min_kills: 60, color: "🟢".to_string() },
                    RankTier { name: "Captain".to_string(), emoji: "🚁".to_string(), min_kills: 120, color: "🟢".to_string() },
                    RankTier { name: "Admiral".to_string(), emoji: "⭐".to_string(), min_kills: 250, color: "🔴".to_string() },
                    RankTier { name: "Fleet Commander".to_string(), emoji: "💫".to_string(), min_kills: 500, color: "🟡".to_string() },
                    RankTier { name: "Galactic Emperor".to_string(), emoji: "🌌".to_string(), min_kills: 1000, color: "💎".to_string() },
                ],
                max_level: 100,
                prestige_available: false,
            },
            target_types: vec![
                TargetType { name: "Scout Ship".to_string(), emoji: "🛸".to_string(), min_profit: 0.0, min_wallets: 0, xp_reward: 75, rarity: "Common".to_string() },
                TargetType { name: "Frigate".to_string(), emoji: "🚀".to_string(), min_profit: 50.0, min_wallets: 0, xp_reward: 200, rarity: "Uncommon".to_string() },
                TargetType { name: "Battlecruiser".to_string(), emoji: "🛰️".to_string(), min_profit: 150.0, min_wallets: 0, xp_reward: 500, rarity: "Rare".to_string() },
                TargetType { name: "Death Star".to_string(), emoji: "💫".to_string(), min_profit: 500.0, min_wallets: 25, xp_reward: 1000, rarity: "Legendary".to_string() },
            ],
            messaging: ThemeMessaging {
                kill_prefix: "💥".to_string(),
                victory_message: "TARGET DESTROYED!".to_string(),
                level_up_message: "🌟 PROMOTION!".to_string(),
                special_unlock_message: "🔫 WEAPON SYSTEM ONLINE:".to_string(),
                stats_title: "🚀 Space Command Record".to_string(),
            },
            special_abilities: vec![
                SpecialAbility { name: "Laser Cannon".to_string(), emoji: "🔫".to_string(), unlock_level: 15, description: "High-energy beam weapon".to_string() },
                SpecialAbility { name: "Ion Torpedo".to_string(), emoji: "🚀".to_string(), unlock_level: 40, description: "Guided space missile".to_string() },
                SpecialAbility { name: "Supernova Bomb".to_string(), emoji: "💥".to_string(), unlock_level: 75, description: "Planet-destroying weapon".to_string() },
            ],
        }
    }

    /// Get the appropriate target type based on profit and wallets
    pub fn classify_target(&self, profit: f64, wallets: usize) -> &TargetType {
        // Find the highest tier target that matches criteria
        self.target_types
            .iter()
            .rev() // Start from highest tier
            .find(|target| profit >= target.min_profit && wallets >= target.min_wallets)
            .unwrap_or(&self.target_types[0]) // Fallback to lowest tier
    }

    /// Get the appropriate rank based on kill count
    pub fn get_rank(&self, kills: u64) -> &RankTier {
        self.rank_progression.ranks
            .iter()
            .rev()
            .find(|rank| kills >= rank.min_kills)
            .unwrap_or(&self.rank_progression.ranks[0])
    }

    /// Generate a kill notification message
    pub fn format_kill_message(&self, target: &TargetType, profit: f64, xp: u64, rank: &RankTier) -> String {
        format!(
            "{} {} {}! {} GP | {} XP | Rank: {} {}",
            self.messaging.kill_prefix,
            target.emoji,
            target.name,
            (profit * 1000.0) as u64,
            xp,
            rank.emoji,
            rank.name
        )
    }

    /// Generate a level up message
    pub fn format_level_up_message(&self, new_level: u32) -> String {
        format!("{} You are now level {}!", self.messaging.level_up_message, new_level)
    }
}

/// Main themeable warfare system
pub struct ThemeableWarfareSystem {
    pub theme_config: ThemeConfig,
    pub player_level: u32,
    pub total_kills: u64,
    pub total_xp: u64,
    pub total_profit: f64,
    pub kill_streak: u32,
    pub prestige_level: u32,
}

impl ThemeableWarfareSystem {
    pub fn new(theme: WarfareTheme) -> Self {
        Self {
            theme_config: ThemeConfig::new(theme),
            player_level: 1,
            total_kills: 0,
            total_xp: 0,
            total_profit: 0.0,
            kill_streak: 0,
            prestige_level: 0,
        }
    }

    /// Switch themes dynamically
    pub fn switch_theme(&mut self, new_theme: WarfareTheme) {
        self.theme_config = ThemeConfig::new(new_theme);
    }

    /// Record a kill and return notifications
    pub async fn record_kill(&mut self, profit: f64, wallets: usize) -> Vec<String> {
        let target = self.theme_config.classify_target(profit, wallets);
        let xp_gained = target.xp_reward;
        
        // Update stats
        self.total_kills += 1;
        self.total_xp += xp_gained;
        self.total_profit += profit;
        self.kill_streak += 1;
        
        // Calculate new level (simple formula)
        let new_level = 1 + ((self.total_xp as f64).sqrt() / 10.0) as u32;
        let level_up = new_level > self.player_level;
        self.player_level = new_level;
        
        let mut notifications = Vec::new();
        
        // Get current rank
        let rank = self.theme_config.get_rank(self.total_kills);
        
        // Kill notification
        notifications.push(self.theme_config.format_kill_message(target, profit, xp_gained, rank));
        
        // Level up notification
        if level_up {
            notifications.push(self.theme_config.format_level_up_message(new_level));
            
            // Check for special ability unlocks
            for ability in &self.theme_config.special_abilities {
                if new_level == ability.unlock_level {
                    notifications.push(format!(
                        "{} {} {}! {}",
                        self.theme_config.messaging.special_unlock_message,
                        ability.emoji,
                        ability.name,
                        ability.description
                    ));
                }
            }
        }
        
        notifications
    }

    /// Get themed stats display
    pub fn get_stats_display(&self) -> String {
        let rank = self.theme_config.get_rank(self.total_kills);
        
        let prestige_display = if self.prestige_level > 0 {
            format!(" ★{}", self.prestige_level)
        } else {
            String::new()
        };
        
        format!(
            "\n{}\n\
            {} Rank: {} {}{}\n\
            Level: {} | Total Kills: {} | Streak: {}\n\
            Total XP: {} | Bank: {} GP\n\
            Total Profit: {:.2} SOL | Avg: {:.2} SOL/kill\n\
            Theme: {} 🎭",
            self.theme_config.messaging.stats_title,
            rank.color,
            rank.emoji,
            rank.name,
            prestige_display,
            self.player_level,
            self.total_kills,
            self.kill_streak,
            self.total_xp,
            (self.total_profit * 1000.0) as u64,
            self.total_profit,
            if self.total_kills > 0 { self.total_profit / self.total_kills as f64 } else { 0.0 },
            self.theme_config.theme.name()
        )
    }

    /// List all available themes
    pub fn list_themes() -> String {
        let mut output = String::from("\n🎭 Available Warfare Themes:\n");
        
        for theme in WarfareTheme::all_themes() {
            output.push_str(&format!(
                "  {} - {}\n",
                theme.name(),
                theme.description()
            ));
        }
        
        output.push_str("\nUse `!theme <name>` to switch themes");
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_multiple_themes() {
        // Test different themes
        let themes = vec![
            WarfareTheme::Military,
            WarfareTheme::Pirate,
            WarfareTheme::StreetFighter,
        ];
        
        for theme in themes {
            let mut warfare = ThemeableWarfareSystem::new(theme.clone());
            
            // Simulate some kills
            let notifications = warfare.record_kill(150.0, 3).await;
            assert!(!notifications.is_empty());
            
            println!("Theme: {:?}", theme);
            for notification in notifications {
                println!("  {}", notification);
            }
            println!("{}\n", warfare.get_stats_display());
        }
    }

    #[test]
    fn test_theme_switching() {
        let mut warfare = ThemeableWarfareSystem::new(WarfareTheme::Military);
        assert_eq!(warfare.theme_config.theme, WarfareTheme::Military);
        
        warfare.switch_theme(WarfareTheme::Pirate);
        assert_eq!(warfare.theme_config.theme, WarfareTheme::Pirate);
        
        // Stats should persist across theme changes
        assert_eq!(warfare.total_kills, 0);
        assert_eq!(warfare.player_level, 1);
    }

    #[test]
    fn test_theme_list() {
        let themes_list = ThemeableWarfareSystem::list_themes();
        assert!(themes_list.contains("Military"));
        assert!(themes_list.contains("Pirate"));
        assert!(themes_list.contains("RuneScape"));
        println!("{}", themes_list);
    }
}