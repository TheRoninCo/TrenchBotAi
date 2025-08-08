use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// **TRENCHBOT RANKING SYSTEM**
/// Themed ranking progression that users can choose from popular games/culture
/// Each theme has unique ranks, titles, perks, and lingo
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankingSystem {
    pub available_themes: HashMap<RankTheme, ThemeData>,
    pub user_rankings: HashMap<String, UserRank>,
    pub global_leaderboards: HashMap<RankTheme, Vec<LeaderboardEntry>>,
    pub rank_requirements: HashMap<RankTheme, Vec<RankRequirement>>,
}

/// **RANK THEMES**
/// Different gaming/cultural themes users can choose from
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RankTheme {
    RuneScape,          // "Combat level, skill mastery"
    CallOfDuty,         // "Military ranks, prestige system"
    Fortnite,           // "Battle pass levels, victory royales"
    LeagueOfLegends,    // "Iron to Challenger"
    CounterStrike,      // "Silver to Global Elite"
    Valorant,           // "Iron to Radiant"
    Apex,               // "Bronze to Apex Predator"
    Crypto,             // "Paper hands to Diamond hands"
    WallStreet,         // "Intern to Wolf of Wall Street"
    Gaming,             // "Noob to God Tier"
    Anime,              // "Genin to Hokage (Naruto style)"
    Military,           // "Private to General"
}

/// **THEME DATA**
/// Complete theming for each ranking system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThemeData {
    pub theme_name: String,
    pub description: String,
    pub ranks: Vec<RankTier>,
    pub theme_lingo: ThemeLingo,
    pub rank_colors: HashMap<String, String>, // Rank name -> color hex
    pub rank_emojis: HashMap<String, String>, // Rank name -> emoji
    pub progression_messages: Vec<String>,
    pub theme_sounds: Vec<String>, // Sound effects for this theme
}

/// **INDIVIDUAL RANK TIER**
/// Each rank within a theme
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankTier {
    pub rank_id: u32,
    pub rank_name: String,
    pub rank_display_name: String,
    pub requirements: RankRequirements,
    pub perks: Vec<RankPerk>,
    pub unlock_message: String,
    pub rank_badge: String, // ASCII art or emoji badge
}

/// **USER RANK**
/// Individual user's ranking progress
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserRank {
    pub user_id: String,
    pub chosen_theme: RankTheme,
    pub current_rank: u32,
    pub rank_progress: RankProgress,
    pub achievements: Vec<Achievement>,
    pub rank_history: Vec<RankHistoryEntry>,
    pub prestige_level: Option<u32>, // For themes that support prestige
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankProgress {
    pub total_profits: f64,
    pub successful_trades: u32,
    pub win_rate: f64,
    pub biggest_win: f64,
    pub total_volume_traded: f64,
    pub days_active: u32,
    pub ai_accuracy_score: f64,
    pub scammers_rekt: u32,
    pub memecoins_sniped: u32,
}

impl RankingSystem {
    pub fn new() -> Self {
        Self {
            available_themes: Self::create_all_themes(),
            user_rankings: HashMap::new(),
            global_leaderboards: HashMap::new(),
            rank_requirements: Self::create_rank_requirements(),
        }
    }
    
    /// **CREATE ALL RANKING THEMES**
    fn create_all_themes() -> HashMap<RankTheme, ThemeData> {
        let mut themes = HashMap::new();
        
        // === RUNESCAPE THEME ===
        themes.insert(RankTheme::RuneScape, ThemeData {
            theme_name: "RuneScape Combat Levels".to_string(),
            description: "Level up your trading skills like combat in Gielinor! 🏰".to_string(),
            ranks: vec![
                RankTier {
                    rank_id: 1,
                    rank_name: "Noob".to_string(),
                    rank_display_name: "Combat Level 3".to_string(),
                    requirements: RankRequirements {
                        min_profits: 0.0,
                        min_successful_trades: 0,
                        min_win_rate: 0.0,
                        special_requirements: vec![],
                    },
                    perks: vec![RankPerk::BasicTrading],
                    unlock_message: "Welcome to TrenchScape! You've spawned in Lumbridge. Time to start your trading journey! ⚔️".to_string(),
                    rank_badge: "🛡️⚔️".to_string(),
                },
                RankTier {
                    rank_id: 10,
                    rank_name: "Fighter".to_string(),
                    rank_display_name: "Combat Level 20".to_string(),
                    requirements: RankRequirements {
                        min_profits: 100.0,
                        min_successful_trades: 10,
                        min_win_rate: 0.6,
                        special_requirements: vec![],
                    },
                    perks: vec![RankPerk::AdvancedAnalytics],
                    unlock_message: "Grats! Combat level 20 achieved! You can now wield better trading strategies! 🗡️".to_string(),
                    rank_badge: "⚔️🛡️".to_string(),
                },
                RankTier {
                    rank_id: 40,
                    rank_name: "Warrior".to_string(), 
                    rank_display_name: "Combat Level 60".to_string(),
                    requirements: RankRequirements {
                        min_profits: 1000.0,
                        min_successful_trades: 50,
                        min_win_rate: 0.7,
                        special_requirements: vec!["Complete Dragon Slayer (make 10x profit on a single trade)".to_string()],
                    },
                    perks: vec![RankPerk::DragonWeapons, RankPerk::CustomStrategies],
                    unlock_message: "Holy moly! Combat 60! You can now wield dragon weapons (advanced trading tools)! 🐉".to_string(),
                    rank_badge: "🐉⚔️".to_string(),
                },
                RankTier {
                    rank_id: 70,
                    rank_name: "Champion".to_string(),
                    rank_display_name: "Combat Level 99".to_string(),
                    requirements: RankRequirements {
                        min_profits: 10000.0,
                        min_successful_trades: 200,
                        min_win_rate: 0.8,
                        special_requirements: vec!["Achieve 99 in Trading skill".to_string()],
                    },
                    perks: vec![RankPerk::MaxCape, RankPerk::UnlimitedTrades],
                    unlock_message: "🎉 DING! 99 Trading achieved! You've mastered the markets! Skillcape unlocked! 🎽".to_string(),
                    rank_badge: "🎽👑".to_string(),
                },
                RankTier {
                    rank_id: 126,
                    rank_name: "Maxed".to_string(),
                    rank_display_name: "Combat Level 126".to_string(),
                    requirements: RankRequirements {
                        min_profits: 100000.0,
                        min_successful_trades: 1000,
                        min_win_rate: 0.85,
                        special_requirements: vec!["Max all trading stats".to_string()],
                    },
                    perks: vec![RankPerk::MaxCape, RankPerk::GodMode],
                    unlock_message: "Congratulations! You're maxed! The ultimate TrenchScape achievement! 👑".to_string(),
                    rank_badge: "👑🏆".to_string(),
                },
            ],
            theme_lingo: ThemeLingo {
                rank_up: "Ding! Level up!".to_string(),
                good_trade: "Nice! That's some good XP right there!".to_string(),
                bad_trade: "Oof, you've been PKed by the market!".to_string(),
                achievement: "Achievement unlocked!".to_string(),
                leaderboard: "Hiscores".to_string(),
            },
            rank_colors: [
                ("Noob".to_string(), "#8B4513".to_string()), // Bronze
                ("Fighter".to_string(), "#C0C0C0".to_string()), // Silver  
                ("Warrior".to_string(), "#FFD700".to_string()), // Gold
                ("Champion".to_string(), "#FF0000".to_string()), // Red (99 cape)
                ("Maxed".to_string(), "#800080".to_string()), // Purple (max cape)
            ].into_iter().collect(),
            rank_emojis: [
                ("Noob".to_string(), "🛡️".to_string()),
                ("Fighter".to_string(), "⚔️".to_string()),
                ("Warrior".to_string(), "🐉".to_string()),
                ("Champion".to_string(), "🎽".to_string()),
                ("Maxed".to_string(), "👑".to_string()),
            ].into_iter().collect(),
            progression_messages: vec![
                "Your Trading level is now {level}!".to_string(),
                "Congratulations! Your Trading level is now {level}!".to_string(),
                "Well done! Your Trading level is now {level}!".to_string(),
            ],
            theme_sounds: vec!["level_up.wav", "achievement.wav", "quest_complete.wav"].into_iter().map(|s| s.to_string()).collect(),
        });
        
        // === CALL OF DUTY THEME ===
        themes.insert(RankTheme::CallOfDuty, ThemeData {
            theme_name: "Military Ranks".to_string(),
            description: "Rise through the ranks like a true warrior! Hoorah! 🪖".to_string(),
            ranks: vec![
                RankTier {
                    rank_id: 1,
                    rank_name: "Private".to_string(),
                    rank_display_name: "Private (E-1)".to_string(),
                    requirements: RankRequirements {
                        min_profits: 0.0,
                        min_successful_trades: 0,
                        min_win_rate: 0.0,
                        special_requirements: vec![],
                    },
                    perks: vec![RankPerk::BasicTraining],
                    unlock_message: "Welcome to the battlefield, soldier! Time to prove yourself! 🪖".to_string(),
                    rank_badge: "🪖".to_string(),
                },
                RankTier {
                    rank_id: 10,
                    rank_name: "Sergeant".to_string(),
                    rank_display_name: "Staff Sergeant (E-6)".to_string(),
                    requirements: RankRequirements {
                        min_profits: 500.0,
                        min_successful_trades: 25,
                        min_win_rate: 0.65,
                        special_requirements: vec![],
                    },
                    perks: vec![RankPerk::SquadLeader],
                    unlock_message: "Promotion earned! You're now leading the squad, Sergeant! 🎖️".to_string(),
                    rank_badge: "🎖️🪖".to_string(),
                },
                RankTier {
                    rank_id: 25,
                    rank_name: "Lieutenant".to_string(),
                    rank_display_name: "First Lieutenant (O-2)".to_string(),
                    requirements: RankRequirements {
                        min_profits: 2500.0,
                        min_successful_trades: 100,
                        min_win_rate: 0.75,
                        special_requirements: vec!["Complete Officer Training (achieve 20+ win streak)".to_string()],
                    },
                    perks: vec![RankPerk::OfficerPrivileges, RankPerk::AdvancedIntel],
                    unlock_message: "Outstanding work, Lieutenant! You've earned your officer bars! 🎖️✨".to_string(),
                    rank_badge: "🎖️⭐".to_string(),
                },
                RankTier {
                    rank_id: 55,
                    rank_name: "Colonel".to_string(),
                    rank_display_name: "Colonel (O-6)".to_string(),
                    requirements: RankRequirements {
                        min_profits: 25000.0,
                        min_successful_trades: 500,
                        min_win_rate: 0.8,
                        special_requirements: vec!["Lead successful operation (manage $100k+ in trades)".to_string()],
                    },
                    perks: vec![RankPerk::CommandAuthority, RankPerk::SpecialOperations],
                    unlock_message: "Exceptional leadership, Colonel! The base salutes you! 🫡".to_string(),
                    rank_badge: "🌟🎖️".to_string(),
                },
                RankTier {
                    rank_id: 80,
                    rank_name: "General".to_string(),
                    rank_display_name: "General (O-10)".to_string(),
                    requirements: RankRequirements {
                        min_profits: 100000.0,
                        min_successful_trades: 1000,
                        min_win_rate: 0.85,
                        special_requirements: vec!["Command respect of the entire army".to_string()],
                    },
                    perks: vec![RankPerk::SupremeCommand, RankPerk::LegendaryStatus],
                    unlock_message: "Sir! General on deck! You've achieved the highest honor! 🫡⭐⭐⭐⭐".to_string(),
                    rank_badge: "⭐⭐⭐⭐".to_string(),
                },
            ],
            theme_lingo: ThemeLingo {
                rank_up: "Promotion earned!".to_string(),
                good_trade: "Mission accomplished, soldier!".to_string(),
                bad_trade: "We've taken casualties, but we'll recover!".to_string(),
                achievement: "Medal of Honor awarded!".to_string(),
                leaderboard: "Chain of Command".to_string(),
            },
            rank_colors: [
                ("Private".to_string(), "#8B7355".to_string()), // Army green
                ("Sergeant".to_string(), "#4B0082".to_string()), // Navy blue
                ("Lieutenant".to_string(), "#FFD700".to_string()), // Gold
                ("Colonel".to_string(), "#C0C0C0".to_string()), // Silver
                ("General".to_string(), "#800080".to_string()), // Purple
            ].into_iter().collect(),
            rank_emojis: [
                ("Private".to_string(), "🪖".to_string()),
                ("Sergeant".to_string(), "🎖️".to_string()),
                ("Lieutenant".to_string(), "⭐".to_string()),
                ("Colonel".to_string(), "🌟".to_string()),
                ("General".to_string(), "⭐⭐⭐⭐".to_string()),
            ].into_iter().collect(),
            progression_messages: vec![
                "Soldier, you've been promoted to {rank}!".to_string(),
                "Outstanding performance! Welcome to {rank}!".to_string(),
                "Your service is exemplary, {rank}!".to_string(),
            ],
            theme_sounds: vec!["promotion.wav", "salute.wav", "medal.wav"].into_iter().map(|s| s.to_string()).collect(),
        });
        
        // === CRYPTO THEME ===
        themes.insert(RankTheme::Crypto, ThemeData {
            theme_name: "Diamond Hands Progression".to_string(),
            description: "From paper hands to diamond hands - the crypto way! 💎🙌".to_string(),
            ranks: vec![
                RankTier {
                    rank_id: 1,
                    rank_name: "Paper Hands".to_string(),
                    rank_display_name: "📄🙌 Paper Hands".to_string(),
                    requirements: RankRequirements {
                        min_profits: 0.0,
                        min_successful_trades: 0,
                        min_win_rate: 0.0,
                        special_requirements: vec![],
                    },
                    perks: vec![RankPerk::BasicTrading],
                    unlock_message: "Welcome to crypto! Time to strengthen those hands! 📄➡️💎".to_string(),
                    rank_badge: "📄🙌".to_string(),
                },
                RankTier {
                    rank_id: 10,
                    rank_name: "Plastic Hands".to_string(),
                    rank_display_name: "🪣 Plastic Hands".to_string(),
                    requirements: RankRequirements {
                        min_profits: 50.0,
                        min_successful_trades: 5,
                        min_win_rate: 0.5,
                        special_requirements: vec![],
                    },
                    perks: vec![RankPerk::HodlStrength],
                    unlock_message: "Getting stronger! Your hands are hardening! 🪣➡️💎".to_string(),
                    rank_badge: "🪣🙌".to_string(),
                },
                RankTier {
                    rank_id: 25,
                    rank_name: "Steel Hands".to_string(),
                    rank_display_name: "⚔️ Steel Hands".to_string(),
                    requirements: RankRequirements {
                        min_profits: 500.0,
                        min_successful_trades: 50,
                        min_win_rate: 0.7,
                        special_requirements: vec!["HODL through a -50% dip".to_string()],
                    },
                    perks: vec![RankPerk::DiamondFormation],
                    unlock_message: "Impressive! Your hands are forged in steel! ⚔️💪".to_string(),
                    rank_badge: "⚔️🙌".to_string(),
                },
                RankTier {
                    rank_id: 50,
                    rank_name: "Diamond Hands".to_string(),
                    rank_display_name: "💎🙌 Diamond Hands".to_string(),
                    requirements: RankRequirements {
                        min_profits: 5000.0,
                        min_successful_trades: 200,
                        min_win_rate: 0.8,
                        special_requirements: vec!["HODL for 6 months minimum".to_string()],
                    },
                    perks: vec![RankPerk::DiamondStatus, RankPerk::MoonMission],
                    unlock_message: "🚀 TO THE MOON! You've achieved DIAMOND HANDS status! 💎🙌🚀".to_string(),
                    rank_badge: "💎🙌".to_string(),
                },
                RankTier {
                    rank_id: 100,
                    rank_name: "Whale".to_string(),
                    rank_display_name: "🐋 Crypto Whale".to_string(),
                    requirements: RankRequirements {
                        min_profits: 100000.0,
                        min_successful_trades: 1000,
                        min_win_rate: 0.85,
                        special_requirements: vec!["Move markets with your trades".to_string()],
                    },
                    perks: vec![RankPerk::WhaleStatus, RankPerk::MarketMover],
                    unlock_message: "🐋 WHALE ALERT! You're now moving markets, captain! 🐋💰".to_string(),
                    rank_badge: "🐋👑".to_string(),
                },
            ],
            theme_lingo: ThemeLingo {
                rank_up: "Hands getting stronger! 💪".to_string(),
                good_trade: "Diamond hands paying off! 💎".to_string(),
                bad_trade: "Just a dip, HODL strong! 📈".to_string(),
                achievement: "Achievement unlocked, degen! 🏆".to_string(),
                leaderboard: "Whale Watching".to_string(),
            },
            rank_colors: [
                ("Paper Hands".to_string(), "#FFFFFF".to_string()), // White
                ("Plastic Hands".to_string(), "#87CEEB".to_string()), // Light blue
                ("Steel Hands".to_string(), "#708090".to_string()), // Steel blue
                ("Diamond Hands".to_string(), "#00FFFF".to_string()), // Cyan
                ("Whale".to_string(), "#4169E1".to_string()), // Royal blue
            ].into_iter().collect(),
            rank_emojis: [
                ("Paper Hands".to_string(), "📄".to_string()),
                ("Plastic Hands".to_string(), "🪣".to_string()),
                ("Steel Hands".to_string(), "⚔️".to_string()),
                ("Diamond Hands".to_string(), "💎".to_string()),
                ("Whale".to_string(), "🐋".to_string()),
            ].into_iter().collect(),
            progression_messages: vec![
                "Your hands are getting stronger! Now at {rank}! 💪".to_string(),
                "HODL gang! You've reached {rank} status! 🚀".to_string(),
                "Wen lambo? At {rank}, you're getting closer! 🏎️".to_string(),
            ],
            theme_sounds: vec!["rocket.wav", "diamond.wav", "whale.wav"].into_iter().map(|s| s.to_string()).collect(),
        });
        
        // === FORTNITE THEME ===
        themes.insert(RankTheme::Fortnite, ThemeData {
            theme_name: "Victory Royale Progression".to_string(),
            description: "Battle your way to Victory Royales in the trading arena! 👑".to_string(),
            ranks: vec![
                RankTier {
                    rank_id: 1,
                    rank_name: "Default Skin".to_string(),
                    rank_display_name: "Default Skin (No Wins)".to_string(),
                    requirements: RankRequirements {
                        min_profits: 0.0,
                        min_successful_trades: 0,
                        min_win_rate: 0.0,
                        special_requirements: vec![],
                    },
                    perks: vec![RankPerk::BasicPickaxe],
                    unlock_message: "Welcome to the island! Time to get your first Victory Royale! 🏝️".to_string(),
                    rank_badge: "🏝️⛏️".to_string(),
                },
                RankTier {
                    rank_id: 10,
                    rank_name: "Battle Pass".to_string(),
                    rank_display_name: "Battle Pass Level 25".to_string(),
                    requirements: RankRequirements {
                        min_profits: 100.0,
                        min_successful_trades: 10,
                        min_win_rate: 0.6,
                        special_requirements: vec![],
                    },
                    perks: vec![RankPerk::CustomSkin],
                    unlock_message: "Battle Pass progress! New skin unlocked! Looking fresh! ✨".to_string(),
                    rank_badge: "✨🎽".to_string(),
                },
                RankTier {
                    rank_id: 25,
                    rank_name: "Sweaty Builder".to_string(),
                    rank_display_name: "Sweaty Builder (50+ Wins)".to_string(),
                    requirements: RankRequirements {
                        min_profits: 1000.0,
                        min_successful_trades: 50,
                        min_win_rate: 0.75,
                        special_requirements: vec!["Get 10 Victory Royales in a row".to_string()],
                    },
                    perks: vec![RankPerk::CrankedBuilds, RankPerk::SweatMode],
                    unlock_message: "You're cranking 90s now! Sweaty builder status achieved! 🏗️💦".to_string(),
                    rank_badge: "🏗️💦".to_string(),
                },
                RankTier {
                    rank_id: 50,
                    rank_name: "Pro Player".to_string(),
                    rank_display_name: "Pro Player (Champion League)".to_string(),
                    requirements: RankRequirements {
                        min_profits: 10000.0,
                        min_successful_trades: 200,
                        min_win_rate: 0.8,
                        special_requirements: vec!["Qualify for Championship".to_string()],
                    },
                    perks: vec![RankPerk::ProStatus, RankPerk::TournamentAccess],
                    unlock_message: "Welcome to the big leagues! You're officially a pro! 🏆🎮".to_string(),
                    rank_badge: "🏆🎮".to_string(),
                },
                RankTier {
                    rank_id: 100,
                    rank_name: "World Champion".to_string(),
                    rank_display_name: "World Cup Champion".to_string(),
                    requirements: RankRequirements {
                        min_profits: 100000.0,
                        min_successful_trades: 1000,
                        min_win_rate: 0.9,
                        special_requirements: vec!["Win the Trading World Cup".to_string()],
                    },
                    perks: vec![RankPerk::WorldChampion, RankPerk::LegendaryStatus],
                    unlock_message: "WORLD CHAMPION! You've won the Trading World Cup! 🌍👑".to_string(),
                    rank_badge: "🌍👑".to_string(),
                },
            ],
            theme_lingo: ThemeLingo {
                rank_up: "Level up! Battle pass progress!".to_string(),
                good_trade: "Victory Royale! Another W!".to_string(),
                bad_trade: "You got sent back to the lobby!".to_string(),
                achievement: "Achievement unlocked, legend!".to_string(),
                leaderboard: "Arena Leaderboard".to_string(),
            },
            rank_colors: [
                ("Default Skin".to_string(), "#808080".to_string()), // Gray
                ("Battle Pass".to_string(), "#00FF00".to_string()), // Green
                ("Sweaty Builder".to_string(), "#FF4500".to_string()), // Orange
                ("Pro Player".to_string(), "#FF0000".to_string()), // Red
                ("World Champion".to_string(), "#FFD700".to_string()), // Gold
            ].into_iter().collect(),
            rank_emojis: [
                ("Default Skin".to_string(), "⛏️".to_string()),
                ("Battle Pass".to_string(), "✨".to_string()),
                ("Sweaty Builder".to_string(), "🏗️".to_string()),
                ("Pro Player".to_string(), "🏆".to_string()),
                ("World Champion".to_string(), "👑".to_string()),
            ].into_iter().collect(),
            progression_messages: vec![
                "GG! You've reached {rank} in the battle royale! 🏆".to_string(),
                "Victory Royale! {rank} status achieved! 👑".to_string(),
                "W in the chat! {rank} unlocked! 🎉".to_string(),
            ],
            theme_sounds: vec!["victory_royale.wav", "level_up.wav", "chest_open.wav"].into_iter().map(|s| s.to_string()).collect(),
        });
        
        themes
    }
    
    /// **GET USER'S CURRENT RANK INFO**
    pub fn get_user_rank_info(&self, user_id: &str) -> Option<UserRankInfo> {
        if let Some(user_rank) = self.user_rankings.get(user_id) {
            if let Some(theme_data) = self.available_themes.get(&user_rank.chosen_theme) {
                if let Some(current_tier) = theme_data.ranks.iter().find(|r| r.rank_id == user_rank.current_rank) {
                    return Some(UserRankInfo {
                        theme: user_rank.chosen_theme.clone(),
                        current_rank: current_tier.clone(),
                        progress: user_rank.rank_progress.clone(),
                        next_rank: self.get_next_rank(&user_rank.chosen_theme, user_rank.current_rank),
                        progress_to_next: self.calculate_progress_to_next(user_rank),
                        prestige_level: user_rank.prestige_level,
                    });
                }
            }
        }
        None
    }
    
    /// **UPDATE USER PROGRESS**
    pub fn update_user_progress(&mut self, user_id: &str, trade_result: TradeResult) -> Option<RankUpdate> {
        if let Some(user_rank) = self.user_rankings.get_mut(user_id) {
            // Update progress stats
            match trade_result.outcome {
                TradeOutcome::Win(profit) => {
                    user_rank.rank_progress.total_profits += profit;
                    user_rank.rank_progress.successful_trades += 1;
                    if profit > user_rank.rank_progress.biggest_win {
                        user_rank.rank_progress.biggest_win = profit;
                    }
                },
                TradeOutcome::Loss(_) => {
                    // Loss stats handled elsewhere
                },
            }
            
            user_rank.rank_progress.win_rate = user_rank.rank_progress.successful_trades as f64 / (user_rank.rank_progress.successful_trades + trade_result.total_trades - user_rank.rank_progress.successful_trades) as f64;
            
            // Check for rank up
            return self.check_rank_up(user_id);
        }
        None
    }
    
    /// **GET RANK UP MESSAGE**
    pub fn get_rank_up_message(&self, user_id: &str, old_rank: u32, new_rank: u32, theme: &RankTheme) -> String {
        if let Some(theme_data) = self.available_themes.get(theme) {
            if let Some(new_tier) = theme_data.ranks.iter().find(|r| r.rank_id == new_rank) {
                let themed_message = match theme {
                    RankTheme::RuneScape => {
                        format!("🎉 **DING! LEVEL UP!**\n\n🎽 Congratulations! Your Trading level is now {}!\n⚔️ {}\n🏆 New perks unlocked!\n\n💪 Keep grinding those gains!", 
                                new_tier.rank_display_name, new_tier.unlock_message)
                    },
                    RankTheme::CallOfDuty => {
                        format!("🪖 **PROMOTION EARNED!**\n\n⭐ You've been promoted to {}!\n🎖️ {}\n🫡 Outstanding service, soldier!\n\n💪 Hoorah!", 
                                new_tier.rank_display_name, new_tier.unlock_message)
                    },
                    RankTheme::Crypto => {
                        format!("💎 **HANDS GETTING STRONGER!**\n\n🙌 You've reached {} status!\n🚀 {}\n💪 HODL gang rise up!\n\n🌙 Wen lambo?", 
                                new_tier.rank_display_name, new_tier.unlock_message)
                    },
                    RankTheme::Fortnite => {
                        format!("🏆 **VICTORY ROYALE!**\n\n👑 You've achieved {} status!\n✨ {}\n🎉 GG! Another W in the books!\n\n💪 Keep cranking those 90s!", 
                                new_tier.rank_display_name, new_tier.unlock_message)
                    },
                    _ => {
                        format!("🎉 **RANK UP!**\n\n🌟 Welcome to {}!\n{}\n🏆 New perks unlocked!", 
                                new_tier.rank_display_name, new_tier.unlock_message)
                    }
                };
                
                return themed_message;
            }
        }
        
        "🎉 Congratulations! You've ranked up!".to_string()
    }
    
    /// **GET THEMED LEADERBOARD**
    pub fn get_themed_leaderboard(&self, theme: &RankTheme, limit: usize) -> Vec<LeaderboardEntry> {
        if let Some(leaderboard) = self.global_leaderboards.get(theme) {
            leaderboard.iter().take(limit).cloned().collect()
        } else {
            vec![]
        }
    }
    
    // Helper methods
    fn get_next_rank(&self, theme: &RankTheme, current_rank: u32) -> Option<RankTier> {
        if let Some(theme_data) = self.available_themes.get(theme) {
            theme_data.ranks.iter().find(|r| r.rank_id > current_rank).cloned()
        } else {
            None
        }
    }
    
    fn calculate_progress_to_next(&self, user_rank: &UserRank) -> f64 {
        // Calculate percentage progress to next rank based on requirements
        // This would be a complex calculation based on multiple factors
        0.65 // Placeholder
    }
    
    fn check_rank_up(&mut self, user_id: &str) -> Option<RankUpdate> {
        // Check if user meets requirements for next rank
        // Return RankUpdate if they rank up
        None // Placeholder
    }
}

// Supporting types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserRankInfo {
    pub theme: RankTheme,
    pub current_rank: RankTier,
    pub progress: RankProgress,
    pub next_rank: Option<RankTier>,
    pub progress_to_next: f64,
    pub prestige_level: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankRequirements {
    pub min_profits: f64,
    pub min_successful_trades: u32,
    pub min_win_rate: f64,
    pub special_requirements: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RankPerk {
    BasicTrading,
    AdvancedAnalytics,
    CustomStrategies,
    DragonWeapons,
    MaxCape,
    UnlimitedTrades,
    GodMode,
    BasicTraining,
    SquadLeader,
    OfficerPrivileges,
    AdvancedIntel,
    CommandAuthority,
    SpecialOperations,
    SupremeCommand,
    LegendaryStatus,
    HodlStrength,
    DiamondFormation,
    DiamondStatus,
    MoonMission,
    WhaleStatus,
    MarketMover,
    BasicPickaxe,
    CustomSkin,
    CrankedBuilds,
    SweatMode,
    ProStatus,
    TournamentAccess,
    WorldChampion,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThemeLingo {
    pub rank_up: String,
    pub good_trade: String,
    pub bad_trade: String,
    pub achievement: String,
    pub leaderboard: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaderboardEntry {
    pub user_id: String,
    pub username: String,
    pub rank: u32,
    pub total_profits: f64,
    pub rank_badge: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankHistoryEntry {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub old_rank: u32,
    pub new_rank: u32,
    pub trigger_event: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Achievement {
    pub achievement_id: String,
    pub name: String,
    pub description: String,
    pub unlocked_at: chrono::DateTime<chrono::Utc>,
    pub rarity: AchievementRarity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AchievementRarity {
    Common,
    Rare,
    Epic,
    Legendary,
    Mythic,
}

#[derive(Debug, Clone)]
pub struct TradeResult {
    pub outcome: TradeOutcome,
    pub total_trades: u32,
}

#[derive(Debug, Clone)]
pub enum TradeOutcome {
    Win(f64), // Profit amount
    Loss(f64), // Loss amount
}

#[derive(Debug, Clone)]
pub struct RankUpdate {
    pub old_rank: u32,
    pub new_rank: u32,
    pub theme: RankTheme,
    pub perks_unlocked: Vec<RankPerk>,
}

impl RankRequirement {
    // Implementation would go here
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankRequirement; // Placeholder