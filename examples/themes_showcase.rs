//! Multi-Theme Warfare Showcase
//! 
//! Demonstrates all available warfare themes in TrenchBot

use tokio::time::{sleep, Duration};

// Simplified theme system for demo
#[derive(Debug, Clone, PartialEq)]
pub enum WarfareTheme {
    RuneScape,
    Military, 
    Pirate,
    StreetFighter,
    CallOfDuty,
    Cyberpunk,
    Medieval,
    SpaceWar,
}

impl WarfareTheme {
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

    pub fn all() -> Vec<Self> {
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

    pub fn kill_message(&self, profit: f64) -> String {
        match self {
            WarfareTheme::RuneScape => {
                let monster = if profit > 500.0 { "🖤 King Black Dragon" } 
                             else if profit > 100.0 { "🐲 Dragon" }
                             else if profit > 50.0 { "🗿 Troll" }
                             else if profit > 10.0 { "🧌 Orc" } 
                             else { "👹 Goblin" };
                format!("🗡️ {} DEFEATED! {} GP | Combat Lvl: 🔴 42", monster, (profit * 1000.0) as u64)
            },
            WarfareTheme::Military => {
                let target = if profit > 500.0 { "🏢 Command Center" }
                           else if profit > 100.0 { "🚗 Tank" }
                           else if profit > 25.0 { "🔫 Squad Leader" }
                           else { "🎯 Insurgent" };
                format!("💥 {} ELIMINATED! Rank: 🌟 Lieutenant", target)
            },
            WarfareTheme::Pirate => {
                let treasure = if profit > 500.0 { "🏴‍☠️ Legendary Treasure" }
                             else if profit > 150.0 { "⛵ Galleon" }
                             else if profit > 50.0 { "💰 Treasure Chest" }
                             else { "🚢 Merchant Ship" };
                format!("🏴‍☠️ {} PLUNDERED! Rank: ⚔️ Commodore", treasure)
            },
            WarfareTheme::StreetFighter => {
                let opponent = if profit > 500.0 { "👹 Final Boss" }
                             else if profit > 100.0 { "💪 Tournament Boss" }
                             else if profit > 25.0 { "👊 Rival Fighter" }
                             else { "🥊 Street Thug" };
                format!("💥 {} K.O.! Rank: 🏆 Champion", opponent)
            },
            WarfareTheme::CallOfDuty => {
                let enemy = if profit > 500.0 { "✈️ AC-130" }
                          else if profit > 100.0 { "🚁 Attack Helicopter" }
                          else if profit > 25.0 { "🔫 Sniper" }
                          else { "🎯 Enemy Soldier" };
                format!("🔥 {} ENEMY DOWN! Rank: 🌟 Lieutenant | 7 Killstreak", enemy)
            },
            WarfareTheme::Cyberpunk => {
                let target = if profit > 500.0 { "🏢 Megacorp Mainframe" }
                           else if profit > 100.0 { "👔 Corpo Executive" }
                           else if profit > 25.0 { "🛡️ Security IC" }
                           else { "💼 Corp Shill" };
                format!("⚡ {} SYSTEM BREACHED! Rank: 🤠 Console Cowboy", target)
            },
            WarfareTheme::Medieval => {
                let foe = if profit > 500.0 { "🐉 Dragon" }
                        else if profit > 200.0 { "🏰 Castle Siege" }
                        else if profit > 50.0 { "⚔️ Enemy Knight" }
                        else { "🗡️ Bandit" };
                format!("⚔️ {} VANQUISHED! Rank: 🏰 Baron", foe)
            },
            WarfareTheme::SpaceWar => {
                let vessel = if profit > 500.0 { "💫 Death Star" }
                           else if profit > 150.0 { "🛰️ Battlecruiser" }
                           else if profit > 50.0 { "🚀 Frigate" }
                           else { "🛸 Scout Ship" };
                format!("💥 {} TARGET DESTROYED! Rank: ⭐ Admiral", vessel)
            },
        }
    }

    pub fn level_up_message(&self) -> String {
        match self {
            WarfareTheme::RuneScape => "🎉 Level Up! Rug Pulling ⚔️ is now level 25!".to_string(),
            WarfareTheme::Military => "🎖️ PROMOTED! You are now Lieutenant!".to_string(),
            WarfareTheme::Pirate => "⚓ PROMOTED ABOARD! You are now First Mate!".to_string(),
            WarfareTheme::StreetFighter => "🥋 SKILL MASTERED! You learned Dragon Punch!".to_string(),
            WarfareTheme::CallOfDuty => "📈 RANK UP! You are now Sergeant!".to_string(),
            WarfareTheme::Cyberpunk => "🧠 NEURAL UPGRADE! Enhanced hacking protocols loaded".to_string(),
            WarfareTheme::Medieval => "🏆 HONOR GAINED! You are now a Knight-Captain!".to_string(),
            WarfareTheme::SpaceWar => "🌟 PROMOTION! You are now Fleet Commander!".to_string(),
        }
    }

    pub fn special_unlock(&self) -> String {
        match self {
            WarfareTheme::RuneScape => "⚔️ SPECIAL ATTACK UNLOCKED: 🗡️ Dragon Dagger!".to_string(),
            WarfareTheme::Military => "🔫 WEAPON UNLOCKED: 💥 Artillery Barrage!".to_string(),
            WarfareTheme::Pirate => "🗡️ WEAPON MASTERED: 💣 Cannon Barrage!".to_string(),
            WarfareTheme::StreetFighter => "🔥 COMBO UNLOCKED: 🐲 Dragon Punch!".to_string(),
            WarfareTheme::CallOfDuty => "🔥 KILLSTREAK REWARD: 📡 UAV Online!".to_string(),
            WarfareTheme::Cyberpunk => "🔧 EXPLOIT UNLOCKED: ⚡ Data Spike!".to_string(),
            WarfareTheme::Medieval => "🗡️ WEAPON MASTERY: ⚔️ Sword Strike!".to_string(),
            WarfareTheme::SpaceWar => "🔫 WEAPON SYSTEM ONLINE: 🚀 Ion Torpedo!".to_string(),
        }
    }

    pub fn victory_taunt(&self) -> String {
        match self {
            WarfareTheme::RuneScape => "💎 RARE DROP: 🟣 🗡️ Dragon Rug Pull Slayer (1/1000)".to_string(),
            WarfareTheme::Military => "🏆 MISSION ACCOMPLISHED! Target neutralized with precision".to_string(),
            WarfareTheme::Pirate => "🏴‍☠️ Their treasure is now ours, matey! Yarrr!".to_string(),
            WarfareTheme::StreetFighter => "💪 PERFECT! Flawless Victory achieved!".to_string(),
            WarfareTheme::CallOfDuty => "🎯 HEADSHOT! +50 XP Precision Bonus".to_string(),
            WarfareTheme::Cyberpunk => "🤖 Neural jack in complete. Data acquired".to_string(),
            WarfareTheme::Medieval => "🏰 For honor and glory! The realm is secured".to_string(),
            WarfareTheme::SpaceWar => "🌌 Another victory for the Galactic Empire!".to_string(),
        }
    }
}

async fn demonstrate_theme(theme: WarfareTheme, profits: &[f64]) {
    println!("\n" + &"=".repeat(50));
    println!("🎭 {} THEME DEMONSTRATION", theme.name().to_uppercase());
    println!("{}", "=".repeat(50));
    
    for (i, profit) in profits.iter().enumerate() {
        println!("\n🎯 Engagement {}: {:.1} SOL profit", i + 1, profit);
        
        // Kill message
        println!("   {}", theme.kill_message(*profit));
        
        // Occasional level up
        if i == 2 {
            println!("   {}", theme.level_up_message());
        }
        
        // Occasional special unlock
        if i == 4 {
            println!("   {}", theme.special_unlock());
        }
        
        // Victory taunt on big kills
        if *profit > 200.0 {
            println!("   {}", theme.victory_taunt());
        }
        
        sleep(Duration::from_millis(800)).await;
    }
}

#[tokio::main]
async fn main() {
    println!("🎮 TrenchBot Multi-Theme Warfare Showcase");
    println!("==========================================");
    println!("Demonstrating all available combat themes!\n");
    
    // Sample profits for demonstration
    let battle_profits = vec![5.0, 25.0, 75.0, 150.0, 350.0, 600.0];
    
    // Showcase each theme
    for theme in WarfareTheme::all() {
        demonstrate_theme(theme, &battle_profits).await;
        
        if theme != WarfareTheme::SpaceWar { // Don't wait after last theme
            println!("\n⏳ Switching themes in 2 seconds...");
            sleep(Duration::from_secs(2)).await;
        }
    }
    
    println!("\n" + &"🎊".repeat(20));
    println!("🎭 THEME SHOWCASE COMPLETE! 🎭");
    println!("{}", "🎊".repeat(20));
    
    println!("\n💡 Theme Selection Guide:");
    println!("   🗡️ RuneScape - Classic MMO nostalgia with levels & monsters");
    println!("   🎖️ Military - Professional tactical operations");
    println!("   🏴‍☠️ Pirate - Swashbuckling adventure on the high seas");
    println!("   🥊 Street Fighter - Tournament fighting action");
    println!("   🔫 Call of Duty - Modern warfare with killstreaks");
    println!("   🤖 Cyberpunk - Futuristic hacking and corpo warfare");
    println!("   ⚔️ Medieval - Knights and castles in honorable combat");
    println!("   🚀 Space War - Galactic conquest across the stars");
    
    println!("\n🎮 Users can switch themes with: !theme <name>");
    println!("⚡ All themes track the same progression stats");
    println!("🏆 Choose your favorite style and dominate the battlefield!");
}

/// Configuration example
fn show_config_example() {
    println!("\n📋 Configuration Example:");
    println!("```toml");
    println!("[warfare]");
    println!("theme = \"Military\"          # or RuneScape, Pirate, etc.");
    println!("show_level_ups = true");
    println!("show_rare_drops = true");  
    println!("combat_notifications = true");
    println!("victory_taunts = true");
    println!("```");
}