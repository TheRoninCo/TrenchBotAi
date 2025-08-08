use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// **TRENCHWARE GAMING UI SYSTEM**
/// Familiar gaming lingo that normies actually understand
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrenchwareUI {
    pub gaming_translations: HashMap<String, GamingTranslation>,
    pub cultural_references: HashMap<String, CulturalRef>,
    pub tooltip_system: TooltipSystem,
    pub status_messages: StatusMessages,
    pub alert_sounds: AlertSounds,
}

/// **GAMING TRANSLATION SYSTEM**
/// Converts boring finance terms into gaming language everyone knows
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GamingTranslation {
    pub original_term: String,
    pub gaming_term: String,
    pub emoji: String,
    pub explanation: String,
    pub example: String,
    pub cultural_reference: Option<String>,
}

/// **CULTURAL REFERENCES**
/// Pop culture that resonates with the target audience
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CulturalRef {
    pub reference: String,
    pub context: String,
    pub when_to_use: String,
}

impl TrenchwareUI {
    pub fn new() -> Self {
        Self {
            gaming_translations: Self::create_gaming_dictionary(),
            cultural_references: Self::create_cultural_references(),
            tooltip_system: TooltipSystem::new(),
            status_messages: StatusMessages::new(),
            alert_sounds: AlertSounds::new(),
        }
    }
    
    /// **GAMING DICTIONARY**
    /// Translation table from finance speak to gaming speak
    fn create_gaming_dictionary() -> HashMap<String, GamingTranslation> {
        let mut translations = HashMap::new();
        
        // === TRADING ACTIONS ===
        translations.insert("buy_token".to_string(), GamingTranslation {
            original_term: "Purchase Token".to_string(),
            gaming_term: "Spawn Kill".to_string(),
            emoji: "🎯".to_string(),
            explanation: "Jump in right when the token launches - like spawn killing noobs".to_string(),
            example: "Spawn killed PEPE at $0.0001 - perfect timing!".to_string(),
            cultural_reference: Some("Call of Duty spawn camping".to_string()),
        });
        
        translations.insert("sell_token".to_string(), GamingTranslation {
            original_term: "Sell Token".to_string(),
            gaming_term: "Take The W".to_string(),
            emoji: "🏆".to_string(),
            explanation: "Cash out your profits - secure the victory".to_string(),
            example: "Taking the W on DOGE +150% - Victory Royale!".to_string(),
            cultural_reference: Some("Fortnite Victory Royale".to_string()),
        });
        
        translations.insert("stop_loss".to_string(), GamingTranslation {
            original_term: "Stop Loss Triggered".to_string(),
            gaming_term: "Respawn Incoming".to_string(),
            emoji: "💀".to_string(),
            explanation: "You got rekt, but you'll be back - that's why we set stop losses".to_string(),
            example: "Got sniped on SHIB -15%, respawning with better position".to_string(),
            cultural_reference: Some("Every FPS respawn system".to_string()),
        });
        
        // === PROFIT LEVELS ===
        translations.insert("small_profit".to_string(), GamingTranslation {
            original_term: "Small Profit ($10-50)".to_string(),
            gaming_term: "Happy Meal Money".to_string(),
            emoji: "🍟".to_string(),
            explanation: "Small gains - enough for a McDonald's run".to_string(),
            example: "Made happy meal money on that trade - $23 profit".to_string(),
            cultural_reference: Some("McDonald's Happy Meal price".to_string()),
        });
        
        translations.insert("medium_profit".to_string(), GamingTranslation {
            original_term: "Medium Profit ($50-200)".to_string(),
            gaming_term: "Gas Money".to_string(),
            emoji: "⛽".to_string(),
            explanation: "Decent profit - covers your gas for the week".to_string(),
            example: "Got gas money from BONK - $85 in the bag".to_string(),
            cultural_reference: Some("Real world gas prices pain".to_string()),
        });
        
        translations.insert("large_profit".to_string(), GamingTranslation {
            original_term: "Large Profit ($200-1000)".to_string(),
            gaming_term: "Rent Money".to_string(),
            emoji: "🏠".to_string(),
            explanation: "Big win - covers your monthly rent".to_string(),
            example: "Rent money secured! $750 profit on memecoin snipe".to_string(),
            cultural_reference: Some("Monthly rent struggle".to_string()),
        });
        
        translations.insert("huge_profit".to_string(), GamingTranslation {
            original_term: "Massive Profit ($1000+)".to_string(),
            gaming_term: "Lambo Fund".to_string(),
            emoji: "🏎️".to_string(),
            explanation: "Massive gains - you're building that Lamborghini fund".to_string(),
            example: "LAMBO FUND ACTIVATED! $3,500 profit - we're cooking!".to_string(),
            cultural_reference: Some("Crypto 'wen lambo' meme".to_string()),
        });
        
        // === MARKET CONDITIONS ===
        translations.insert("whale_detected".to_string(), GamingTranslation {
            original_term: "Large Wallet Movement".to_string(),
            gaming_term: "AC-130 Inbound".to_string(),
            emoji: "✈️".to_string(),
            explanation: "Big player moving serious money - incoming market impact".to_string(),
            example: "AC-130 inbound! Whale moved 50M tokens - brace for impact".to_string(),
            cultural_reference: Some("Call of Duty killstreak".to_string()),
        });
        
        translations.insert("market_crash".to_string(), GamingTranslation {
            original_term: "Market Downturn".to_string(),
            gaming_term: "Tactical Nuke Incoming".to_string(),
            emoji: "☢️".to_string(),
            explanation: "Everything's getting wiped - take cover!".to_string(),
            example: "TACTICAL NUKE INCOMING! Market down 20% - all positions defensive".to_string(),
            cultural_reference: Some("MW2 tactical nuke".to_string()),
        });
        
        translations.insert("pump_detected".to_string(), GamingTranslation {
            original_term: "Price Increase".to_string(),
            gaming_term: "Going Nuclear".to_string(),
            emoji: "🚀".to_string(),
            explanation: "Price is mooning hard - this is the pump you've been waiting for".to_string(),
            example: "PEPE going nuclear! +400% and climbing - rockets engaged".to_string(),
            cultural_reference: Some("'Going nuclear' intensity".to_string()),
        });
        
        // === TRADING EXECUTION ===
        translations.insert("trade_executed".to_string(), GamingTranslation {
            original_term: "Order Filled".to_string(),
            gaming_term: "Target Eliminated".to_string(),
            emoji: "💥".to_string(),
            explanation: "Trade went through perfectly - target acquired and eliminated".to_string(),
            example: "Target eliminated! Bought 1000 DOGE at perfect entry".to_string(),
            cultural_reference: Some("FPS target elimination".to_string()),
        });
        
        translations.insert("perfect_timing".to_string(), GamingTranslation {
            original_term: "Optimal Entry/Exit".to_string(),
            gaming_term: "Headshot".to_string(),
            emoji: "🎯".to_string(),
            explanation: "Perfect timing - one shot, one kill precision".to_string(),
            example: "HEADSHOT! Bought the exact bottom and sold the exact top".to_string(),
            cultural_reference: Some("FPS headshot mechanics".to_string()),
        });
        
        translations.insert("lucky_trade".to_string(), GamingTranslation {
            original_term: "Unexpected Success".to_string(),
            gaming_term: "No Scope".to_string(),
            emoji: "🔫".to_string(),
            explanation: "Blind trade that somehow worked - pure luck but we'll take it".to_string(),
            example: "NO SCOPE! Bought random memecoin, up 200% - didn't even research it".to_string(),
            cultural_reference: Some("COD no-scope sniper shots".to_string()),
        });
        
        // === PERFORMANCE METRICS ===
        translations.insert("win_rate".to_string(), GamingTranslation {
            original_term: "Success Rate".to_string(),
            gaming_term: "K/D Ratio".to_string(),
            emoji: "📊".to_string(),
            explanation: "Kill/Death ratio - how many wins vs losses you have".to_string(),
            example: "Your K/D ratio: 3.2 - you're eliminating 3 bad trades for every loss".to_string(),
            cultural_reference: Some("Every FPS game ever".to_string()),
        });
        
        translations.insert("portfolio_analysis".to_string(), GamingTranslation {
            original_term: "Portfolio Review".to_string(),
            gaming_term: "Kill Cam Review".to_string(),
            emoji: "📹".to_string(),
            explanation: "Replay of exactly how your trades went down - learn from wins and losses".to_string(),
            example: "Kill cam shows you bought SHIB right before Elon tweeted - perfect timing!".to_string(),
            cultural_reference: Some("COD kill cam replays".to_string()),
        });
        
        // === RISK MANAGEMENT ===
        translations.insert("high_risk".to_string(), GamingTranslation {
            original_term: "High Risk Trade".to_string(),
            gaming_term: "YOLO Play".to_string(),
            emoji: "🎰".to_string(),
            explanation: "High risk, high reward - you only live once".to_string(),
            example: "YOLO play on new memecoin - could 10x or could go to zero".to_string(),
            cultural_reference: Some("YOLO meme culture".to_string()),
        });
        
        translations.insert("safe_trade".to_string(), GamingTranslation {
            original_term: "Conservative Trade".to_string(),
            gaming_term: "Playing it Safe".to_string(),
            emoji: "🛡️".to_string(),
            explanation: "Low risk trade - not flashy but keeps your account alive".to_string(),
            example: "Playing it safe with BTC - slow and steady wins the race".to_string(),
            cultural_reference: Some("Camping strategies in FPS".to_string()),
        });
        
        // === SYSTEM STATUS ===
        translations.insert("bot_active".to_string(), GamingTranslation {
            original_term: "System Online".to_string(),
            gaming_term: "Locked and Loaded".to_string(),
            emoji: "🔥".to_string(),
            explanation: "Bot is ready for action - all systems go".to_string(),
            example: "TrenchBot locked and loaded - ready to hunt some profits".to_string(),
            cultural_reference: Some("Military readiness".to_string()),
        });
        
        translations.insert("emergency_stop".to_string(), GamingTranslation {
            original_term: "Emergency Stop Activated".to_string(),
            gaming_term: "Cease Fire".to_string(),
            emoji: "🛑".to_string(),
            explanation: "All trading stopped immediately - safety first".to_string(),
            example: "CEASE FIRE! Emergency stop activated - all positions secure".to_string(),
            cultural_reference: Some("Military cease fire commands".to_string()),
        });

        translations
    }
    
    /// **CULTURAL REFERENCES**
    /// Pop culture that everyone gets
    fn create_cultural_references() -> HashMap<String, CulturalRef> {
        let mut refs = HashMap::new();
        
        refs.insert("moon_mission".to_string(), CulturalRef {
            reference: "🚀 TO THE MOON!".to_string(),
            context: "When a token is pumping hard".to_string(),
            when_to_use: "Any time profits are going up fast".to_string(),
        });
        
        refs.insert("diamond_hands".to_string(), CulturalRef {
            reference: "💎🙌 DIAMOND HANDS".to_string(),
            context: "Holding through dips - not selling".to_string(),
            when_to_use: "When user doesn't panic sell during red candles".to_string(),
        });
        
        refs.insert("paper_hands".to_string(), CulturalRef {
            reference: "📄🙌 Paper Hands".to_string(),
            context: "Selling too early out of fear".to_string(),
            when_to_use: "When user sells right before a pump".to_string(),
        });
        
        refs.insert("ape_in".to_string(), CulturalRef {
            reference: "🦍 APE IN!".to_string(),
            context: "Buy without thinking - FOMO buy".to_string(),
            when_to_use: "When making aggressive entries".to_string(),
        });
        
        refs.insert("rekt".to_string(), CulturalRef {
            reference: "☠️ GET REKT".to_string(),
            context: "Got destroyed by the market".to_string(),
            when_to_use: "When a trade goes very bad".to_string(),
        });
        
        refs.insert("stonks".to_string(), CulturalRef {
            reference: "📈 STONKS".to_string(),
            context: "Stocks/crypto only go up (meme)".to_string(),
            when_to_use: "When celebrating any green candles".to_string(),
        });

        refs
    }
    
    /// **GET GAMING MESSAGE**
    /// Translate boring messages into gaming lingo
    pub fn get_gaming_message(&self, event_type: &str, context: MessageContext) -> String {
        match event_type {
            "trade_success" => {
                let profit_level = self.categorize_profit(context.profit_amount.unwrap_or(0.0));
                match profit_level.as_str() {
                    "happy_meal" => format!("🍟 **HAPPY MEAL MONEY SECURED!**\n💰 Profit: ${:.2}\n🎯 Target eliminated with precision!", 
                                           context.profit_amount.unwrap_or(0.0)),
                    "gas_money" => format!("⛽ **GAS MONEY IN THE BAG!**\n💰 Profit: ${:.2}\n🚗 Tank's getting filled this week!", 
                                          context.profit_amount.unwrap_or(0.0)),
                    "rent_money" => format!("🏠 **RENT MONEY SECURED!**\n💰 Profit: ${:.2}\n🏆 Landlord's gonna be happy!", 
                                           context.profit_amount.unwrap_or(0.0)),
                    "lambo_fund" => format!("🏎️ **LAMBO FUND ACTIVATED!**\n💰 Profit: ${:.2}\n🚀 We're actually cooking now!", 
                                           context.profit_amount.unwrap_or(0.0)),
                    _ => format!("💰 Profit secured: ${:.2}", context.profit_amount.unwrap_or(0.0))
                }
            },
            
            "whale_alert" => {
                format!("✈️ **AC-130 INBOUND!**\n\
                        🐋 Whale moving: ${:.0}\n\
                        📍 Token: {}\n\
                        ⚠️ Incoming market impact - brace for volatility!\n\
                        🎯 TrenchBot analyzing trajectory...", 
                        context.amount.unwrap_or(0.0),
                        context.token_name.as_ref().unwrap_or(&"Unknown".to_string()))
            },
            
            "perfect_entry" => {
                format!("🎯 **HEADSHOT!**\n\
                        🔥 Perfect entry at ${:.6}\n\
                        📊 Precision: 99.8%\n\
                        💀 Target acquired and eliminated\n\
                        🏆 One shot, one kill!", 
                        context.entry_price.unwrap_or(0.0))
            },
            
            "stop_loss_hit" => {
                format!("💀 **RESPAWN INCOMING**\n\
                        📉 Got sniped: -{:.1}%\n\
                        🛡️ Stop loss saved you from bigger L\n\
                        ⚡ Respawning in 3... 2... 1...\n\
                        🎮 Back to the lobby - better luck next round!", 
                        context.loss_percent.unwrap_or(0.0))
            },
            
            "market_dump" => {
                format!("☢️ **TACTICAL NUKE INCOMING!**\n\
                        📉 Market down: -{:.1}%\n\
                        🏃‍♂️ TAKE COVER! Everyone's getting rekt\n\
                        🛡️ TrenchBot activating defensive positions\n\
                        💎 Diamond hands mode: ENGAGED")
            },
            
            "memecoin_pump" => {
                format!("🚀 **TARGET IS GOING NUCLEAR!**\n\
                        📈 {} pumping: +{:.1}%\n\
                        🔥 This is not a drill - rockets engaged!\n\
                        🌙 Destination: THE MOON\n\
                        💎 Hold the line, soldiers!", 
                        context.token_name.as_ref().unwrap_or(&"Token".to_string()),
                        context.pump_percent.unwrap_or(0.0))
            },
            
            "scan_complete" => {
                let tokens_found = context.tokens_scanned.unwrap_or(0);
                format!("🔍 **RECON COMPLETE**\n\
                        📊 Scanned {} targets\n\
                        🎯 {} high-value targets identified\n\
                        ⚡ AC-130 standing by for engagement\n\
                        💀 Permission to engage?", 
                        tokens_found,
                        tokens_found / 10) // Rough estimate of good targets
            },
            
            "bot_startup" => {
                format!("🔥 **LOCKED AND LOADED**\n\
                        🤖 TrenchBot systems: ONLINE\n\
                        🎯 Targeting systems: ARMED\n\
                        📡 Market radar: ACTIVE\n\
                        💀 Ready to eliminate some profits!\n\
                        🚀 LET'S GET THIS BREAD!")
            },
            
            _ => format!("🎮 TrenchBot event: {}", event_type)
        }
    }
    
    /// **CATEGORIZE PROFIT**
    /// Determine what level of profit this is
    fn categorize_profit(&self, amount: f64) -> String {
        match amount {
            x if x < 50.0 => "happy_meal".to_string(),
            x if x < 200.0 => "gas_money".to_string(),
            x if x < 1000.0 => "rent_money".to_string(),
            _ => "lambo_fund".to_string(),
        }
    }
    
    /// **GET TOOLTIP WITH GAMING CONTEXT**
    /// Explain gaming terms for normies
    pub fn get_gaming_tooltip(&self, term: &str) -> Option<String> {
        if let Some(translation) = self.gaming_translations.get(term) {
            Some(format!("{} {}\n\n💡 **What this means:**\n{}\n\n📝 **Example:**\n{}\n\n🎮 **Gaming Reference:**\n{}", 
                        translation.emoji,
                        translation.gaming_term,
                        translation.explanation,
                        translation.example,
                        translation.cultural_reference.as_ref().unwrap_or(&"Gaming culture".to_string())
            ))
        } else {
            None
        }
    }
    
    /// **GET STATUS MESSAGE**
    /// System status in gaming terms
    pub fn get_status_message(&self, system_health: SystemHealth) -> String {
        match system_health {
            SystemHealth::Excellent => {
                format!("🔥 **LOCKED AND LOADED**\n\
                        ✅ All systems: ONLINE\n\
                        📡 Connection: SOLID\n\
                        🎯 Targeting: ARMED\n\
                        💀 Ready to secure the bag!")
            },
            SystemHealth::Good => {
                format!("✅ **SYSTEMS OPERATIONAL**\n\
                        🟢 Status: Good to go\n\
                        📊 Performance: Solid\n\
                        🎯 Ready for engagement")
            },
            SystemHealth::Warning => {
                format!("⚠️ **MINOR TECHNICAL ISSUES**\n\
                        🟡 Some lag detected\n\
                        🔧 Running diagnostics\n\
                        📶 Connection unstable")
            },
            SystemHealth::Critical => {
                format!("🚨 **SYSTEM MALFUNCTION**\n\
                        🔴 Critical issues detected\n\
                        🛠️ Emergency repairs needed\n\
                        ⛔ Trading temporarily disabled")
            },
        }
    }
}

/// **SUPPORTING TYPES**
#[derive(Debug, Clone)]
pub struct MessageContext {
    pub profit_amount: Option<f64>,
    pub loss_percent: Option<f64>,
    pub token_name: Option<String>,
    pub amount: Option<f64>,
    pub entry_price: Option<f64>,
    pub pump_percent: Option<f64>,
    pub tokens_scanned: Option<u32>,
}

#[derive(Debug, Clone)]
pub enum SystemHealth {
    Excellent,
    Good,
    Warning,
    Critical,
}

/// **TOOLTIP SYSTEM**
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TooltipSystem {
    pub show_gaming_explanations: bool,
    pub show_traditional_terms: bool,
    pub tooltip_delay_ms: u32,
}

impl TooltipSystem {
    fn new() -> Self {
        Self {
            show_gaming_explanations: true,
            show_traditional_terms: true,
            tooltip_delay_ms: 500,
        }
    }
}

/// **STATUS MESSAGES**
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusMessages {
    pub use_gaming_language: bool,
    pub include_emojis: bool,
    pub cultural_reference_level: CulturalLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CulturalLevel {
    Normie,      // Basic references everyone gets
    Degen,       // Deep crypto/gaming culture
    GigaChad,    // Maximum meme energy
}

impl StatusMessages {
    fn new() -> Self {
        Self {
            use_gaming_language: true,
            include_emojis: true,
            cultural_reference_level: CulturalLevel::Degen,
        }
    }
}

/// **ALERT SOUNDS**
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertSounds {
    pub enable_sound_effects: bool,
    pub volume_level: f32,
    pub sound_pack: SoundPack,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SoundPack {
    Gaming,      // FPS sound effects
    Retro,       // 8-bit arcade sounds
    Modern,      // Clean notification sounds
    Meme,        // Meme sound effects
}

impl AlertSounds {
    fn new() -> Self {
        Self {
            enable_sound_effects: true,
            volume_level: 0.7,
            sound_pack: SoundPack::Gaming,
        }
    }
}

/// **EXAMPLE USAGE**
impl TrenchwareUI {
    pub fn demo_messages() -> Vec<String> {
        let ui = TrenchwareUI::new();
        
        vec![
            ui.get_gaming_message("trade_success", MessageContext {
                profit_amount: Some(25.50),
                loss_percent: None,
                token_name: Some("PEPE".to_string()),
                amount: None,
                entry_price: None,
                pump_percent: None,
                tokens_scanned: None,
            }),
            
            ui.get_gaming_message("whale_alert", MessageContext {
                profit_amount: None,
                loss_percent: None,
                token_name: Some("DOGE".to_string()),
                amount: Some(5000000.0),
                entry_price: None,
                pump_percent: None,
                tokens_scanned: None,
            }),
            
            ui.get_gaming_message("memecoin_pump", MessageContext {
                profit_amount: None,
                loss_percent: None,
                token_name: Some("SHIB".to_string()),
                amount: None,
                entry_price: None,
                pump_percent: Some(450.0),
                tokens_scanned: None,
            }),
        ]
    }
}