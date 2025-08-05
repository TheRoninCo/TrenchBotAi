#!/usr/bin/env python3
"""
TrenchBot MEV Telegram Alert Bot
Real-time MEV monitoring, alerts, and wallet management via Telegram
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import aiohttp
import subprocess
from pathlib import Path

# Telegram bot dependencies (install with: pip install python-telegram-bot[all])
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler, 
    MessageHandler, filters, ContextTypes
)

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class TrenchBotTelegramBot:
    def __init__(self, bot_token: str, authorized_users: List[int]):
        self.bot_token = bot_token
        self.authorized_users = set(authorized_users)
        self.app = Application.builder().token(bot_token).build()
        
        # MEV monitoring state
        self.mev_alerts_enabled = True
        self.whale_alerts_enabled = True
        self.profit_threshold = 0.01  # 1% minimum profit to alert
        self.last_alert_time = {}  # Rate limiting
        
        # Wallet management
        self.temp_wallets = {}
        self.funding_requests = {}
        
        # Performance monitoring
        self.last_performance_check = datetime.now()
        self.performance_data = {}
        
        self.setup_handlers()
    
    def setup_handlers(self):
        """Setup Telegram command handlers"""
        # Command handlers
        self.app.add_handler(CommandHandler("start", self.start_command))
        self.app.add_handler(CommandHandler("status", self.status_command))
        self.app.add_handler(CommandHandler("mev", self.mev_status_command))
        self.app.add_handler(CommandHandler("whales", self.whale_status_command))
        self.app.add_handler(CommandHandler("performance", self.performance_command))
        self.app.add_handler(CommandHandler("alerts", self.alerts_command))
        self.app.add_handler(CommandHandler("fund", self.fund_wallet_command))
        self.app.add_handler(CommandHandler("wallets", self.list_wallets_command))
        self.app.add_handler(CommandHandler("dashboard", self.dashboard_command))
        self.app.add_handler(CommandHandler("emergency", self.emergency_stop_command))
        
        # Callback query handler for inline keyboards
        self.app.add_handler(CallbackQueryHandler(self.button_callback))
        
        # Message handler for unauthorized access
        self.app.add_handler(MessageHandler(filters.TEXT, self.unauthorized_message))
    
    def is_authorized(self, user_id: int) -> bool:
        """Check if user is authorized"""
        return user_id in self.authorized_users
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Start command - welcome message"""
        user_id = update.effective_user.id
        
        if not self.is_authorized(user_id):
            await update.message.reply_text("🚫 Unauthorized access. Contact admin.")
            return
        
        welcome_msg = """
🤖 **TrenchBot MEV Alert System** 🤖

🎯 **Available Commands:**
• `/status` - System status overview
• `/mev` - MEV opportunities & stats  
• `/whales` - Whale activity monitoring
• `/performance` - AI performance metrics
• `/alerts` - Configure alert settings
• `/fund <wallet> <amount>` - Fund temp wallet
• `/wallets` - List managed wallets
• `/dashboard` - View web dashboard
• `/emergency` - Emergency stop all operations

⚡ **Real-time Monitoring Active**
🐋 Whale tracking: ON
💰 MEV scanning: ON
🚀 Performance: 122x optimized

Type any command to get started!
        """
        
        keyboard = [
            [InlineKeyboardButton("📊 MEV Status", callback_data="mev_status")],
            [InlineKeyboardButton("🐋 Whale Activity", callback_data="whale_activity")],
            [InlineKeyboardButton("⚡ Performance", callback_data="performance")],
            [InlineKeyboardButton("💰 Fund Wallet", callback_data="fund_wallet")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(welcome_msg, reply_markup=reply_markup, parse_mode='Markdown')
    
    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """System status overview"""
        if not self.is_authorized(update.effective_user.id):
            await update.message.reply_text("🚫 Unauthorized")
            return
        
        # Get system status
        status_data = await self.get_system_status()
        
        status_msg = f"""
🎯 **TrenchBot System Status**

🚀 **Performance:**
• Quantum: {status_data['quantum_latency']:.2f}ms
• Monte Carlo: {status_data['monte_carlo_latency']:.2f}ms  
• Overall Speedup: {status_data['speedup']}x

💰 **MEV Operations:**
• Opportunities Found: {status_data['mev_opportunities']}
• Successful Trades: {status_data['successful_trades']}
• Total Profit: ${status_data['total_profit']:.2f}

🐋 **Whale Activity:**
• Active Whales: {status_data['active_whales']}
• Large Transactions: {status_data['large_txs']}
• Whale Alerts: {status_data['whale_alerts']}

🏦 **Wallets:**
• Active Wallets: {len(self.temp_wallets)}
• Total Balance: ${status_data['total_balance']:.2f}
• Pending Funding: {len(self.funding_requests)}

⏰ Last Updated: {datetime.now().strftime('%H:%M:%S')}
        """
        
        keyboard = [
            [InlineKeyboardButton("🔄 Refresh", callback_data="refresh_status")],
            [InlineKeyboardButton("📈 Dashboard", callback_data="open_dashboard")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(status_msg, reply_markup=reply_markup, parse_mode='Markdown')
    
    async def mev_status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """MEV opportunities and statistics"""
        if not self.is_authorized(update.effective_user.id):
            return
        
        mev_data = await self.get_mev_status()
        
        mev_msg = f"""
⚡ **MEV Status Report**

🎯 **Current Opportunities:**
• Arbitrage: {mev_data['arbitrage_ops']} active
• Sandwich: {mev_data['sandwich_ops']} pending
• Liquidation: {mev_data['liquidation_ops']} ready

📊 **24h Statistics:**
• Total Opportunities: {mev_data['total_opportunities']}
• Success Rate: {mev_data['success_rate']:.1f}%
• Average Profit: ${mev_data['avg_profit']:.2f}
• Gas Efficiency: {mev_data['gas_efficiency']:.1f}%

🏆 **Top Performers:**
• Best Trade: +${mev_data['best_trade']:.2f}
• Best Token: {mev_data['best_token']} (+{mev_data['best_token_profit']:.1f}%)
• Most Active: {mev_data['most_active_pair']}

⚠️ **Risk Metrics:**
• Max Drawdown: {mev_data['max_drawdown']:.2f}%
• Sharpe Ratio: {mev_data['sharpe_ratio']:.2f}
• Win Rate: {mev_data['win_rate']:.1f}%
        """
        
        keyboard = [
            [InlineKeyboardButton("🔄 Refresh MEV", callback_data="refresh_mev")],
            [InlineKeyboardButton("⚠️ Set Alerts", callback_data="mev_alerts")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(mev_msg, reply_markup=reply_markup, parse_mode='Markdown')
    
    async def whale_status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Whale activity monitoring"""
        if not self.is_authorized(update.effective_user.id):
            return
        
        whale_data = await self.get_whale_activity()
        
        whale_msg = f"""
🐋 **Whale Activity Report**

🎯 **Active Whales (Last 1h):**
• Large Wallets: {whale_data['large_wallets']} active
• Mega Transactions: {whale_data['mega_txs']}
• Total Volume: ${whale_data['total_volume']:,.0f}

📈 **Whale Movements:**
• SOL Whales: {whale_data['sol_whales']} ({whale_data['sol_volume']:,.0f} SOL)
• Token Whales: {whale_data['token_whales']} active
• New Whales: {whale_data['new_whales']} detected

🎯 **Tracking Targets:**
• Followed Whales: {len(whale_data['followed_whales'])}
• Copy Trading: {whale_data['copy_trades']} positions
• Success Rate: {whale_data['copy_success_rate']:.1f}%

⚡ **Recent Activity:**
        """
        
        # Add recent whale activities
        for activity in whale_data['recent_activities'][:5]:
            whale_msg += f"• {activity['wallet'][:8]}... {activity['action']} {activity['amount']:,.0f} {activity['token']} ({activity['time']})\n"
        
        keyboard = [
            [InlineKeyboardButton("🔄 Refresh Whales", callback_data="refresh_whales")],
            [InlineKeyboardButton("🎯 Track New Whale", callback_data="track_whale")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(whale_msg, reply_markup=reply_markup, parse_mode='Markdown')
    
    async def performance_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """AI performance metrics"""
        if not self.is_authorized(update.effective_user.id):
            return
        
        # Get latest performance data
        perf_data = await self.get_performance_metrics()
        
        perf_msg = f"""
⚡ **AI Performance Metrics**

🚀 **System Performance:**
• Overall Speedup: {perf_data['overall_speedup']}x
• Total Latency: {perf_data['total_latency']:.2f}ms
• Status: {"🟢 EXCELLENT" if perf_data['total_latency'] < 100 else "🟡 GOOD"}

🧠 **Component Performance:**
• Quantum: {perf_data['quantum_ms']:.2f}ms ({perf_data['quantum_speedup']}x)
• Transformer: {perf_data['transformer_ms']:.2f}ms ({perf_data['transformer_speedup']}x)
• Monte Carlo: {perf_data['monte_carlo_ms']:.2f}ms ({perf_data['monte_carlo_speedup']}x)
• Graph Neural: {perf_data['graph_ms']:.2f}ms
• Competitive: {perf_data['competitive_ms']:.2f}ms

💾 **System Resources:**
• CPU Usage: {perf_data['cpu_usage']:.1f}%
• Memory: {perf_data['memory_usage']:.1f}%
• GPU Utilization: {perf_data['gpu_usage']:.1f}%

📊 **Trading Performance:**
• Decision Latency: {perf_data['decision_latency']:.2f}ms
• Execution Speed: {perf_data['execution_speed']:.2f}ms
• Orders/Second: {perf_data['orders_per_sec']:.0f}
        """
        
        keyboard = [
            [InlineKeyboardButton("🔄 Refresh Perf", callback_data="refresh_performance")],
            [InlineKeyboardButton("🧪 Run Benchmark", callback_data="run_benchmark")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(perf_msg, reply_markup=reply_markup, parse_mode='Markdown')
    
    async def fund_wallet_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Fund a temporary wallet"""
        if not self.is_authorized(update.effective_user.id):
            return
        
        args = context.args
        if len(args) != 2:
            await update.message.reply_text("""
💰 **Wallet Funding**

Usage: `/fund <wallet_address> <amount>`

Example: `/fund temp_wallet_1 100`

This will fund the specified wallet with the amount in SOL.
            """, parse_mode='Markdown')
            return
        
        wallet_address = args[0]
        try:
            amount = float(args[1])
        except ValueError:
            await update.message.reply_text("❌ Invalid amount. Please enter a number.")
            return
        
        # Create funding request
        request_id = f"fund_{int(time.time())}"
        self.funding_requests[request_id] = {
            'wallet': wallet_address,
            'amount': amount,
            'user_id': update.effective_user.id,
            'timestamp': datetime.now(),
            'status': 'pending'
        }
        
        # Simulate funding process (replace with actual Solana integration)
        funding_msg = f"""
💰 **Funding Request Created**

🏦 **Details:**
• Wallet: `{wallet_address}`
• Amount: {amount} SOL
• Request ID: `{request_id}`
• Status: ⏳ Pending

⚡ **Processing...**
This may take 30-60 seconds to complete.
        """
        
        keyboard = [
            [InlineKeyboardButton("✅ Confirm", callback_data=f"confirm_fund_{request_id}")],
            [InlineKeyboardButton("❌ Cancel", callback_data=f"cancel_fund_{request_id}")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(funding_msg, reply_markup=reply_markup, parse_mode='Markdown')
    
    async def list_wallets_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """List managed wallets"""
        if not self.is_authorized(update.effective_user.id):
            return
        
        wallet_data = await self.get_wallet_info()
        
        wallet_msg = """
🏦 **Managed Wallets**

"""
        
        if not wallet_data['wallets']:
            wallet_msg += "No wallets currently managed.\n\nUse `/fund <wallet> <amount>` to create and fund a wallet."
        else:
            total_balance = 0
            for wallet in wallet_data['wallets']:
                wallet_msg += f"""
💼 **{wallet['name']}**
• Address: `{wallet['address'][:8]}...{wallet['address'][-8:]}`
• Balance: {wallet['balance']:.3f} SOL
• Status: {"🟢 Active" if wallet['active'] else "🔴 Inactive"}
• Last Used: {wallet['last_used']}

"""
                total_balance += wallet['balance']
            
            wallet_msg += f"💰 **Total Balance:** {total_balance:.3f} SOL"
        
        keyboard = [
            [InlineKeyboardButton("🔄 Refresh", callback_data="refresh_wallets")],
            [InlineKeyboardButton("➕ Fund New", callback_data="fund_new_wallet")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(wallet_msg, reply_markup=reply_markup, parse_mode='Markdown')
    
    async def dashboard_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Link to web dashboard"""
        if not self.is_authorized(update.effective_user.id):
            return
        
        dashboard_msg = """
📊 **TrenchBot Dashboard**

🌐 **Web Interface:**
[Main Dashboard](http://localhost:3000/dashboard)
[Whale Tracker](http://localhost:3000/whales)
[Performance Metrics](http://localhost:3000/performance)
[MEV Analytics](http://localhost:3000/mev)

📱 **Grafana Dashboards:**
• Main: http://localhost:3000/d/trenchbot-main
• Whale Activity: http://localhost:3000/d/whale-tracker
• Combat Operations: http://localhost:3000/d/combat-ops

🔐 **Access:** Use your authorized credentials
⚡ **Real-time:** All dashboards update live
        """
        
        keyboard = [
            [InlineKeyboardButton("🌐 Open Dashboard", url="http://localhost:3000/dashboard")],
            [InlineKeyboardButton("📊 Grafana", url="http://localhost:3000")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(dashboard_msg, reply_markup=reply_markup, parse_mode='Markdown', disable_web_page_preview=True)
    
    async def emergency_stop_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Emergency stop all operations"""
        if not self.is_authorized(update.effective_user.id):
            return
        
        emergency_msg = """
🚨 **EMERGENCY STOP PROTOCOL**

⚠️ This will immediately halt:
• All MEV operations
• Active trades
• Whale tracking
• AI processing

Are you sure you want to proceed?
        """
        
        keyboard = [
            [InlineKeyboardButton("🚨 CONFIRM STOP", callback_data="emergency_stop_confirm")],
            [InlineKeyboardButton("❌ Cancel", callback_data="emergency_stop_cancel")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(emergency_msg, reply_markup=reply_markup, parse_mode='Markdown')
    
    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle inline keyboard button presses"""
        query = update.callback_query
        await query.answer()
        
        if not self.is_authorized(query.from_user.id):
            await query.message.reply_text("🚫 Unauthorized")
            return
        
        data = query.data
        
        if data == "mev_status":
            await self.mev_status_command(update, context)
        elif data == "whale_activity":
            await self.whale_status_command(update, context)
        elif data == "performance":
            await self.performance_command(update, context)
        elif data == "refresh_status":
            await self.status_command(update, context)
        elif data.startswith("confirm_fund_"):
            request_id = data.replace("confirm_fund_", "")
            await self.process_funding(query, request_id, True)
        elif data.startswith("cancel_fund_"):
            request_id = data.replace("cancel_fund_", "")
            await self.process_funding(query, request_id, False)
        elif data == "emergency_stop_confirm":
            await self.execute_emergency_stop(query)
        elif data == "emergency_stop_cancel":
            await query.message.reply_text("✅ Emergency stop cancelled.")
    
    async def process_funding(self, query, request_id: str, confirm: bool):
        """Process wallet funding request"""
        if request_id not in self.funding_requests:
            await query.message.reply_text("❌ Funding request not found.")
            return
        
        request = self.funding_requests[request_id]
        
        if confirm:
            # Simulate funding (replace with actual Solana integration)
            success = await self.fund_wallet_actual(request['wallet'], request['amount'])
            
            if success:
                request['status'] = 'completed'
                msg = f"""
✅ **Funding Successful**

💰 **Details:**
• Wallet: `{request['wallet']}`
• Amount: {request['amount']} SOL
• Status: ✅ Completed
• Transaction: [View on Explorer](https://solscan.io/tx/mock_tx_hash)

🚀 Wallet is now funded and ready for trading!
                """
            else:
                request['status'] = 'failed'
                msg = "❌ Funding failed. Please try again or contact support."
        else:
            request['status'] = 'cancelled'
            msg = "❌ Funding request cancelled."
        
        await query.message.reply_text(msg, parse_mode='Markdown')
        del self.funding_requests[request_id]
    
    async def execute_emergency_stop(self, query):
        """Execute emergency stop"""
        # Simulate emergency stop procedures
        stop_msg = """
🚨 **EMERGENCY STOP EXECUTED**

✅ All operations halted:
• MEV scanning: STOPPED
• Active trades: CANCELLED  
• Whale tracking: PAUSED
• AI processing: SUSPENDED

🛡️ System is now in safe mode.
Use `/start` to reactivate when ready.
        """
        
        # Update system state
        self.mev_alerts_enabled = False
        self.whale_alerts_enabled = False
        
        await query.message.reply_text(stop_msg, parse_mode='Markdown')
    
    async def unauthorized_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle unauthorized messages"""
        if not self.is_authorized(update.effective_user.id):
            await update.message.reply_text(
                "🚫 Unauthorized access. This bot is for authorized users only.\n\n"
                f"Your ID: {update.effective_user.id}\n"
                "Contact the administrator for access."
            )
    
    # Data fetching methods (simulate real data)
    async def get_system_status(self) -> Dict:
        """Get current system status"""
        return {
            'quantum_latency': 0.18,
            'monte_carlo_latency': 0.01,
            'speedup': 122,
            'mev_opportunities': 47,
            'successful_trades': 23,
            'total_profit': 1247.83,
            'active_whales': 12,
            'large_txs': 156,
            'whale_alerts': 8,
            'total_balance': 450.25
        }
    
    async def get_mev_status(self) -> Dict:
        """Get MEV status data"""
        return {
            'arbitrage_ops': 12,
            'sandwich_ops': 5,
            'liquidation_ops': 3,
            'total_opportunities': 247,
            'success_rate': 78.5,
            'avg_profit': 15.23,
            'gas_efficiency': 92.1,
            'best_trade': 234.56,
            'best_token': 'SOL/USDC',
            'best_token_profit': 12.7,
            'most_active_pair': 'RAY/SOL',
            'max_drawdown': 2.3,
            'sharpe_ratio': 1.87,
            'win_rate': 73.2
        }
    
    async def get_whale_activity(self) -> Dict:
        """Get whale activity data"""
        return {
            'large_wallets': 15,
            'mega_txs': 8,
            'total_volume': 2450000,
            'sol_whales': 6,
            'sol_volume': 15000,
            'token_whales': 9,
            'new_whales': 2,
            'followed_whales': ['whale_1', 'whale_2', 'whale_3'],
            'copy_trades': 12,
            'copy_success_rate': 85.7,
            'recent_activities': [
                {'wallet': 'GjK8p2x3...', 'action': 'bought', 'amount': 50000, 'token': 'SOL', 'time': '2 min ago'},
                {'wallet': 'Hm9kL5n1...', 'action': 'sold', 'amount': 25000, 'token': 'RAY', 'time': '5 min ago'},
                {'wallet': 'Tp7qR3m8...', 'action': 'swapped', 'amount': 100000, 'token': 'USDC', 'time': '8 min ago'},
            ]
        }
    
    async def get_performance_metrics(self) -> Dict:
        """Get AI performance metrics"""
        return {
            'overall_speedup': 122,
            'total_latency': 55.83,
            'quantum_ms': 0.18,
            'quantum_speedup': 306,
            'transformer_ms': 50.0,
            'transformer_speedup': 85,
            'monte_carlo_ms': 0.01,
            'monte_carlo_speedup': 249612,
            'graph_ms': 3.68,
            'competitive_ms': 1.96,
            'cpu_usage': 45.2,
            'memory_usage': 67.8,
            'gpu_usage': 23.4,
            'decision_latency': 12.5,
            'execution_speed': 8.3,
            'orders_per_sec': 120
        }
    
    async def get_wallet_info(self) -> Dict:
        """Get wallet information"""
        return {
            'wallets': [
                {
                    'name': 'temp_wallet_1',
                    'address': 'GjK8p2x3mL9qR7zS4v1nB6cH8dF5wX2yT9uA3kP7sQ1m',
                    'balance': 15.456,
                    'active': True,
                    'last_used': '2 hours ago'
                },
                {
                    'name': 'temp_wallet_2', 
                    'address': 'Hm9kL5n1wP8tR2xS6v4nC9dG7bF3yX5uT1oA8kM4sQ7p',
                    'balance': 8.234,
                    'active': True,
                    'last_used': '15 min ago'
                }
            ]
        }
    
    async def fund_wallet_actual(self, wallet_address: str, amount: float) -> bool:
        """Actually fund a wallet (simulate for now)"""
        # Simulate funding delay
        await asyncio.sleep(2)
        
        # Simulate 95% success rate
        import random
        return random.random() > 0.05
    
    async def send_mev_alert(self, chat_id: int, opportunity: Dict):
        """Send MEV opportunity alert"""
        alert_msg = f"""
🚨 **MEV OPPORTUNITY DETECTED**

💰 **Opportunity:** {opportunity['type']}
🎯 **Profit Potential:** +${opportunity['profit']:.2f}
⚡ **Confidence:** {opportunity['confidence']:.1f}%
🕐 **Time Sensitive:** {opportunity['urgency']}

📊 **Details:**
• Token: {opportunity['token']}
• Pool: {opportunity['pool']}
• Gas Cost: ~{opportunity['gas_cost']:.3f} SOL

⚠️ **Risk Level:** {opportunity['risk_level']}
        """
        
        keyboard = [
            [InlineKeyboardButton("✅ Execute", callback_data=f"execute_{opportunity['id']}")],
            [InlineKeyboardButton("❌ Skip", callback_data=f"skip_{opportunity['id']}")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await self.app.bot.send_message(
            chat_id=chat_id,
            text=alert_msg,
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
    
    def run(self):
        """Start the Telegram bot"""
        logger.info("Starting TrenchBot Telegram MEV Alert Bot...")
        self.app.run_polling(allowed_updates=Update.ALL_TYPES)

def main():
    """Main function to start the bot"""
    # Configuration
    BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', 'YOUR_BOT_TOKEN_HERE')
    
    # Authorized user IDs (replace with actual Telegram user IDs)
    AUTHORIZED_USERS = [
        123456789,  # Replace with your Telegram user ID
        987654321,  # Add more authorized users
    ]
    
    if BOT_TOKEN == 'YOUR_BOT_TOKEN_HERE':
        print("❌ Please set TELEGRAM_BOT_TOKEN environment variable")
        print("   Get a bot token from @BotFather on Telegram")
        return
    
    # Create and start bot
    bot = TrenchBotTelegramBot(BOT_TOKEN, AUTHORIZED_USERS)
    
    print("🚀 TrenchBot Telegram MEV Alert Bot starting...")
    print(f"📱 Authorized users: {len(AUTHORIZED_USERS)}")
    print("💡 Use /start command in Telegram to begin")
    
    try:
        bot.run()
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user")
    except Exception as e:
        print(f"❌ Bot error: {e}")

if __name__ == '__main__':
    main()