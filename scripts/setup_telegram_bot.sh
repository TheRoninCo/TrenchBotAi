#!/bin/bash

# TrenchBot Telegram Bot Setup Script
# Sets up and configures the Telegram MEV alert bot

set -e

echo "🤖 TrenchBot Telegram Bot Setup"
echo "================================"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "Cargo.toml" ]; then
    echo -e "${RED}❌ Please run this script from the TrenchBot project root directory${NC}"
    exit 1
fi

echo -e "${BLUE}📋 Step 1: Installing Python Dependencies${NC}"
echo "Installing required Python packages..."

# Install Telegram bot dependencies
pip3 install python-telegram-bot[all] aiohttp python-dotenv toml

echo -e "${GREEN}✅ Dependencies installed${NC}"

echo -e "${BLUE}📱 Step 2: Telegram Bot Setup${NC}"
echo "To create a Telegram bot, follow these steps:"
echo ""
echo "1. Open Telegram and search for @BotFather"
echo "2. Send /newbot command"
echo "3. Choose a name: TrenchBot MEV Alert"
echo "4. Choose a username: trenchbot_mev_bot (or similar)"
echo "5. Copy the bot token from BotFather"
echo ""

# Prompt for bot token
read -p "Enter your Telegram bot token: " BOT_TOKEN

if [ -z "$BOT_TOKEN" ]; then
    echo -e "${RED}❌ Bot token cannot be empty${NC}"
    exit 1
fi

echo -e "${BLUE}👤 Step 3: User Authorization Setup${NC}"
echo "To get your Telegram user ID:"
echo "1. Search for @userinfobot on Telegram"
echo "2. Send /start to get your user ID"
echo ""

read -p "Enter your Telegram user ID: " USER_ID

if [ -z "$USER_ID" ]; then
    echo -e "${RED}❌ User ID cannot be empty${NC}"
    exit 1
fi

echo -e "${BLUE}🔧 Step 4: Creating Environment File${NC}"

# Create .env file for bot configuration
cat > .env.telegram << EOF
# TrenchBot Telegram Bot Environment Variables
TELEGRAM_BOT_TOKEN=$BOT_TOKEN
TELEGRAM_USER_ID=$USER_ID

# Optional: Grafana API key for dashboard integration
GRAFANA_API_KEY=your_grafana_api_key_here

# Optional: Custom dashboard URLs
MAIN_DASHBOARD_URL=http://localhost:3000/dashboard
WHALE_DASHBOARD_URL=http://localhost:3000/whales
PERFORMANCE_DASHBOARD_URL=http://localhost:3000/performance
EOF

echo -e "${GREEN}✅ Environment file created: .env.telegram${NC}"

echo -e "${BLUE}⚙️  Step 5: Updating Configuration${NC}"

# Update the telegram.toml config with the user ID
if [ -f "configs/telegram.toml" ]; then
    # Use sed to update the authorized_users array
    sed -i.bak "s/# 123456789,/$USER_ID,/" configs/telegram.toml
    echo -e "${GREEN}✅ Updated configs/telegram.toml with your user ID${NC}"
else
    echo -e "${YELLOW}⚠️  configs/telegram.toml not found, using defaults${NC}"
fi

echo -e "${BLUE}🚀 Step 6: Creating Launch Script${NC}"

# Create a launch script
cat > launch_telegram_bot.sh << 'EOF'
#!/bin/bash

# TrenchBot Telegram Bot Launcher
echo "🚀 Starting TrenchBot Telegram MEV Alert Bot..."

# Load environment variables
if [ -f .env.telegram ]; then
    export $(cat .env.telegram | grep -v '^#' | xargs)
else
    echo "❌ .env.telegram file not found!"
    echo "   Run setup_telegram_bot.sh first"
    exit 1
fi

# Check if bot token is set
if [ -z "$TELEGRAM_BOT_TOKEN" ]; then
    echo "❌ TELEGRAM_BOT_TOKEN not set!"
    echo "   Please check your .env.telegram file"
    exit 1
fi

# Start the bot
cd "$(dirname "$0")"
python3 scripts/telegram_mev_bot.py
EOF

chmod +x launch_telegram_bot.sh

echo -e "${GREEN}✅ Launch script created: launch_telegram_bot.sh${NC}"

echo -e "${BLUE}🛠️  Step 7: Creating Systemd Service (Optional)${NC}"

# Create systemd service file
cat > trenchbot-telegram.service << EOF
[Unit]
Description=TrenchBot Telegram MEV Alert Bot
After=network.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=$(pwd)
Environment=PATH=$(pwd)/venv/bin:/usr/local/bin:/usr/bin:/bin
EnvironmentFile=$(pwd)/.env.telegram
ExecStart=/usr/bin/python3 $(pwd)/scripts/telegram_mev_bot.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

echo -e "${GREEN}✅ Systemd service file created: trenchbot-telegram.service${NC}"
echo -e "${YELLOW}   To install: sudo cp trenchbot-telegram.service /etc/systemd/system/${NC}"
echo -e "${YELLOW}   To enable: sudo systemctl enable trenchbot-telegram${NC}"
echo -e "${YELLOW}   To start: sudo systemctl start trenchbot-telegram${NC}"

echo -e "${BLUE}📋 Step 8: Testing Bot Setup${NC}"

echo "Testing bot configuration..."

# Test if we can import the required modules
python3 -c "
import sys
try:
    from telegram import Bot
    from telegram.ext import Application
    print('✅ Telegram bot libraries imported successfully')
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)

# Test bot token format
token = '$BOT_TOKEN'
if ':' not in token or len(token) < 40:
    print('❌ Bot token format appears invalid')
    sys.exit(1)
else:
    print('✅ Bot token format looks valid')

print('✅ Basic configuration test passed')
"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Configuration test passed${NC}"
else
    echo -e "${RED}❌ Configuration test failed${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}🎉 Telegram Bot Setup Complete!${NC}"
echo ""
echo -e "${BLUE}📱 Next Steps:${NC}"
echo "1. Test your bot: ./launch_telegram_bot.sh"
echo "2. Open Telegram and search for your bot"
echo "3. Send /start to begin using the bot"
echo ""
echo -e "${BLUE}🔧 Available Commands:${NC}"
echo "• /status - System status overview"
echo "• /mev - MEV opportunities & stats"
echo "• /whales - Whale activity monitoring"
echo "• /performance - AI performance metrics"
echo "• /fund <wallet> <amount> - Fund temp wallets"
echo "• /dashboard - Access web dashboards"
echo "• /emergency - Emergency stop all operations"
echo ""
echo -e "${BLUE}📊 Integration Features:${NC}"
echo "• Real-time MEV opportunity alerts"
echo "• Whale movement notifications"
echo "• Performance degradation warnings"
echo "• Instant wallet funding via commands"
echo "• Direct dashboard access links"
echo "• Emergency stop capabilities"
echo ""
echo -e "${YELLOW}⚠️  Security Notes:${NC}"
echo "• Only authorized users can access the bot"
echo "• All commands are logged for security"
echo "• Emergency stop available for safety"
echo "• Rate limiting prevents abuse"
echo ""
echo -e "${BLUE}🚀 Start the bot now: ./launch_telegram_bot.sh${NC}"