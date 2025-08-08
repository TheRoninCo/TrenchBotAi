#!/bin/bash

# TrenchBot AI - Foolproof RunPod Deployment Script
# This script sets up and runs TrenchBot AI with minimal dependencies

set -e  # Exit on any error

echo "🔥 TrenchBot AI - RunPod Deployment Starting"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Step 1: Environment Check
print_info "Checking system environment..."

if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    print_status "GPU Detected: $GPU_NAME ($GPU_MEMORY MB)"
else
    print_warning "No NVIDIA GPU detected - running in CPU mode"
fi

MEMORY_GB=$(free -g | awk '/^Mem:/ {print $2}')
CPU_CORES=$(nproc)
print_status "System: ${CPU_CORES} CPU cores, ${MEMORY_GB}GB RAM"

# Step 2: Install Dependencies
print_info "Installing system dependencies..."

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    ca-certificates \
    git \
    htop \
    vim

print_status "System dependencies installed"

# Step 3: Install Rust if not present
if ! command -v cargo &> /dev/null; then
    print_info "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
    source ~/.cargo/env
    print_status "Rust installed successfully"
else
    print_status "Rust already installed: $(rustc --version)"
fi

# Step 4: Clone or update repository
if [ ! -d "TrenchBotAi" ]; then
    print_info "Cloning TrenchBot AI repository..."
    git clone https://github.com/yourusername/TrenchBotAi.git 2>/dev/null || {
        print_error "Repository not found. Using local files."
        # Assume we're already in the project directory
    }
else
    print_info "Repository already exists, updating..."
    cd TrenchBotAi
    git pull origin main 2>/dev/null || print_warning "Could not update repository"
fi

# Navigate to project directory
if [ -d "TrenchBotAi" ]; then
    cd TrenchBotAi
fi

# Step 5: Setup minimal configuration
print_info "Setting up minimal configuration..."

# Use the minimal main.rs
if [ -f "src/main_minimal.rs" ]; then
    cp src/main_minimal.rs src/main.rs
    print_status "Using minimal main.rs for stability"
fi

# Step 6: Build the application
print_info "Building TrenchBot AI (this may take several minutes)..."
export RUST_LOG=info
export RUST_BACKTRACE=1

# Build with minimal features to avoid problematic modules
if cargo build --release --no-default-features --features core 2>&1 | tee build.log; then
    print_status "Build completed successfully!"
else
    print_error "Build failed. Trying alternative approach..."
    
    # Try building just the flash loan module
    print_info "Building with basic features only..."
    if cargo build --release --bin trenchbot-dex 2>&1 | tee build_fallback.log; then
        print_status "Fallback build completed!"
    else
        print_error "Build failed completely. Check build logs."
        tail -20 build_fallback.log
        exit 1
    fi
fi

# Step 7: Setup environment variables
print_info "Setting up environment..."

export PORT=8080
export RUST_LOG="info,trenchbot_dex=debug"
export DEPLOYMENT_MODE="runpod-minimal"

# Create a simple environment file
cat > .env << EOF
PORT=8080
RUST_LOG=info,trenchbot_dex=debug  
DEPLOYMENT_MODE=runpod-minimal
HELIUS_API_KEY=your_helius_key_here
SOLSCAN_API_KEY=your_solscan_key_here
JUPITER_API_KEY=your_jupiter_key_here
EOF

print_status "Environment configured"

# Step 8: Start the application
print_info "Starting TrenchBot AI server..."

echo ""
echo "🚀 TrenchBot AI - Flash Loan MEV Trading System"
echo "=============================================="
echo ""
echo "📊 System Information:"
echo "   • Version: 1.0.0-minimal"
echo "   • Mode: RunPod Deployment"
echo "   • Port: 8080"
echo "   • Features: Core flash loans + MEV detection"
echo ""
echo "📡 API Endpoints:"
echo "   • Health Check: http://localhost:8080/"
echo "   • System Status: http://localhost:8080/status"
echo "   • Test Flash Loans: http://localhost:8080/flash-loan/test"
echo "   • Test MEV Detection: http://localhost:8080/mev/detect"
echo ""
echo "🔧 Configuration:"
echo "   • Edit .env file to add your API keys"
echo "   • Check logs in trenchbot.log"
echo "   • Use Ctrl+C to stop the server"
echo ""

# Run the application with logging
print_status "Server starting on port 8080..."

if [ -f "target/release/trenchbot-dex" ]; then
    exec ./target/release/trenchbot-dex 2>&1 | tee trenchbot.log
else
    print_error "Binary not found. Build may have failed."
    print_info "Available files in target/release/:"
    ls -la target/release/ 2>/dev/null || print_error "No release target found"
    exit 1
fi