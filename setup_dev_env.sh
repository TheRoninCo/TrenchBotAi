#!/bin/bash
# MacBook development environment setup

echo "🖥️ Setting up MacBook development environment..."

# Install dependencies
cargo install cargo-watch cargo-expand

# Create .env for development
if [ ! -f .env ]; then
    cat > .env << 'DEV_ENV'
# Development environment
RUST_LOG=info
DEV_MODE=true
HELIUS_API_KEY=your_key_here
SOLANA_RPC_URL=https://api.devnet.solana.com
SOLANA_WS_URL=wss://api.devnet.solana.com

# MacBook optimizations
RUST_BACKTRACE=1
TOKIO_WORKER_THREADS=4
DEV_ENV
fi

echo "✅ Development environment ready!"
echo "Run: cargo run --features local-dev"
