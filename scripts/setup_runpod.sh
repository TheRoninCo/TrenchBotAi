#!/bin/bash
set -e

echo "🚀 Setting up TrenchBotAI on RunPod A100 SXM..."

# Verify CUDA installation
echo "📊 CUDA Version:"
nvcc --version

echo "🔧 GPU Information:"
nvidia-smi

# Set up Rust environment
source ~/.cargo/env
echo "🦀 Rust Version:"
rustc --version

# Build the project with GPU features
echo "⚙️  Building TrenchBotAI with GPU features..."
cd /workspace
export CUDA_VISIBLE_DEVICES=0
export LIBTORCH_USE_PYTORCH=1

# Clean previous builds
cargo clean

# Build with all features for RunPod
cargo build --release --features gpu,training,runpod

# Create logs directory
mkdir -p logs

# Set up Python environment
echo "🐍 Python Environment:"
python --version
pip list | grep torch

# Test GPU availability in PyTorch
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Create production environment file
cat > .env.runpod << 'ENVEOF'
# Solana/Helius Configuration
HELIUS_API_KEY=${HELIUS_API_KEY}
HELIUS_RPC_HTTP=https://mainnet.helius-rpc.com/?api-key=${HELIUS_API_KEY}
HELIUS_RPC_WSS=wss://mainnet.helius-rpc.com/?api-key=${HELIUS_API_KEY}
HELIUS_FAST_SENDER=http://ewr-sender.helius-rpc.com/fast

# Solscan Configuration  
SOLSCAN_API_KEY=${SOLSCAN_API_KEY}
SOLSCAN_API_BASE=https://pro-api.solscan.io

# Jupiter Configuration
JUPITER_API_BASE=https://lite-api.jup.ag

# Jito Configuration
JITO_TIP_ACCOUNTS=4ACfpUFoaSD9bfPdeu6DBt89gB6ENTeHBXCAi87NhDEE,D2L6yPZ2FmmmTKPgzaMKdhu6EWZcTpLy1Vhx8uvZe7NZ,9bnz4RShgq1hAnLnZbP8kbgBg1kEmcJBYQq3gQbmnSta,5VY91ws6B2hMmBFRsXkoAAdsPHBJwRfBht4DXox3xkwn,2nyhqdwKcJZR2vcqCyrYsaPVdAnFoJjiksCXJ7hfEYgD,2q5pghRs6arqVjRvT5gfgWfWcHWmw1ZuCzphgd5KfWGJ,wyvPkWjVZz1M8fHQnMMCDTQDbkManefNNhweYk5WkcF,3KCKozbAaF75qEU33jtzozcJ29yJuaLJTy2jFdzUY8bT,4vieeGHPYPG2MmyPRcYjdiDmmhN3ww7hsFNap8pVN3Ey,4TQLFNWK8AovT1gFvda5jfw2oJeRMKEmw7aH6MGBJ3or

# RunPod Configuration
RUNPOD_ENVIRONMENT=true
CUDA_VISIBLE_DEVICES=0
RUST_LOG=info
RUST_BACKTRACE=1

# Training Configuration
BATCH_SIZE=64
LEARNING_RATE=0.0001
PRECISION=fp16
MAX_SEQUENCE_LENGTH=512

# API Configuration
API_HOST=0.0.0.0
API_PORT=8080
JUPYTER_PORT=8888
TENSORBOARD_PORT=6006

# Monitoring
WANDB_API_KEY=${WANDB_API_KEY:-}
ENABLE_MONITORING=true
LOG_LEVEL=INFO
ENVEOF

echo "✅ TrenchBotAI setup complete!"
echo "🎯 Ready for A100 SXM training and deployment"