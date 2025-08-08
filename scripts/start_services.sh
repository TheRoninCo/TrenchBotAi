#!/bin/bash
set -e

echo "🚀 Starting TrenchBotAI services..."

# Source environment
source .env.runpod

# Start Jupyter Lab in background
echo "📊 Starting Jupyter Lab..."
nohup jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password='' > logs/jupyter.log 2>&1 &

# Start TensorBoard in background
echo "📈 Starting TensorBoard..."
nohup tensorboard --logdir=logs --host=0.0.0.0 --port=6006 > logs/tensorboard.log 2>&1 &

# Start the main TrenchBotAI application
echo "⚡ Starting TrenchBotAI..."
export RUST_LOG=info
export RUST_BACKTRACE=1

# Run in training mode for RunPod
./target/release/trenchbot-dex --help

echo "✅ All services started successfully!"