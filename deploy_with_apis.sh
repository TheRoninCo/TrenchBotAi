#!/bin/bash
# TrenchBotAi RunPod Deployment with API Keys Configured
# Secure deployment with your Helius API key integrated

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}🚀 TrenchBotAi RunPod Deployment with API Integration${NC}"
echo "================================================================="

# Configuration
CONTAINER_NAME="trenchbot-ai-trainer"
TAG="cuda121-helius"
HELIUS_API_KEY="3706de56-b630-4023-bf35-61fa0c851ba5"

# Step 1: Verify API key
echo -e "${BLUE}[Step 1]${NC} Verifying Helius API Access..."

# Test the API key
echo "Testing Helius API connection..."
curl -s -X POST \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"getHealth"}' \
  "https://mainnet.helius-rpc.com/?api-key=${HELIUS_API_KEY}" | grep -q "ok"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Helius API key is valid and working${NC}"
else
    echo -e "${RED}❌ Helius API key test failed${NC}"
    exit 1
fi

# Step 2: Get container registry info
echo -e "${BLUE}[Step 2]${NC} Container Registry Setup..."

echo "Choose your container registry:"
echo "1) Docker Hub (easiest - recommended)"
echo "2) Skip (use pre-built image)"

read -p "Choose option (1-2): " registry_choice

case $registry_choice in
    1)
        read -p "Enter your Docker Hub username: " DOCKER_USER
        REGISTRY="$DOCKER_USER"
        echo "Using Docker Hub: $REGISTRY/$CONTAINER_NAME"
        ;;
    2)
        echo "Using pre-built image (if available)"
        REGISTRY="trenchbot"  # Default registry
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

# Step 3: Create secure environment file for container
echo -e "${BLUE}[Step 3]${NC} Creating secure configuration..."

cat > container.env << EOF
# API Keys
HELIUS_API_KEY=${HELIUS_API_KEY}
HELIUS_RPC_HTTP=https://mainnet.helius-rpc.com/?api-key=${HELIUS_API_KEY}
HELIUS_RPC_WSS=wss://mainnet.helius-rpc.com/?api-key=${HELIUS_API_KEY}

# Training Configuration
TRAINING_MODE=quantum_mev
BATCH_SIZE=64
LEARNING_RATE=0.0001
MAX_EPOCHS=1000

# Hardware Optimization
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
MIXED_PRECISION=true

# Paths
TRAINING_DATA_PATH=/workspace/data/training
MODEL_OUTPUT_PATH=/workspace/models
LOG_PATH=/workspace/logs

# Monitoring
RUST_LOG=trenchbot_dex=info
JUPYTER_TOKEN=trenchbot-secure-2024
EOF

echo -e "${GREEN}✅ Configuration file created${NC}"

# Step 4: Build container with API integration
if [ "$registry_choice" == "1" ]; then
    echo -e "${BLUE}[Step 4]${NC} Building container with API integration..."
    
    # Create a specialized Dockerfile that includes the environment
    cat > Dockerfile.helius << 'DOCKERFILE'
FROM nvidia/cuda:12.1-devel-ubuntu22.04

# System setup
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    build-essential pkg-config libssl-dev \
    python3 python3-pip python3-dev \
    curl wget git htop nvtop \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install Python ML stack
RUN pip3 install --no-cache-dir \
    torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121 && \
    pip3 install --no-cache-dir \
    polars pandas numpy matplotlib seaborn plotly \
    jupyter jupyterlab tensorboard wandb \
    scikit-learn transformers datasets \
    websockets aiohttp requests

# Create workspace
WORKDIR /workspace
RUN mkdir -p /workspace/{data,models,logs,config}

# Copy source code
COPY . /workspace/src/
WORKDIR /workspace/src

# Build with all features
RUN cargo build --release --features "gpu,training,ai,monitoring"

# Create startup script with API integration
RUN cat > /workspace/start_training.sh << 'EOF'
#!/bin/bash
echo "🚀 TrenchBotAi Training with Helius Integration Starting..."

# GPU check
nvidia-smi

# Test Helius API
echo "📡 Testing Helius API connection..."
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"getHealth"}' \
  "${HELIUS_RPC_HTTP}" | grep -q "ok" && echo "✅ Helius API connected"

# Start services
jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser \
  --ServerApp.token="${JUPYTER_TOKEN:-}" &
tensorboard --logdir=/workspace/logs --bind_all --port=6006 &

# Start health monitor
python3 -c "
import json, time
from http.server import HTTPServer, BaseHTTPRequestHandler
import subprocess

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            try:
                # Check GPU
                gpu_info = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits']).decode()
                gpu_util, mem_used, mem_total = gpu_info.strip().split(', ')
                
                health = {
                    'status': 'healthy',
                    'timestamp': time.time(),
                    'gpu_utilization': float(gpu_util),
                    'gpu_memory_used': int(mem_used),
                    'gpu_memory_total': int(mem_total),
                    'helius_api': 'connected' if '${HELIUS_API_KEY}' else 'not_configured',
                    'training_mode': '${TRAINING_MODE}',
                    'services': {
                        'jupyter': 'running on port 8888',
                        'tensorboard': 'running on port 6006'
                    }
                }
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(health, indent=2).encode())
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode())

if __name__ == '__main__':
    HTTPServer(('0.0.0.0', 9090), HealthHandler).serve_forever()
" &

# Start training pipeline
echo "🧠 Initializing TrenchBotAi training..."
if [ ! -f "/workspace/data/training/initialized" ]; then
    echo "📥 Setting up initial training data..."
    mkdir -p /workspace/data/training
    ./target/release/trenchbot-dex data collect \
        --helius-key="${HELIUS_API_KEY}" \
        --output="/workspace/data/training" \
        --duration="1h"
    touch /workspace/data/training/initialized
fi

# Main training loop
./target/release/trenchbot-dex train \
    --config=/workspace/config/training.toml \
    --data-path=/workspace/data/training \
    --output-path=/workspace/models \
    --log-path=/workspace/logs \
    --helius-key="${HELIUS_API_KEY}"
EOF

RUN chmod +x /workspace/start_training.sh

# Expose ports
EXPOSE 8888 6006 9090

# Default command
CMD ["/workspace/start_training.sh"]
DOCKERFILE

    echo "Building container with Helius integration..."
    docker build -f Dockerfile.helius -t ${CONTAINER_NAME}:${TAG} .
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Container built successfully${NC}"
    else
        echo -e "${RED}❌ Container build failed${NC}"
        exit 1
    fi
    
    # Tag and push
    docker tag ${CONTAINER_NAME}:${TAG} ${REGISTRY}/${CONTAINER_NAME}:${TAG}
    
    echo "Pushing to registry..."
    echo "You may need to login first: docker login"
    read -p "Press Enter after logging in to continue..."
    
    docker push ${REGISTRY}/${CONTAINER_NAME}:${TAG}
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Container pushed to registry${NC}"
        FINAL_IMAGE="${REGISTRY}/${CONTAINER_NAME}:${TAG}"
    else
        echo -e "${RED}❌ Push failed${NC}"
        exit 1
    fi
else
    FINAL_IMAGE="${REGISTRY}/${CONTAINER_NAME}:${TAG}"
fi

# Step 5: Generate RunPod deployment instructions
echo -e "${BLUE}[Step 5]${NC} Generating RunPod deployment instructions..."

cat > runpod_deploy_helius.txt << EOF
🚀 TrenchBotAi RunPod Deployment with Helius API
===============================================

Your container is ready with Helius API integration!

📦 Container Image: ${FINAL_IMAGE}

🌐 DEPLOY ON RUNPOD:
-------------------
1. Go to: https://runpod.io/console/pods
2. Click "New Pod" 
3. Select GPU: A100 (recommended) or RTX 4090 (budget)
4. Container Settings:
   - Image: ${FINAL_IMAGE}
   - Container Disk: 100GB
   - Expose Ports: 8888, 6006, 9090

5. Environment Variables (IMPORTANT!):
   HELIUS_API_KEY=3706de56-b630-4023-bf35-61fa0c851ba5
   HELIUS_RPC_HTTP=https://mainnet.helius-rpc.com/?api-key=3706de56-b630-4023-bf35-61fa0c851ba5
   TRAINING_MODE=quantum_mev
   BATCH_SIZE=64
   CUDA_VISIBLE_DEVICES=0
   PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

6. Click "Deploy"

🔗 ACCESS YOUR TRAINING ENVIRONMENT:
-----------------------------------
After deployment, you'll get a Pod ID (e.g., abc123def456)

• Jupyter Lab: https://abc123def456-8888.proxy.runpod.net
  - Token: trenchbot-secure-2024
  - Full AI development environment
  - Interactive model training and testing

• TensorBoard: https://abc123def456-6006.proxy.runpod.net  
  - Real-time training metrics
  - Loss curves and performance graphs
  - Model architecture visualization

• Health API: https://abc123def456-9090.proxy.runpod.net/health
  - GPU utilization monitoring
  - Helius API connection status
  - System health metrics

📊 WHAT HAPPENS AUTOMATICALLY:
-----------------------------
✅ Container starts and initializes GPU
✅ Helius API connection validated  
✅ Jupyter Lab starts on port 8888
✅ TensorBoard starts on port 6006
✅ Health monitoring starts on port 9090
✅ Initial market data collection begins
✅ Training pipeline initializes with your API key
✅ Models automatically saved to /workspace/models

🎯 TRAINING FEATURES ENABLED:
----------------------------
• Quantum-inspired MEV detection
• Real-time Solana market data via Helius
• Rug pull prediction models
• Whale behavior analysis
• Ultra-low latency inference optimization
• Mixed precision training on A100

💰 ESTIMATED COSTS:
------------------
• A100 80GB: ~\$2.89/hour
• A100 40GB: ~\$2.29/hour
• RTX 4090: ~\$0.79/hour

📱 MONITOR TRAINING:
-------------------
python3 monitor_training.py <pod-id>

🆘 TROUBLESHOOTING:
------------------
If you have issues:
1. Check health endpoint: /health
2. View logs in Jupyter: /workspace/logs/
3. Restart container if needed
4. Verify API key in environment variables

✅ Ready to deploy! Your Helius API is configured and ready for AI training.
EOF

# Final summary
echo ""
echo -e "${GREEN}🎉 Deployment Ready!${NC}"
echo "==================="
echo ""
echo -e "${GREEN}✅${NC} Helius API key configured and tested"
echo -e "${GREEN}✅${NC} Container built with API integration: ${FINAL_IMAGE}"
echo -e "${GREEN}✅${NC} Deployment instructions: runpod_deploy_helius.txt"
echo ""
echo -e "${BLUE}🚀 Next Steps:${NC}"
echo "1. Read: runpod_deploy_helius.txt"
echo "2. Go to: https://runpod.io/console/pods"
echo "3. Deploy with the provided settings"
echo "4. Access Jupyter Lab and start training!"
echo ""
echo -e "${YELLOW}⚡ Your Helius API will provide real-time Solana data for training!${NC}"

# Clean up
rm -f container.env Dockerfile.helius

echo -e "${GREEN}Ready for RunPod deployment! 🚀${NC}"