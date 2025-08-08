#!/bin/bash
# Complete TrenchBotAi RunPod Deployment Script
# Deploys fully configured AI training environment

set -e  # Exit on error

echo "🚀 TrenchBotAi Complete RunPod Deployment"
echo "========================================"

# Configuration
CONTAINER_NAME="trenchbot-ai-trainer"
REGISTRY="your-registry"  # Update with your registry
TAG="cuda121-complete"
RUNPOD_TEMPLATE="runpod-training-template.yml"

# Build stages
echo "📦 Building complete training container..."

# Stage 1: Build optimized container
echo "Stage 1: Building Docker container..."
docker build -f Dockerfile.complete -t ${CONTAINER_NAME}:${TAG} . \
    --build-arg CUDA_VERSION=12.1 \
    --build-arg PYTHON_VERSION=3.11 \
    --build-arg RUST_VERSION=1.75

# Stage 2: Test container locally (optional)
echo "Stage 2: Testing container..."
if [ "$1" == "--test" ]; then
    echo "🧪 Running local test..."
    docker run --gpus all --rm -p 8888:8888 -p 6006:6006 -p 9090:9090 \
        -v $(pwd)/test_data:/workspace/data \
        -it ${CONTAINER_NAME}:${TAG} /bin/bash -c "
        echo 'Testing GPU access...'
        nvidia-smi
        echo 'Testing Rust compilation...'
        cargo check --features gpu,training
        echo 'Testing Python ML stack...'
        python3 -c 'import torch; print(f\"PyTorch: {torch.__version__}\"); print(f\"CUDA available: {torch.cuda.is_available()}\")'
        "
fi

# Stage 3: Tag and push to registry
echo "Stage 3: Pushing to registry..."
docker tag ${CONTAINER_NAME}:${TAG} ${REGISTRY}/${CONTAINER_NAME}:${TAG}
docker tag ${CONTAINER_NAME}:${TAG} ${REGISTRY}/${CONTAINER_NAME}:latest

if [ "$2" == "--push" ]; then
    docker push ${REGISTRY}/${CONTAINER_NAME}:${TAG}
    docker push ${REGISTRY}/${CONTAINER_NAME}:latest
    echo "✅ Container pushed to registry"
fi

# Stage 4: Generate RunPod deployment command
echo "Stage 4: Generating RunPod commands..."

cat > runpod_deploy_commands.txt << EOF
# RunPod CLI Deployment Commands
# Make sure you have runpodctl installed: https://github.com/runpod/runpodctl

# Option 1: Deploy with template file
runpodctl create template --file=${RUNPOD_TEMPLATE}

# Option 2: Deploy with CLI arguments
runpodctl create instance \\
  --image-name=${REGISTRY}/${CONTAINER_NAME}:${TAG} \\
  --gpu-type=A100 \\
  --container-disk=100GB \\
  --volume-in-path=/workspace/data \\
  --volume-out-path=/workspace/models \\
  --port=8888:8888 \\
  --port=6006:6006 \\
  --port=9090:9090 \\
  --env="CUDA_VISIBLE_DEVICES=0" \\
  --env="TRAINING_MODE=quantum_mev" \\
  --env="BATCH_SIZE=64"

# Option 3: Web UI deployment
# 1. Go to https://runpod.io/console/pods
# 2. Click "New Pod"
# 3. Use image: ${REGISTRY}/${CONTAINER_NAME}:${TAG}
# 4. GPU: A100 (40GB or 80GB)
# 5. Container Disk: 100GB
# 6. Add ports: 8888, 6006, 9090
# 7. Add environment variables from template

# Access URLs after deployment:
# - Jupyter Lab: https://[pod-id]-8888.proxy.runpod.net
# - TensorBoard: https://[pod-id]-6006.proxy.runpod.net  
# - Health/Metrics: https://[pod-id]-9090.proxy.runpod.net/health
EOF

# Stage 5: Create monitoring script
cat > monitor_training.py << 'EOF'
#!/usr/bin/env python3
"""
TrenchBotAi Training Monitor
Monitors training progress and GPU utilization on RunPod
"""

import requests
import time
import json
from datetime import datetime

class TrainingMonitor:
    def __init__(self, pod_url, health_port=9090):
        self.base_url = f"https://{pod_url}-{health_port}.proxy.runpod.net"
        
    def check_health(self):
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def monitor_loop(self, interval=30):
        print("🔍 Starting TrenchBotAi Training Monitor...")
        print("=" * 50)
        
        while True:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            health = self.check_health()
            
            if "error" in health:
                print(f"[{timestamp}] ❌ Error: {health['error']}")
            else:
                gpu_util = health.get('gpu_utilization', 0)
                gpu_mem = health.get('gpu_memory_used', 0) / health.get('gpu_memory_total', 1) * 100
                cpu_util = health.get('cpu_percent', 0)
                
                print(f"[{timestamp}] 📊 GPU: {gpu_util:.1f}% | VRAM: {gpu_mem:.1f}% | CPU: {cpu_util:.1f}%")
            
            time.sleep(interval)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python3 monitor_training.py <runpod-pod-id>")
        sys.exit(1)
    
    monitor = TrainingMonitor(sys.argv[1])
    monitor.monitor_loop()
EOF

chmod +x monitor_training.py

echo ""
echo "✅ Deployment Complete!"
echo "======================"
echo ""
echo "📋 Next Steps:"
echo "1. Review runpod_deploy_commands.txt for deployment options"
echo "2. Update REGISTRY variable in this script with your container registry"
echo "3. Push container: ./deploy_complete.sh --push"
echo "4. Deploy on RunPod using provided commands"
echo "5. Monitor training: python3 monitor_training.py <pod-id>"
echo ""
echo "🔗 Access URLs after deployment:"
echo "- Jupyter Lab: https://[pod-id]-8888.proxy.runpod.net"
echo "- TensorBoard: https://[pod-id]-6006.proxy.runpod.net"
echo "- Metrics: https://[pod-id]-9090.proxy.runpod.net/health"
echo ""
echo "🎯 Training will start automatically when pod boots!"

# Optional: Show container size info
echo ""
echo "📊 Container Information:"
docker images ${CONTAINER_NAME}:${TAG} --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"