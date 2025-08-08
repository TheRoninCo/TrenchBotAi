#!/bin/bash
# TrenchBotAi RunPod Startup Guide
# Interactive script to help deploy to RunPod

set -e

echo "🚀 TrenchBotAi RunPod Startup Assistant"
echo "======================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
CONTAINER_NAME="trenchbot-ai-trainer"
TAG="cuda121-complete"
REGISTRY="your-registry"  # Will be updated during setup

print_step() {
    echo -e "${BLUE}[STEP $1]${NC} $2"
}

print_success() {
    echo -e "${GREEN}✅${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠️${NC} $1"
}

print_error() {
    echo -e "${RED}❌${NC} $1"
}

# Step 1: Check prerequisites
print_step "1" "Checking Prerequisites..."

# Check Docker
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    echo "Install from: https://docs.docker.com/get-docker/"
    exit 1
fi
print_success "Docker is installed"

# Check if we have NVIDIA Docker (optional but recommended for local testing)
if command -v nvidia-docker &> /dev/null; then
    print_success "NVIDIA Docker is available for local GPU testing"
else
    print_warning "NVIDIA Docker not found - local GPU testing not available"
fi

# Step 2: Configure container registry
print_step "2" "Setting up Container Registry..."

echo "Where do you want to push your container image?"
echo "1) Docker Hub (easiest)"
echo "2) Amazon ECR"  
echo "3) Google Container Registry"
echo "4) Other registry"
echo "5) Skip push (local testing only)"

read -p "Choose option (1-5): " registry_choice

case $registry_choice in
    1)
        read -p "Enter your Docker Hub username: " docker_username
        REGISTRY="$docker_username"
        echo "Using Docker Hub: $REGISTRY/$CONTAINER_NAME"
        ;;
    2)
        read -p "Enter your AWS ECR registry URL: " ecr_url
        REGISTRY="$ecr_url"
        echo "Using AWS ECR: $REGISTRY/$CONTAINER_NAME"
        ;;
    3)
        read -p "Enter your GCR project ID: " gcr_project
        REGISTRY="gcr.io/$gcr_project"
        echo "Using Google Container Registry: $REGISTRY/$CONTAINER_NAME"
        ;;
    4)
        read -p "Enter your registry URL: " custom_registry
        REGISTRY="$custom_registry"
        echo "Using custom registry: $REGISTRY/$CONTAINER_NAME"
        ;;
    5)
        print_warning "Skipping registry push - local testing only"
        REGISTRY="local"
        ;;
    *)
        print_error "Invalid choice"
        exit 1
        ;;
esac

# Step 3: Build container
print_step "3" "Building TrenchBotAi Training Container..."

echo "Building container image..."
echo "This may take 10-15 minutes on first build..."

# Build the complete container
docker build -f Dockerfile.complete -t ${CONTAINER_NAME}:${TAG} . \
    --progress=plain \
    2>&1 | tee build.log

if [ $? -eq 0 ]; then
    print_success "Container built successfully!"
else
    print_error "Container build failed. Check build.log for details."
    exit 1
fi

# Show container size
echo "📊 Container Information:"
docker images ${CONTAINER_NAME}:${TAG} --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"

# Step 4: Local testing (optional)
print_step "4" "Local Testing (Optional)"

echo "Would you like to test the container locally first?"
read -p "Test locally? (y/n): " test_locally

if [[ $test_locally =~ ^[Yy]$ ]]; then
    echo "Starting local test container..."
    echo "This will run the container and show you the logs."
    echo "Press Ctrl+C to stop the test."
    
    # Test with or without GPU
    if command -v nvidia-docker &> /dev/null; then
        echo "Testing with GPU support..."
        docker run --gpus all --rm -p 8888:8888 -p 6006:6006 -p 9090:9090 \
            -v $(pwd)/test_data:/workspace/data \
            -it ${CONTAINER_NAME}:${TAG} /bin/bash -c "
            echo '🧪 Testing container...'
            nvidia-smi
            echo 'GPU access: OK'
            python3 -c 'import torch; print(f\"PyTorch: {torch.__version__}\"); print(f\"CUDA available: {torch.cuda.is_available()}\")'
            echo 'PyTorch: OK'
            cargo --version
            echo 'Rust: OK'
            echo '✅ All tests passed!'
            echo 'Container is ready for RunPod deployment!'
            "
    else
        echo "Testing without GPU..."
        docker run --rm -p 8888:8888 -p 6006:6006 -p 9090:9090 \
            -it ${CONTAINER_NAME}:${TAG} /bin/bash -c "
            echo '🧪 Testing container...'
            python3 -c 'import torch; print(f\"PyTorch: {torch.__version__}\")'
            echo 'PyTorch: OK'
            cargo --version  
            echo 'Rust: OK'
            echo '✅ Tests passed (GPU will be available on RunPod)!'
            "
    fi
fi

# Step 5: Push to registry
if [ "$REGISTRY" != "local" ]; then
    print_step "5" "Pushing to Container Registry..."
    
    # Tag for registry
    docker tag ${CONTAINER_NAME}:${TAG} ${REGISTRY}/${CONTAINER_NAME}:${TAG}
    docker tag ${CONTAINER_NAME}:${TAG} ${REGISTRY}/${CONTAINER_NAME}:latest
    
    echo "Pushing to registry..."
    echo "You may need to login to your registry first:"
    
    case $registry_choice in
        1)
            echo "Run: docker login"
            ;;
        2) 
            echo "Run: aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $REGISTRY"
            ;;
        3)
            echo "Run: gcloud auth configure-docker"
            ;;
    esac
    
    read -p "Press Enter after logging in to continue..."
    
    docker push ${REGISTRY}/${CONTAINER_NAME}:${TAG}
    docker push ${REGISTRY}/${CONTAINER_NAME}:latest
    
    print_success "Container pushed to registry!"
fi

# Step 6: Generate RunPod deployment instructions
print_step "6" "Generating RunPod Deployment Instructions..."

# Create customized deployment instructions
cat > runpod_deploy_instructions.txt << EOF
🚀 TrenchBotAi RunPod Deployment Instructions
============================================

Your container is ready! Here's how to deploy on RunPod:

📦 Container Image: ${REGISTRY}/${CONTAINER_NAME}:${TAG}

🌐 OPTION 1: Web UI Deployment (Recommended for beginners)
----------------------------------------------------------
1. Go to: https://runpod.io/console/pods
2. Click "New Pod"
3. Select GPU:
   - Recommended: A100 (40GB or 80GB)
   - Budget option: RTX 4090 or RTX 3090
4. Container Settings:
   - Image: ${REGISTRY}/${CONTAINER_NAME}:${TAG}
   - Container Disk: 100GB
   - Expose Ports: 8888, 6006, 9090
5. Environment Variables:
   - CUDA_VISIBLE_DEVICES=0
   - TRAINING_MODE=quantum_mev
   - BATCH_SIZE=64
6. Click "Deploy"

⚡ OPTION 2: CLI Deployment (Advanced users)
--------------------------------------------
# Install RunPod CLI
npm install -g runpodctl

# Deploy instance
runpodctl create instance \\
  --image-name=${REGISTRY}/${CONTAINER_NAME}:${TAG} \\
  --gpu-type=A100 \\
  --container-disk=100GB \\
  --port=8888:8888 \\
  --port=6006:6006 \\
  --port=9090:9090 \\
  --env="CUDA_VISIBLE_DEVICES=0" \\
  --env="TRAINING_MODE=quantum_mev" \\
  --env="BATCH_SIZE=64"

🔗 Access Your Training Environment
-----------------------------------
After deployment, you'll get a Pod ID (e.g., abc123def456)
Access these services:

• Jupyter Lab: https://abc123def456-8888.proxy.runpod.net
• TensorBoard: https://abc123def456-6006.proxy.runpod.net
• Health Check: https://abc123def456-9090.proxy.runpod.net/health

📊 Monitor Training Progress
----------------------------
python3 monitor_training.py <your-pod-id>

💰 Estimated Costs
-------------------
• A100 80GB: ~\$2.89/hour
• A100 40GB: ~\$2.29/hour  
• RTX 4090: ~\$0.79/hour

🎯 What Happens Next
--------------------
1. Container will auto-start training pipeline
2. Jupyter Lab will be available for interactive development
3. TensorBoard will show real-time training metrics
4. Models will be saved to /workspace/models
5. Training logs available at /workspace/logs

✅ You're ready to start training!
EOF

print_success "Deployment instructions saved to: runpod_deploy_instructions.txt"

# Step 7: Final summary
print_step "7" "Deployment Summary"

echo ""
echo "🎉 TrenchBotAi RunPod Setup Complete!"
echo "====================================="
echo ""
echo "✅ Container built: ${CONTAINER_NAME}:${TAG}"
if [ "$REGISTRY" != "local" ]; then
    echo "✅ Pushed to registry: ${REGISTRY}/${CONTAINER_NAME}:${TAG}"
fi
echo "✅ Deployment instructions: runpod_deploy_instructions.txt"
echo ""
echo "🚀 Next Steps:"
echo "1. Read: runpod_deploy_instructions.txt"
echo "2. Go to: https://runpod.io/console/pods"
echo "3. Deploy with image: ${REGISTRY}/${CONTAINER_NAME}:${TAG}"
echo "4. Access Jupyter Lab and start training!"
echo ""
echo "🆘 Need Help?"
echo "- Check RUNPOD_DEPLOYMENT.md for detailed guide"
echo "- Monitor training with: python3 monitor_training.py <pod-id>"
echo ""

print_success "Ready for deployment! 🚀"
EOF