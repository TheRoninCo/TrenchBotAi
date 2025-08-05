#!/bin/bash
# Deploy to RunPod for GPU training

echo "🚀 Deploying to RunPod..."

# Build and push Docker image
docker build -f Dockerfile.runpod -t mev-bot:gpu .

# Upload to RunPod (customize with your RunPod setup)
echo "Upload this image to your RunPod instance"
echo "Then run: docker run --gpus all mev-bot:gpu"
