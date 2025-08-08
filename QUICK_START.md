# 🚀 TrenchBotAi RunPod Quick Start

## ⏱️ 5-Minute Setup

### Prerequisites Checklist
- [ ] **Docker installed** - [Get Docker](https://docs.docker.com/get-docker/)
- [ ] **RunPod account** - [Sign up](https://runpod.io) with GPU credits
- [ ] **Container registry access** - Docker Hub account (easiest option)

### Option A: Super Quick Deploy (Recommended)

```bash
# Run the interactive setup script
./startup_runpod.sh
```

The script will:
1. ✅ Check your system requirements
2. 🏗️ Build the optimized training container  
3. 🐳 Push to your container registry
4. 📋 Generate RunPod deployment instructions
5. 🚀 Give you direct deployment commands

### Option B: Manual Quick Deploy

```bash
# 1. Build container
docker build -f Dockerfile.complete -t trenchbot-trainer .

# 2. Tag for Docker Hub (replace 'yourusername')
docker tag trenchbot-trainer yourusername/trenchbot-trainer

# 3. Push to Docker Hub
docker login
docker push yourusername/trenchbot-trainer

# 4. Deploy on RunPod Web UI:
# - Go to runpod.io/console/pods
# - Image: yourusername/trenchbot-trainer
# - GPU: A100
# - Ports: 8888, 6006, 9090
```

## 🎯 Immediate Access

Once deployed, access your training environment:

| Service | URL | Purpose |
|---------|-----|---------|
| **Jupyter Lab** | `https://[pod-id]-8888.proxy.runpod.net` | Interactive development |
| **TensorBoard** | `https://[pod-id]-6006.proxy.runpod.net` | Training visualization |
| **Health API** | `https://[pod-id]-9090.proxy.runpod.net/health` | System monitoring |

## ⚡ What You Get

- **Complete AI training environment** with Jupyter, TensorBoard, monitoring
- **A100 GPU optimization** with CUDA 12.1, PyTorch 2.1, mixed precision
- **Quantum MEV models** ready for training
- **Multi-data source integration** (Helius, Solscan, Jupiter)
- **Automated training pipeline** with checkpointing and early stopping

## 💡 Pro Tips

1. **Start with A100 40GB** ($2.29/hour) for development
2. **Use Spot Instances** for 50% cost savings  
3. **Monitor via health endpoint** to track GPU utilization
4. **Save models frequently** - container storage is ephemeral
5. **Use Jupyter for experimentation** before full training runs

## 🆘 Need Help?

- Run: `./startup_runpod.sh` for guided setup
- Check: `RUNPOD_DEPLOYMENT.md` for detailed guide
- Monitor: `python3 monitor_training.py <pod-id>` for progress

**Ready to start training in 5 minutes!** 🚀