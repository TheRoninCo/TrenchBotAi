# 🔥 TrenchBot AI - One-Click RunPod Deployment

## 🚀 Foolproof Deployment Guide

This deployment focuses on **proven, working systems** only:
- ✅ Flash Loan System (Solend + Mango Markets)
- ✅ MEV Detection Engine  
- ✅ Atomic Transaction Execution
- ✅ Liquidation Opportunity Scanner
- ✅ RESTful API Server

---

## 🎯 One-Click Deployment

### Step 1: Start RunPod Instance
1. Go to [runpod.io](https://runpod.io) → **Pods** → **Deploy**
2. Select **A100 SXM (40GB)** GPU
3. Choose **Ubuntu 22.04 + CUDA 12.1** template
4. Set **Container Disk**: 50GB, **Volume**: 20GB
5. **Deploy Pod**

### Step 2: Run Deployment Script
Open pod terminal and run:

```bash
curl -fsSL https://raw.githubusercontent.com/yourusername/TrenchBotAi/main/deploy_runpod.sh | bash
```

**That's it! 🎉** System auto-installs and starts on port 8080.

---

## 🧪 Test Your Deployment

### Quick Health Check
```bash
curl http://localhost:8080/status
```

### Test Flash Loans
```bash
# Arbitrage test
curl 'http://localhost:8080/flash-loan/test?operation=arbitrage&amount=1000000000'

# Liquidation test  
curl 'http://localhost:8080/flash-loan/test?operation=liquidation&amount=5000000000'
```

### Test MEV Detection
```bash
curl http://localhost:8080/mev/detect
```

---

## ⚙️ Configuration

Edit `.env` file for your API keys:
```bash
HELIUS_API_KEY=your_helius_key_here
SOLSCAN_API_KEY=your_solscan_key_here
JUPITER_API_KEY=your_jupiter_key_here
```

---

## 🎉 Success Indicator

If you see this from `curl http://localhost:8080/status`:
```json
{
  "status": "running",
  "flash_loan_ready": true,
  "mev_detection_ready": true,
  "uptime_seconds": 42
}
```

**🎊 Congratulations! Your TrenchBot AI is live!** 

You now have a production-ready Solana flash loan and MEV detection system running on RunPod!