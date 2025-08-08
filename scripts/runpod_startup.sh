#!/bin/bash
# TrenchBot AI - RunPod Startup Script
# Comprehensive GPU Training System Deployment

set -e

echo "🚀 TrenchBot AI - RunPod Deployment Starting..."
echo "================================================"

# Environment setup
export RUST_LOG=info
export CUDA_VISIBLE_DEVICES=0
export TRENCHBOT_MODE=gpu_training
export MODEL_SAVE_PATH=/workspace/models

# System info logging
echo "📊 System Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
echo "CPU Cores: $(nproc)"
echo "RAM Total: $(free -h | grep Mem | awk '{print $2}')"
echo "RAM Available: $(free -h | grep Mem | awk '{print $7}')"
echo "Disk Available: $(df -h /workspace | tail -1 | awk '{print $4}')"
echo "CUDA Version: $(nvcc --version | grep release)"

# Create directories
echo "📁 Creating workspace directories..."
mkdir -p /workspace/{models,data,logs,checkpoints}
chmod 755 /workspace/{models,data,logs,checkpoints}

# Navigate to TrenchBot directory
cd /workspace/TrenchBotAi

# Verify GPU availability
echo "🔍 Verifying GPU availability..."
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}'); print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Build TrenchBot with GPU features
echo "🔨 Building TrenchBot with GPU acceleration..."
export CUDA_ROOT=/usr/local/cuda
export LD_LIBRARY_PATH="$CUDA_ROOT/lib64:$LD_LIBRARY_PATH"
export PATH="$CUDA_ROOT/bin:$PATH"

# Clean previous builds
cargo clean

# Build with GPU features
echo "Building TrenchBot release binary..."
cargo build --release --features="gpu"

# Build GPU training demo
echo "Building GPU training demo..."
cargo build --release --features="gpu" --example gpu_training_demo

# Verify builds succeeded
if [ ! -f "target/release/trenchbot_ai" ] || [ ! -f "target/release/examples/gpu_training_demo" ]; then
    echo "❌ Build failed - checking for compilation errors..."
    cargo check --features="gpu" 2>&1 | head -50
    exit 1
fi

echo "✅ Build completed successfully!"

# Start Redis (for coordination caching)
echo "🔧 Starting Redis server..."
redis-server --daemonize yes --bind 127.0.0.1 --port 6379

# Health check endpoint setup
echo "🏥 Setting up health monitoring..."
cat > /tmp/health_check.py << 'EOF'
#!/usr/bin/env python3
import http.server
import socketserver
import json
import subprocess
import threading
import time

class HealthHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            try:
                # Check GPU status
                gpu_check = subprocess.run(['nvidia-smi', '-q'], capture_output=True, text=True)
                gpu_ok = gpu_check.returncode == 0
                
                # Check disk space
                disk_check = subprocess.run(['df', '/workspace'], capture_output=True, text=True)
                disk_ok = disk_check.returncode == 0
                
                health_data = {
                    "status": "healthy" if gpu_ok and disk_ok else "degraded",
                    "gpu_available": gpu_ok,
                    "disk_ok": disk_ok,
                    "timestamp": time.time()
                }
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(health_data).encode())
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                error_data = {"status": "error", "message": str(e)}
                self.wfile.write(json.dumps(error_data).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress default logging

# Start health check server in background
def start_health_server():
    with socketserver.TCPServer(("", 8080), HealthHandler) as httpd:
        httpd.serve_forever()

threading.Thread(target=start_health_server, daemon=True).start()
print("Health check server started on port 8080")
time.sleep(1)  # Give server time to start
EOF

python3 /tmp/health_check.py &
HEALTH_PID=$!

# Start main TrenchBot training
echo "🧠 Starting TrenchBot AI GPU Training Pipeline..."
echo "================================================"

# Launch GPU training demo with resource monitoring
echo "Starting comprehensive GPU training..."
cargo run --release --features="gpu" --example gpu_training_demo 2>&1 | tee /workspace/logs/training.log &
TRAINING_PID=$!

# Monitor system resources
echo "📈 Starting resource monitoring..."
cat > /tmp/monitor.py << 'EOF'
#!/usr/bin/env python3
import time
import json
import subprocess
import psutil
import GPUtil

def get_system_stats():
    try:
        # Get GPU stats
        gpus = GPUtil.getGPUs()
        gpu_stats = []
        for gpu in gpus:
            gpu_stats.append({
                "id": gpu.id,
                "name": gpu.name,
                "load": gpu.load * 100,
                "memory_used": gpu.memoryUsed,
                "memory_total": gpu.memoryTotal,
                "memory_percent": (gpu.memoryUsed / gpu.memoryTotal) * 100,
                "temperature": gpu.temperature
            })
        
        # Get CPU and memory stats
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/workspace')
        
        stats = {
            "timestamp": time.time(),
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_used_gb": memory.used / (1024**3),
            "memory_total_gb": memory.total / (1024**3),
            "disk_used_gb": disk.used / (1024**3),
            "disk_free_gb": disk.free / (1024**3),
            "disk_percent": (disk.used / disk.total) * 100,
            "gpus": gpu_stats
        }
        
        return stats
    except Exception as e:
        return {"error": str(e), "timestamp": time.time()}

# Monitor loop
print("System monitoring started...")
while True:
    stats = get_system_stats()
    
    # Print summary every 30 seconds
    if int(time.time()) % 30 == 0:
        print(f"\n🔥 TrenchBot System Status:")
        print(f"   CPU: {stats.get('cpu_percent', 0):.1f}%")
        print(f"   Memory: {stats.get('memory_percent', 0):.1f}%")
        print(f"   Disk: {stats.get('disk_percent', 0):.1f}%")
        
        for gpu in stats.get('gpus', []):
            print(f"   GPU {gpu['id']} ({gpu['name']}):")
            print(f"     Load: {gpu['load']:.1f}%")
            print(f"     Memory: {gpu['memory_percent']:.1f}% ({gpu['memory_used']:.1f}GB/{gpu['memory_total']:.1f}GB)")
            print(f"     Temp: {gpu['temperature']}°C")
    
    # Save stats to log file every 10 seconds
    if int(time.time()) % 10 == 0:
        with open('/workspace/logs/system_stats.json', 'a') as f:
            json.dump(stats, f)
            f.write('\n')
    
    time.sleep(5)
EOF

python3 /tmp/monitor.py &
MONITOR_PID=$!

echo "🎯 TrenchBot AI Deployment Complete!"
echo "===================================="
echo ""
echo "🔥 Services Running:"
echo "   • GPU Training Pipeline (PID: $TRAINING_PID)"
echo "   • System Monitor (PID: $MONITOR_PID)" 
echo "   • Health Check API (PID: $HEALTH_PID)"
echo "   • Redis Cache Server"
echo ""
echo "📊 Monitoring:"
echo "   • Health: http://localhost:8080/health"
echo "   • Training logs: /workspace/logs/training.log"
echo "   • System stats: /workspace/logs/system_stats.json"
echo ""
echo "💾 Model Storage:"
echo "   • Trained models: /workspace/models/"
echo "   • Training data: /workspace/data/"
echo "   • Checkpoints: /workspace/checkpoints/"
echo ""
echo "🚀 Ready for MEV detection and AI training!"

# Keep container running and monitor training
wait $TRAINING_PID

echo "🎉 Training completed! Models saved to /workspace/models/"
echo "🔍 Check /workspace/logs/training.log for detailed results"

# Keep health monitoring running
wait