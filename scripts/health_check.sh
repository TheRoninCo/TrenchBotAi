#!/bin/bash

# Check GPU availability
nvidia-smi > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "❌ GPU not available"
    exit 1
fi

# Check if main process is running
pgrep -f trenchbot-dex > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "❌ TrenchBotAI process not running"
    exit 1
fi

# Check API endpoint
curl -f http://localhost:8080/health > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "❌ API endpoint not responding"
    exit 1
fi

echo "✅ All health checks passed"
exit 0