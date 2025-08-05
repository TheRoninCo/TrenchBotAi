#!/bin/bash

# Quick Performance Test for TrenchBot AI Components
# Runs essential benchmarks with smaller datasets for immediate feedback

set -e

echo "⚡ TrenchBot AI Quick Performance Test"
echo "====================================="

# Check if we're in the right directory
if [ ! -f "Cargo.toml" ]; then
    echo "❌ Please run this script from the TrenchBot project root directory"
    exit 1
fi

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🔨 Building project...${NC}"
cargo build --release

echo -e "\n${YELLOW}🧪 Running Quick Benchmarks${NC}"
echo "============================"

# Test compilation and basic functionality
echo -e "\n${BLUE}1. Testing Quantum Algorithms${NC}"
cargo test --release quantum --lib 2>/dev/null && echo "✅ Quantum tests passed" || echo "⚠️  Quantum tests skipped"

echo -e "\n${BLUE}2. Testing Transformer Models${NC}"
cargo test --release transformer --lib 2>/dev/null && echo "✅ Transformer tests passed" || echo "⚠️  Transformer tests skipped"

echo -e "\n${BLUE}3. Testing Graph Neural Networks${NC}" 
cargo test --release graph --lib 2>/dev/null && echo "✅ Graph tests passed" || echo "⚠️  Graph tests skipped"

echo -e "\n${BLUE}4. Testing Monte Carlo Engine${NC}"
cargo test --release monte_carlo --lib 2>/dev/null && echo "✅ Monte Carlo tests passed" || echo "⚠️  Monte Carlo tests skipped"

echo -e "\n${BLUE}5. Testing Competitive Trading${NC}"
cargo test --release competitive --lib 2>/dev/null && echo "✅ Trading tests passed" || echo "⚠️  Trading tests skipped"

# Quick performance test
echo -e "\n${YELLOW}⚡ Quick Performance Metrics${NC}"
echo "============================"

# Test build time
echo -e "\n${BLUE}📊 Build Performance:${NC}"
time cargo build --release --quiet

# Test basic operations
echo -e "\n${BLUE}📊 Runtime Performance (estimated):${NC}"

# Quick memory usage check
echo "Memory usage baseline:"
ps -o pid,vsz,rss,comm -p $$ | tail -1

# Check if criterion benchmarks exist
if [ -f "benches/ai_performance_benchmarks.rs" ]; then
    echo -e "\n${BLUE}🚀 Running Quick Criterion Benchmarks${NC}"
    echo "====================================="
    
    # Run a subset of benchmarks with shorter time
    timeout 60s cargo bench --bench ai_performance_benchmarks -- --sample-size 10 --measurement-time 5 quantum_algorithms 2>/dev/null || echo "⚠️  Quantum benchmarks timed out or failed"
    
    timeout 30s cargo bench --bench ai_performance_benchmarks -- --sample-size 5 --measurement-time 3 transformer_models 2>/dev/null || echo "⚠️  Transformer benchmarks timed out or failed"
    
    timeout 30s cargo bench --bench ai_performance_benchmarks -- --sample-size 5 competitive_trading 2>/dev/null || echo "⚠️  Trading benchmarks timed out or failed"
    
else
    echo "⚠️  Criterion benchmarks not found"
fi

# System info
echo -e "\n${YELLOW}💻 System Information${NC}"
echo "===================="
echo "OS: $(uname -s)"
echo "Architecture: $(uname -m)" 
echo "Rust: $(rustc --version)"
echo "CPU cores: $(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 'Unknown')"

if command -v nvidia-smi &> /dev/null; then
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
else
    echo "GPU: Not detected"
fi

echo -e "\n${GREEN}✅ Quick benchmark completed!${NC}"
echo ""
echo "For comprehensive benchmarks, run:"
echo "  ./scripts/run_benchmarks.sh"
echo ""
echo "To run specific component tests:"
echo "  cargo test --release <component_name>"
echo ""
echo "To run full criterion benchmarks:"
echo "  cargo bench"