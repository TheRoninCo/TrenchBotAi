#!/bin/bash

# TrenchBot AI Performance Benchmark Runner
# Comprehensive testing suite for all cutting-edge components

set -e

echo "🚀 TrenchBot AI Performance Benchmarks"
echo "======================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create benchmark results directory
RESULTS_DIR="benchmark_results/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo -e "${BLUE}📁 Results will be saved to: $RESULTS_DIR${NC}"

# System information
echo -e "\n${YELLOW}🖥️  System Information${NC}"
echo "======================"
echo "Date: $(date)"
echo "OS: $(uname -s)"
echo "Architecture: $(uname -m)"
echo "CPU: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'Unknown')"
echo "Memory: $(sysctl -n hw.memsize 2>/dev/null | awk '{print $1/1024/1024/1024 " GB"}' || echo 'Unknown')"

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
    nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits | head -1 | awk '{print "GPU Memory: " $1 " MB total, " $2 " MB used, " $3 " MB free"}'
else
    echo "GPU: Not available or CUDA not detected"
fi

echo -e "\n${YELLOW}🔧 Build Configuration${NC}"
echo "======================"
echo "Rust version: $(rustc --version)"
echo "Cargo version: $(cargo --version)"

# Check features
echo "Features enabled:"
if grep -q 'gpu.*=.*\["tch"\]' Cargo.toml; then
    echo "  ✅ GPU acceleration (tch)"
else
    echo "  ❌ GPU acceleration disabled"
fi

if grep -q 'monitoring.*=.*\["metrics-exporter-prometheus"\]' Cargo.toml; then
    echo "  ✅ Monitoring (Prometheus)"
else
    echo "  ❌ Monitoring disabled"
fi

# Build project in release mode
echo -e "\n${YELLOW}🔨 Building Project${NC}"
echo "==================="
echo "Building in release mode for optimal performance..."
cargo build --release --features gpu 2>&1 | tee "$RESULTS_DIR/build.log"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo -e "${RED}❌ Build failed! Trying without GPU features...${NC}"
    cargo build --release 2>&1 | tee "$RESULTS_DIR/build_fallback.log"
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo -e "${RED}❌ Build failed completely! Check build logs.${NC}"
        exit 1
    fi
    echo -e "${YELLOW}⚠️  Built without GPU features${NC}"
fi

echo -e "${GREEN}✅ Build successful${NC}"

# Function to run benchmark with error handling
run_benchmark() {
    local bench_name=$1
    local description=$2
    
    echo -e "\n${BLUE}🧪 Running: $description${NC}"
    echo "=================================================="
    
    local output_file="$RESULTS_DIR/${bench_name}_results.txt"
    local timing_file="$RESULTS_DIR/${bench_name}_timing.txt"
    
    # Measure total time for benchmark
    local start_time=$(date +%s)
    
    if cargo bench --bench ai_performance_benchmarks -- "$bench_name" --output-format pretty 2>&1 | tee "$output_file"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        echo "Benchmark completed in ${duration}s" | tee "$timing_file"
        echo -e "${GREEN}✅ $description completed successfully${NC}"
        return 0
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        echo "Benchmark failed after ${duration}s" | tee "$timing_file"
        echo -e "${RED}❌ $description failed${NC}"
        return 1
    fi
}

# Run individual benchmark suites
echo -e "\n${YELLOW}🎯 Starting Benchmark Suites${NC}"
echo "==============================="

# Track success/failure counts
TOTAL_BENCHMARKS=0
SUCCESSFUL_BENCHMARKS=0

# Quantum Algorithm Benchmarks
TOTAL_BENCHMARKS=$((TOTAL_BENCHMARKS + 1))
if run_benchmark "quantum_algorithms" "Quantum-Inspired Algorithms"; then
    SUCCESSFUL_BENCHMARKS=$((SUCCESSFUL_BENCHMARKS + 1))
fi

# Transformer Model Benchmarks
TOTAL_BENCHMARKS=$((TOTAL_BENCHMARKS + 1))
if run_benchmark "transformer_models" "GPU-Accelerated Transformers"; then
    SUCCESSFUL_BENCHMARKS=$((SUCCESSFUL_BENCHMARKS + 1))
fi

# Graph Neural Network Benchmarks
TOTAL_BENCHMARKS=$((TOTAL_BENCHMARKS + 1))
if run_benchmark "graph_neural_networks" "Real-time Graph Neural Networks"; then
    SUCCESSFUL_BENCHMARKS=$((SUCCESSFUL_BENCHMARKS + 1))
fi

# Monte Carlo Simulation Benchmarks
TOTAL_BENCHMARKS=$((TOTAL_BENCHMARKS + 1))
if run_benchmark "monte_carlo_simulations" "CUDA Monte Carlo Simulations"; then
    SUCCESSFUL_BENCHMARKS=$((SUCCESSFUL_BENCHMARKS + 1))
fi

# Neural Architecture Search Benchmarks
TOTAL_BENCHMARKS=$((TOTAL_BENCHMARKS + 1))
if run_benchmark "neural_architecture_search" "Neural Architecture Search"; then
    SUCCESSFUL_BENCHMARKS=$((SUCCESSFUL_BENCHMARKS + 1))
fi

# Competitive Trading Benchmarks
TOTAL_BENCHMARKS=$((TOTAL_BENCHMARKS + 1))
if run_benchmark "competitive_trading" "Competitive Trading Strategies"; then
    SUCCESSFUL_BENCHMARKS=$((SUCCESSFUL_BENCHMARKS + 1))
fi

# Memory Usage Benchmarks
TOTAL_BENCHMARKS=$((TOTAL_BENCHMARKS + 1))
if run_benchmark "memory_usage" "Memory Usage Optimization"; then
    SUCCESSFUL_BENCHMARKS=$((SUCCESSFUL_BENCHMARKS + 1))
fi

# End-to-End Pipeline Benchmarks
TOTAL_BENCHMARKS=$((TOTAL_BENCHMARKS + 1))
if run_benchmark "end_to_end_pipeline" "Complete AI Pipeline"; then
    SUCCESSFUL_BENCHMARKS=$((SUCCESSFUL_BENCHMARKS + 1))
fi

# Generate comprehensive report
echo -e "\n${YELLOW}📊 Generating Comprehensive Report${NC}"
echo "===================================="

REPORT_FILE="$RESULTS_DIR/benchmark_report.md"

cat > "$REPORT_FILE" << EOF
# TrenchBot AI Performance Benchmark Report

**Generated:** $(date)  
**System:** $(uname -s) $(uname -m)  
**Rust Version:** $(rustc --version)  

## Summary

- **Total Benchmark Suites:** $TOTAL_BENCHMARKS
- **Successful:** $SUCCESSFUL_BENCHMARKS
- **Failed:** $((TOTAL_BENCHMARKS - SUCCESSFUL_BENCHMARKS))
- **Success Rate:** $(( (SUCCESSFUL_BENCHMARKS * 100) / TOTAL_BENCHMARKS ))%

## System Specifications

- **OS:** $(uname -s)
- **Architecture:** $(uname -m)
- **CPU:** $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'Unknown')
- **Memory:** $(sysctl -n hw.memsize 2>/dev/null | awk '{print $1/1024/1024/1024 " GB"}' || echo 'Unknown')
EOF

if command -v nvidia-smi &> /dev/null; then
    cat >> "$REPORT_FILE" << EOF
- **GPU:** $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
- **GPU Memory:** $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1) MB
EOF
else
    echo "- **GPU:** Not available" >> "$REPORT_FILE"
fi

cat >> "$REPORT_FILE" << EOF

## Benchmark Results

EOF

# Process each benchmark result
for result_file in "$RESULTS_DIR"/*_results.txt; do
    if [ -f "$result_file" ]; then
        benchmark_name=$(basename "$result_file" _results.txt)
        echo "### $benchmark_name" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
        echo '```' >> "$REPORT_FILE"
        tail -20 "$result_file" >> "$REPORT_FILE"
        echo '```' >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
    fi
done

cat >> "$REPORT_FILE" << EOF

## Performance Analysis

### Key Metrics Observed:

1. **Quantum Algorithms:** Superposition analysis and interference calculations
2. **Transformers:** Attention mechanism throughput and memory efficiency
3. **Graph Networks:** Real-time transaction processing and pattern detection
4. **Monte Carlo:** Simulation speed and variance reduction effectiveness
5. **NAS:** Architecture discovery and optimization convergence
6. **Trading Strategies:** Signal generation and order routing performance

### Recommendations:

Based on the benchmark results, consider the following optimizations:

- **GPU Utilization:** $(if command -v nvidia-smi &> /dev/null; then echo "GPU detected - leverage CUDA acceleration"; else echo "No GPU detected - consider adding GPU support"; fi)
- **Memory Optimization:** Monitor memory usage patterns for large datasets
- **Parallel Processing:** Utilize multi-core processing for CPU-bound operations
- **Caching:** Implement intelligent caching for frequently accessed computations

## Files Generated:

EOF

ls -la "$RESULTS_DIR" | tail -n +2 | awk '{print "- " $9 " (" $5 " bytes)"}' >> "$REPORT_FILE"

echo -e "${GREEN}📋 Report generated: $REPORT_FILE${NC}"

# Create summary statistics
echo -e "\n${YELLOW}📈 Performance Summary${NC}"
echo "======================"

# Find the fastest and slowest benchmarks
echo "Analyzing benchmark timings..."
if ls "$RESULTS_DIR"/*_timing.txt 1> /dev/null 2>&1; then
    echo "Benchmark completion times:"
    for timing_file in "$RESULTS_DIR"/*_timing.txt; do
        if [ -f "$timing_file" ]; then
            benchmark_name=$(basename "$timing_file" _timing.txt)
            time_info=$(cat "$timing_file")
            echo "  $benchmark_name: $time_info"
        fi
    done
else
    echo "No timing files found."
fi

# Generate performance dashboard data (JSON)
DASHBOARD_FILE="$RESULTS_DIR/performance_dashboard.json"
cat > "$DASHBOARD_FILE" << EOF
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "system": {
    "os": "$(uname -s)",
    "arch": "$(uname -m)",
    "rust_version": "$(rustc --version)",
    "gpu_available": $(if command -v nvidia-smi &> /dev/null; then echo "true"; else echo "false"; fi)
  },
  "summary": {
    "total_benchmarks": $TOTAL_BENCHMARKS,
    "successful": $SUCCESSFUL_BENCHMARKS,
    "failed": $((TOTAL_BENCHMARKS - SUCCESSFUL_BENCHMARKS)),
    "success_rate": $(( (SUCCESSFUL_BENCHMARKS * 100) / TOTAL_BENCHMARKS ))
  },
  "results_directory": "$RESULTS_DIR"
}
EOF

echo -e "${GREEN}📊 Dashboard data: $DASHBOARD_FILE${NC}"

# Final summary
echo -e "\n${BLUE}🎯 Benchmark Summary${NC}"
echo "===================="
echo -e "Total Suites: $TOTAL_BENCHMARKS"
echo -e "Successful: ${GREEN}$SUCCESSFUL_BENCHMARKS${NC}"
echo -e "Failed: ${RED}$((TOTAL_BENCHMARKS - SUCCESSFUL_BENCHMARKS))${NC}"
echo -e "Success Rate: ${GREEN}$(( (SUCCESSFUL_BENCHMARKS * 100) / TOTAL_BENCHMARKS ))%${NC}"

if [ $SUCCESSFUL_BENCHMARKS -eq $TOTAL_BENCHMARKS ]; then
    echo -e "\n${GREEN}🎉 All benchmarks completed successfully!${NC}"
    echo -e "${GREEN}🚀 TrenchBot AI is performing optimally.${NC}"
else
    echo -e "\n${YELLOW}⚠️  Some benchmarks failed or had issues.${NC}"
    echo -e "${YELLOW}📝 Check the individual result files for details.${NC}"
fi

echo -e "\n${BLUE}📁 All results saved to: $RESULTS_DIR${NC}"
echo -e "${BLUE}📋 Full report: $REPORT_FILE${NC}"

# Optional: Open report in default viewer
if command -v open &> /dev/null; then
    echo -e "\n${YELLOW}Would you like to open the report? (y/n)${NC}"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        open "$REPORT_FILE"
    fi
fi

echo -e "\n${GREEN}✅ Benchmark suite completed!${NC}"