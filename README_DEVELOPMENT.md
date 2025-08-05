# MEV Bot Development Workflow

## MacBook Development (CPU)
```bash
# Setup environment
./setup_dev_env.sh

# Run in development mode
cargo run --features local-dev

# Watch for changes
cargo watch -x "run --features local-dev"

# Test sandwich detection
cargo test sandwich_tests
```

## RunPod Deployment (GPU)
```bash
# Build for GPU
cargo build --release --features gpu,training

# Deploy to RunPod
./deploy_to_runpod.sh
```

## Testing Your Sandwich Formula
1. Implement your algorithm in `src/engines/mev/sandwich.rs`
2. Test locally with simulated data
3. Deploy to RunPod for alpha wallet hunting
4. Train AI models on GPU for pattern recognition

## Development Tips
- Use `local-dev` feature for MacBook testing
- Enable `gpu` feature only on RunPod
- Your sandwich formula goes in `calculate_front_run_amount()` and `estimate_profit()`
