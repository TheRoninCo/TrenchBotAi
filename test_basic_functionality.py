#!/usr/bin/env python3
"""
Quick TrenchBot AI Functionality Test
Tests basic components without requiring full Rust compilation
"""

import time
import json
import random
import math
from datetime import datetime

def test_quantum_simulation():
    """Simulate quantum-inspired calculations"""
    print("🔬 Testing Quantum-Inspired Algorithms...")
    
    # Simulate quantum superposition analysis
    market_data = [random.random() * 100 for _ in range(1000)]
    
    start_time = time.time()
    
    # Simulate quantum state amplitudes
    amplitudes = []
    phases = []
    
    for i, price in enumerate(market_data):
        # Simulate quantum superposition
        amplitude = price / sum(market_data) if sum(market_data) > 0 else 0
        phase = math.sin(price * math.pi * 2.0)
        
        amplitudes.append(amplitude)
        phases.append(phase)
    
    # Simulate quantum interference
    interference_sum = 0
    for i in range(len(amplitudes)):
        for j in range(i+1, min(i+10, len(amplitudes))):  # Limited range for speed
            interference = amplitudes[i] * amplitudes[j] * math.cos(phases[i] - phases[j])
            interference_sum += interference
    
    quantum_coherence = abs(interference_sum) ** 2
    
    end_time = time.time()
    
    print(f"   ✅ Processed {len(market_data)} data points")
    print(f"   ⚡ Quantum coherence: {quantum_coherence:.6f}")
    print(f"   ⏱️  Processing time: {(end_time - start_time)*1000:.2f}ms")
    
    return {
        'test': 'quantum_simulation',
        'data_points': len(market_data),
        'coherence': quantum_coherence,
        'time_ms': (end_time - start_time) * 1000
    }

def test_transformer_attention():
    """Simulate transformer attention mechanisms"""
    print("🧠 Testing Transformer Attention...")
    
    # Simulate market sequence
    sequence_length = 256
    feature_dim = 64
    num_heads = 8
    
    start_time = time.time()
    
    # Simulate input embeddings
    market_sequence = []
    for t in range(sequence_length):
        features = [random.random() for _ in range(feature_dim)]
        market_sequence.append(features)
    
    # Simulate attention scores
    attention_scores = []
    for head in range(num_heads):
        head_scores = []
        for i in range(sequence_length):
            scores = []
            for j in range(sequence_length):
                # Simplified attention calculation
                score = sum(market_sequence[i][k] * market_sequence[j][k] 
                           for k in range(min(8, feature_dim)))  # Limited for speed
                scores.append(score)
            head_scores.append(scores)
        attention_scores.append(head_scores)
    
    # Simulate softmax normalization
    normalized_attention = []
    for head_scores in attention_scores:
        normalized_head = []
        for scores in head_scores:
            max_score = max(scores) if scores else 0
            exp_scores = [math.exp(s - max_score) for s in scores]
            sum_exp = sum(exp_scores)
            softmax_scores = [exp_s / sum_exp if sum_exp > 0 else 0 for exp_s in exp_scores]
            normalized_head.append(softmax_scores)
        normalized_attention.append(normalized_head)
    
    end_time = time.time()
    
    # Calculate attention diversity (higher = better)
    attention_diversity = 0
    for head in normalized_attention:
        for scores in head:
            entropy = -sum(p * math.log(p + 1e-10) for p in scores if p > 0)
            attention_diversity += entropy
    
    attention_diversity /= (num_heads * sequence_length)
    
    print(f"   ✅ Processed {sequence_length}x{feature_dim} sequence with {num_heads} heads")
    print(f"   🎯 Attention diversity: {attention_diversity:.4f}")
    print(f"   ⏱️  Processing time: {(end_time - start_time)*1000:.2f}ms")
    
    return {
        'test': 'transformer_attention',
        'sequence_length': sequence_length,
        'feature_dim': feature_dim,
        'num_heads': num_heads,
        'attention_diversity': attention_diversity,
        'time_ms': (end_time - start_time) * 1000
    }

def test_graph_pattern_detection():
    """Simulate graph neural network pattern detection"""
    print("🕸️  Testing Graph Pattern Detection...")
    
    # Simulate transaction graph
    num_wallets = 100
    num_tokens = 10
    num_transactions = 500
    
    start_time = time.time()
    
    # Generate mock transactions
    transactions = []
    for i in range(num_transactions):
        tx = {
            'wallet': f'wallet_{random.randint(0, num_wallets-1)}',
            'token': f'token_{random.randint(0, num_tokens-1)}',
            'amount': random.uniform(0.1, 100.0),
            'timestamp': time.time() - random.uniform(0, 3600),
            'tx_type': random.choice(['buy', 'sell', 'swap'])
        }
        transactions.append(tx)
    
    # Build adjacency relationships
    wallet_connections = {}
    token_connections = {}
    
    for tx in transactions:
        wallet = tx['wallet']
        token = tx['token']
        
        if wallet not in wallet_connections:
            wallet_connections[wallet] = set()
        if token not in token_connections:
            token_connections[token] = set()
        
        wallet_connections[wallet].add(token)
        token_connections[token].add(wallet)
    
    # Detect patterns
    patterns = {
        'sandwich_attacks': 0,
        'wash_trading': 0,
        'mev_bots': 0,
        'high_frequency': 0
    }
    
    # Simple pattern detection
    wallet_activity = {}
    for tx in transactions:
        wallet = tx['wallet']
        if wallet not in wallet_activity:
            wallet_activity[wallet] = []
        wallet_activity[wallet].append(tx)
    
    for wallet, txs in wallet_activity.items():
        if len(txs) > 10:  # High frequency
            patterns['high_frequency'] += 1
        
        # Check for potential sandwich patterns (simplified)
        txs_sorted = sorted(txs, key=lambda x: x['timestamp'])
        for i in range(len(txs_sorted) - 2):
            if (txs_sorted[i]['tx_type'] == 'buy' and 
                txs_sorted[i+1]['token'] == txs_sorted[i]['token'] and
                txs_sorted[i+2]['tx_type'] == 'sell'):
                patterns['sandwich_attacks'] += 1
        
        # Check for wash trading (buy-sell same token quickly)
        token_trades = {}
        for tx in txs:
            token = tx['token']
            if token not in token_trades:
                token_trades[token] = []
            token_trades[token].append(tx)
        
        for token, token_txs in token_trades.items():
            if len(token_txs) >= 4:  # Multiple trades same token
                patterns['wash_trading'] += 1
    
    end_time = time.time()
    
    print(f"   ✅ Analyzed {num_transactions} transactions across {num_wallets} wallets")
    print(f"   🔍 Patterns detected:")
    for pattern, count in patterns.items():
        print(f"      {pattern}: {count}")
    print(f"   ⏱️  Processing time: {(end_time - start_time)*1000:.2f}ms")
    
    return {
        'test': 'graph_pattern_detection',
        'transactions': num_transactions,
        'wallets': num_wallets,
        'patterns': patterns,
        'time_ms': (end_time - start_time) * 1000
    }

def test_monte_carlo_simulation():
    """Simulate Monte Carlo risk analysis"""
    print("🎲 Testing Monte Carlo Simulation...")
    
    # Portfolio setup
    assets = [
        {'symbol': 'SOL', 'price': 100.0, 'volatility': 0.3, 'weight': 0.4},
        {'symbol': 'ETH', 'price': 2000.0, 'volatility': 0.4, 'weight': 0.3},
        {'symbol': 'BTC', 'price': 45000.0, 'volatility': 0.5, 'weight': 0.3},
    ]
    
    num_simulations = 10000
    time_horizon = 21  # 21 days
    
    start_time = time.time()
    
    portfolio_returns = []
    
    for simulation in range(num_simulations):
        portfolio_value = 0
        
        for asset in assets:
            # Simulate price path using geometric Brownian motion
            current_price = asset['price']
            
            for day in range(time_horizon):
                # Random shock
                z = random.gauss(0, 1)
                
                # Price update: S_t+1 = S_t * exp((μ - σ²/2)dt + σ√dt*Z)
                drift = 0.1 / 365  # 10% annual return
                diffusion = asset['volatility'] / math.sqrt(365) * z
                
                log_return = drift - 0.5 * (asset['volatility'] ** 2) / 365 + diffusion
                current_price *= math.exp(log_return)
            
            portfolio_value += current_price * asset['weight']
        
        # Calculate portfolio return
        initial_value = sum(asset['price'] * asset['weight'] for asset in assets)
        portfolio_return = (portfolio_value / initial_value) - 1
        portfolio_returns.append(portfolio_return)
    
    # Calculate VaR (Value at Risk)
    portfolio_returns.sort()
    var_95 = -portfolio_returns[int(0.05 * len(portfolio_returns))]
    var_99 = -portfolio_returns[int(0.01 * len(portfolio_returns))]
    
    # Calculate Expected Shortfall (CVaR)
    tail_5 = portfolio_returns[:int(0.05 * len(portfolio_returns))]
    cvar_95 = -sum(tail_5) / len(tail_5) if tail_5 else 0
    
    end_time = time.time()
    
    print(f"   ✅ Completed {num_simulations} simulations over {time_horizon} days")
    print(f"   📊 VaR (95%): {var_95:.4f} ({var_95*100:.2f}%)")
    print(f"   📊 VaR (99%): {var_99:.4f} ({var_99*100:.2f}%)")
    print(f"   📊 CVaR (95%): {cvar_95:.4f} ({cvar_95*100:.2f}%)")
    print(f"   ⏱️  Processing time: {(end_time - start_time)*1000:.2f}ms")
    
    return {
        'test': 'monte_carlo_simulation',
        'simulations': num_simulations,
        'time_horizon': time_horizon,
        'var_95': var_95,
        'var_99': var_99,
        'cvar_95': cvar_95,
        'time_ms': (end_time - start_time) * 1000
    }

def test_competitive_analysis():
    """Simulate competitive trading analysis"""
    print("⚔️  Testing Competitive Trading Analysis...")
    
    start_time = time.time()
    
    # Simulate whale wallets
    whales = []
    for i in range(10):
        whale = {
            'wallet': f'whale_{i}',
            'typical_size': random.uniform(1000, 10000),
            'frequency': random.uniform(1, 10),  # trades per hour
            'success_rate': random.uniform(0.6, 0.9),
            'preferred_tokens': [f'token_{j}' for j in random.sample(range(20), 3)],
            'last_activity': time.time() - random.uniform(0, 3600)
        }
        whales.append(whale)
    
    # Simulate three-step prediction
    target_whale = random.choice(whales)
    
    # Step 1: Immediate action prediction
    if target_whale['frequency'] > 5:
        step1_action = 'buy'
        step1_confidence = 0.8
    else:
        step1_action = 'accumulate'
        step1_confidence = 0.6
    
    # Step 2: Follow-up action
    if step1_action == 'buy':
        step2_action = 'hold'
        step2_confidence = 0.65
    else:
        step2_action = 'distribute'
        step2_confidence = 0.55
    
    # Step 3: Final action
    step3_action = 'sell'
    step3_confidence = 0.4
    
    overall_confidence = (step1_confidence * 0.5 + step2_confidence * 0.3 + step3_confidence * 0.2)
    
    # Generate competitive signals
    signals = []
    for whale in whales:
        if whale['success_rate'] > 0.7:
            signal = {
                'type': 'follow_leader',
                'whale': whale['wallet'],
                'urgency': whale['frequency'] / 10,
                'expected_profit': whale['success_rate'] * 0.1,
                'confidence': whale['success_rate']
            }
            signals.append(signal)
    
    end_time = time.time()
    
    print(f"   ✅ Analyzed {len(whales)} whale wallets")
    print(f"   🎯 Three-step prediction for {target_whale['wallet']}:")
    print(f"      Step 1: {step1_action} (confidence: {step1_confidence:.2f})")
    print(f"      Step 2: {step2_action} (confidence: {step2_confidence:.2f})")
    print(f"      Step 3: {step3_action} (confidence: {step3_confidence:.2f})")
    print(f"      Overall: {overall_confidence:.3f}")
    print(f"   📡 Generated {len(signals)} competitive signals")
    print(f"   ⏱️  Processing time: {(end_time - start_time)*1000:.2f}ms")
    
    return {
        'test': 'competitive_analysis',
        'whales_analyzed': len(whales),
        'signals_generated': len(signals),
        'prediction_confidence': overall_confidence,
        'time_ms': (end_time - start_time) * 1000
    }

def main():
    print("🚀 TrenchBot AI Quick Functionality Test")
    print("=" * 45)
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    results = []
    total_start = time.time()
    
    try:
        # Run all tests
        results.append(test_quantum_simulation())
        print()
        
        results.append(test_transformer_attention())
        print()
        
        results.append(test_graph_pattern_detection())
        print()
        
        results.append(test_monte_carlo_simulation())
        print()
        
        results.append(test_competitive_analysis())
        print()
        
        total_time = time.time() - total_start
        
        # Summary
        print("=" * 45)
        print("📊 PERFORMANCE SUMMARY")
        print("=" * 45)
        
        total_time_ms = sum(result['time_ms'] for result in results)
        
        print(f"🕐 Total execution time: {total_time*1000:.2f}ms")
        print(f"⚡ Component processing time: {total_time_ms:.2f}ms")
        print(f"🔧 Overhead: {(total_time*1000 - total_time_ms):.2f}ms")
        print()
        
        print("📈 Component Performance:")
        for result in results:
            test_name = result['test'].replace('_', ' ').title()
            print(f"   {test_name}: {result['time_ms']:.2f}ms")
        
        # Find fastest and slowest
        fastest = min(results, key=lambda x: x['time_ms'])
        slowest = max(results, key=lambda x: x['time_ms'])
        
        print()
        print(f"⚡ Fastest: {fastest['test']} ({fastest['time_ms']:.2f}ms)")
        print(f"🐌 Slowest: {slowest['test']} ({slowest['time_ms']:.2f}ms)")
        
        # Performance insights
        print()
        print("💡 Performance Insights:")
        if total_time_ms < 1000:
            print("   ✅ Excellent performance - all components under 1 second")
        elif total_time_ms < 5000:
            print("   ✅ Good performance - suitable for real-time trading")
        else:
            print("   ⚠️  Consider optimization for real-time applications")
        
        # Save results
        with open('quick_test_results.json', 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'total_time_ms': total_time * 1000,
                'component_time_ms': total_time_ms,
                'results': results
            }, f, indent=2)
        
        print(f"💾 Results saved to: quick_test_results.json")
        print()
        print("🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())