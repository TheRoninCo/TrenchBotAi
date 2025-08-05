#!/usr/bin/env python3
"""
TrenchBot AI Performance Monitor
Real-time performance metrics and benchmark result analysis
"""

import json
import sys
import time
import os
import subprocess
from datetime import datetime
from pathlib import Path
import argparse

def get_system_info():
    """Get current system information"""
    try:
        # CPU usage
        cpu_info = subprocess.check_output(['sysctl', '-n', 'hw.ncpu'], text=True).strip()
        
        # Memory info
        mem_info = subprocess.check_output(['sysctl', '-n', 'hw.memsize'], text=True).strip()
        mem_gb = int(mem_info) / (1024**3)
        
        # Load average
        load_avg = subprocess.check_output(['sysctl', '-n', 'vm.loadavg'], text=True).strip()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'cpu_cores': int(cpu_info),
            'memory_gb': round(mem_gb, 2),
            'load_average': load_avg,
        }
    except subprocess.CalledProcessError:
        return {
            'timestamp': datetime.now().isoformat(),
            'cpu_cores': 'unknown',
            'memory_gb': 'unknown',
            'load_average': 'unknown',
        }

def analyze_benchmark_results(results_dir):
    """Analyze benchmark results from a results directory"""
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return None
    
    analysis = {
        'directory': str(results_path),
        'timestamp': datetime.now().isoformat(),
        'benchmarks': {},
        'summary': {}
    }
    
    # Look for result files
    result_files = list(results_path.glob('*_results.txt'))
    timing_files = list(results_path.glob('*_timing.txt'))
    
    print(f"📊 Analyzing {len(result_files)} benchmark result files...")
    
    for result_file in result_files:
        benchmark_name = result_file.stem.replace('_results', '')
        
        try:
            with open(result_file, 'r') as f:
                content = f.read()
            
            # Extract key metrics (simplified parsing)
            lines = content.split('\n')
            metrics = {
                'lines_processed': len(lines),
                'file_size_bytes': result_file.stat().st_size,
                'contains_error': 'error' in content.lower() or 'failed' in content.lower(),
                'contains_success': 'completed' in content.lower() or 'success' in content.lower(),
            }
            
            # Look for timing information
            timing_file = results_path / f"{benchmark_name}_timing.txt"
            if timing_file.exists():
                with open(timing_file, 'r') as f:
                    timing_content = f.read().strip()
                    # Extract seconds from "Benchmark completed in Xs"
                    if 'completed in' in timing_content:
                        try:
                            seconds = timing_content.split('completed in ')[1].split('s')[0]
                            metrics['duration_seconds'] = float(seconds)
                        except (IndexError, ValueError):
                            metrics['duration_seconds'] = None
            
            analysis['benchmarks'][benchmark_name] = metrics
            
        except Exception as e:
            print(f"⚠️  Error analyzing {result_file}: {e}")
    
    # Generate summary
    total_benchmarks = len(analysis['benchmarks'])
    successful = sum(1 for b in analysis['benchmarks'].values() if not b['contains_error'])
    failed = total_benchmarks - successful
    
    analysis['summary'] = {
        'total_benchmarks': total_benchmarks,
        'successful': successful,
        'failed': failed,
        'success_rate': round((successful / total_benchmarks * 100), 2) if total_benchmarks > 0 else 0
    }
    
    return analysis

def display_analysis(analysis):
    """Display benchmark analysis in a nice format"""
    if not analysis:
        return
    
    print("\n" + "="*50)
    print("📈 BENCHMARK ANALYSIS REPORT")
    print("="*50)
    
    print(f"📁 Directory: {analysis['directory']}")
    print(f"🕐 Analyzed: {analysis['timestamp']}")
    
    summary = analysis['summary']
    print(f"\n📊 Summary:")
    print(f"   Total Benchmarks: {summary['total_benchmarks']}")
    print(f"   ✅ Successful: {summary['successful']}")
    print(f"   ❌ Failed: {summary['failed']}")
    print(f"   📈 Success Rate: {summary['success_rate']}%")
    
    print(f"\n🔍 Benchmark Details:")
    for name, metrics in analysis['benchmarks'].items():
        status = "✅" if not metrics['contains_error'] else "❌"
        duration = f" ({metrics['duration_seconds']}s)" if metrics.get('duration_seconds') else ""
        print(f"   {status} {name:<25} {duration}")
    
    # Performance insights
    print(f"\n💡 Performance Insights:")
    
    durations = [m['duration_seconds'] for m in analysis['benchmarks'].values() if m.get('duration_seconds')]
    if durations:
        fastest = min(durations)
        slowest = max(durations)
        avg_duration = sum(durations) / len(durations)
        
        print(f"   ⚡ Fastest benchmark: {fastest:.2f}s")
        print(f"   🐌 Slowest benchmark: {slowest:.2f}s")
        print(f"   📊 Average duration: {avg_duration:.2f}s")
    
    # Find largest result files (might indicate verbose output or errors)
    sizes = [(name, metrics['file_size_bytes']) for name, metrics in analysis['benchmarks'].items()]
    sizes.sort(key=lambda x: x[1], reverse=True)
    
    if sizes:
        print(f"   📄 Largest output: {sizes[0][0]} ({sizes[0][1]} bytes)")

def monitor_system(duration_seconds=60):
    """Monitor system performance for a given duration"""
    print(f"🖥️  Monitoring system performance for {duration_seconds} seconds...")
    
    start_time = time.time()
    samples = []
    
    try:
        while time.time() - start_time < duration_seconds:
            sample = get_system_info()
            samples.append(sample)
            
            # Display current info
            print(f"\r⏱️  {sample['timestamp'][:19]} | "
                  f"Load: {sample['load_average']} | "
                  f"Cores: {sample['cpu_cores']} | "
                  f"RAM: {sample['memory_gb']}GB", end='', flush=True)
            
            time.sleep(2)
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Monitoring stopped by user")
    
    print(f"\n\n📊 Collected {len(samples)} samples")
    return samples

def find_latest_results():
    """Find the most recent benchmark results directory"""
    results_base = Path('benchmark_results')
    if not results_base.exists():
        return None
    
    # Find directories with datetime format YYYYMMDD_HHMMSS
    result_dirs = [d for d in results_base.iterdir() if d.is_dir()]
    if not result_dirs:
        return None
    
    # Sort by modification time and return the most recent
    latest = max(result_dirs, key=lambda d: d.stat().st_mtime)
    return str(latest)

def main():
    parser = argparse.ArgumentParser(description='TrenchBot AI Performance Monitor')
    parser.add_argument('--analyze', type=str, help='Analyze benchmark results from directory')
    parser.add_argument('--monitor', type=int, help='Monitor system for N seconds', default=0)
    parser.add_argument('--latest', action='store_true', help='Analyze latest benchmark results')
    parser.add_argument('--run-quick', action='store_true', help='Run quick benchmark and analyze')
    
    args = parser.parse_args()
    
    print("🚀 TrenchBot AI Performance Monitor")
    print("=" * 40)
    
    # Display current system info
    system_info = get_system_info()
    print(f"🖥️  System: {system_info['cpu_cores']} cores, {system_info['memory_gb']}GB RAM")
    print(f"📊 Load: {system_info['load_average']}")
    
    if args.run_quick:
        print("\n🏃 Running quick benchmark...")
        try:
            result = subprocess.run(['./scripts/quick_benchmark.sh'], 
                                  capture_output=True, text=True, check=True)
            print(result.stdout)
            if result.stderr:
                print("Errors/Warnings:")
                print(result.stderr)
        except subprocess.CalledProcessError as e:
            print(f"❌ Quick benchmark failed: {e}")
            return 1
    
    if args.analyze:
        analysis = analyze_benchmark_results(args.analyze)
        display_analysis(analysis)
    
    elif args.latest:
        latest_dir = find_latest_results()
        if latest_dir:
            print(f"\n🔍 Found latest results: {latest_dir}")
            analysis = analyze_benchmark_results(latest_dir)
            display_analysis(analysis)
        else:
            print("❌ No benchmark results found. Run benchmarks first!")
            return 1
    
    if args.monitor > 0:
        samples = monitor_system(args.monitor)
        print(f"✅ Monitoring completed with {len(samples)} samples")
    
    if not any([args.analyze, args.latest, args.monitor, args.run_quick]):
        print("\n💡 Usage examples:")
        print("  python3 scripts/performance_monitor.py --latest")
        print("  python3 scripts/performance_monitor.py --analyze benchmark_results/20231201_143022")
        print("  python3 scripts/performance_monitor.py --monitor 30")
        print("  python3 scripts/performance_monitor.py --run-quick")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())