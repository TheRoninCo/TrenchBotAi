#!/usr/bin/env python3
"""
Claude Doctor - TrenchBot AI System Diagnostics
Comprehensive health check and optimization recommendations
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime
import platform

class TrenchBotDoctor:
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.recommendations = []
        self.system_info = {}
        
    def run_diagnosis(self):
        """Run comprehensive system diagnosis"""
        print("🩺 Claude Doctor - TrenchBot AI System Diagnostics")
        print("=" * 55)
        print(f"🕐 Diagnosis started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Collect system information
        self._collect_system_info()
        
        # Run diagnostic checks
        self._check_project_structure()
        self._check_dependencies()
        self._check_rust_environment()
        self._check_gpu_support()
        self._check_performance_files()
        self._check_configuration()
        self._check_memory_usage()
        self._check_disk_space()
        
        # Generate report
        self._generate_report()
        
    def _collect_system_info(self):
        """Collect system information"""
        print("🖥️  Collecting system information...")
        
        try:
            self.system_info = {
                'platform': platform.system(),
                'architecture': platform.machine(),
                'python_version': platform.python_version(),
                'timestamp': datetime.now().isoformat()
            }
            
            # CPU info
            try:
                if platform.system() == 'Darwin':  # macOS
                    cpu_info = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
                    cpu_cores = subprocess.check_output(['sysctl', '-n', 'hw.ncpu'], text=True).strip()
                    memory = subprocess.check_output(['sysctl', '-n', 'hw.memsize'], text=True).strip()
                    
                    self.system_info.update({
                        'cpu': cpu_info,
                        'cpu_cores': int(cpu_cores),
                        'memory_gb': round(int(memory) / (1024**3), 2)
                    })
                else:
                    self.system_info.update({
                        'cpu': 'Unknown',
                        'cpu_cores': os.cpu_count() or 'Unknown',
                        'memory_gb': 'Unknown'
                    })
            except:
                self.warnings.append("Could not gather detailed CPU/memory information")
                
            print(f"   ✅ System: {self.system_info['platform']} {self.system_info['architecture']}")
            print(f"   ✅ CPU: {self.system_info.get('cpu', 'Unknown')}")
            print(f"   ✅ Cores: {self.system_info.get('cpu_cores', 'Unknown')}")
            print(f"   ✅ Memory: {self.system_info.get('memory_gb', 'Unknown')}GB")
            
        except Exception as e:
            self.issues.append(f"Failed to collect system info: {e}")
    
    def _check_project_structure(self):
        """Check TrenchBot project structure"""
        print("\n📁 Checking project structure...")
        
        required_files = [
            'Cargo.toml',
            'src/main.rs',
            'src/lib.rs',
            'src/analytics/mod.rs',
            'src/gpu_ai/mod.rs',
            'src/strategies/mod.rs'
        ]
        
        optional_files = [
            'src/gpu_ai/quantum_prediction.rs',
            'src/gpu_ai/transformer_engine.rs',
            'src/gpu_ai/graph_neural_engine.rs',
            'src/gpu_ai/monte_carlo_engine.rs',
            'src/gpu_ai/neural_architecture_search.rs',
            'src/strategies/competitive_trading.rs'
        ]
        
        for file_path in required_files:
            if Path(file_path).exists():
                print(f"   ✅ {file_path}")
            else:
                self.issues.append(f"Missing required file: {file_path}")
                print(f"   ❌ {file_path}")
        
        print(f"\n   📊 Advanced AI Components:")
        for file_path in optional_files:
            if Path(file_path).exists():
                print(f"   ✅ {file_path}")
            else:
                print(f"   ⚠️  {file_path} (optional)")
    
    def _check_dependencies(self):
        """Check Rust dependencies"""
        print("\n📦 Checking dependencies...")
        
        try:
            with open('Cargo.toml', 'r') as f:
                cargo_content = f.read()
            
            # Check for key dependencies
            key_deps = {
                'tokio': 'Async runtime',
                'serde': 'Serialization',
                'anyhow': 'Error handling', 
                'chrono': 'Date/time handling',
                'ndarray': 'Numerical computing',
                'rayon': 'Parallel processing',
                'tch': 'PyTorch bindings (GPU)',
                'criterion': 'Benchmarking'
            }
            
            for dep, description in key_deps.items():
                if dep in cargo_content:
                    print(f"   ✅ {dep}: {description}")
                else:
                    if dep == 'tch':
                        self.warnings.append(f"Optional dependency missing: {dep} ({description})")
                        print(f"   ⚠️  {dep}: {description} (optional)")
                    else:
                        self.issues.append(f"Missing dependency: {dep} ({description})")
                        print(f"   ❌ {dep}: {description}")
            
            # Check features
            if 'gpu = ["tch"]' in cargo_content:
                print(f"   ✅ GPU feature configured")
            else:
                self.recommendations.append("Consider adding GPU feature for acceleration")
                print(f"   ⚠️  GPU feature not configured")
                
        except Exception as e:
            self.issues.append(f"Could not read Cargo.toml: {e}")
    
    def _check_rust_environment(self):
        """Check Rust environment"""
        print("\n🦀 Checking Rust environment...")
        
        try:
            # Check Rust version
            rust_version = subprocess.check_output(['rustc', '--version'], text=True).strip()
            print(f"   ✅ Rust: {rust_version}")
            
            # Check Cargo version
            cargo_version = subprocess.check_output(['cargo', '--version'], text=True).strip()
            print(f"   ✅ Cargo: {cargo_version}")
            
            # Try compilation check
            print("   🔨 Testing compilation...")
            result = subprocess.run(['cargo', 'check', '--quiet'], 
                                  capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                print("   ✅ Project compiles successfully")
            else:
                self.issues.append("Project has compilation errors")
                print("   ❌ Compilation errors detected")
                if result.stderr:
                    print(f"      Error details: {result.stderr[:200]}...")
                    
        except subprocess.TimeoutExpired:
            self.warnings.append("Compilation check timed out (project may be large)")
            print("   ⚠️  Compilation check timed out")
        except FileNotFoundError:
            self.issues.append("Rust/Cargo not found in PATH")
            print("   ❌ Rust/Cargo not installed or not in PATH")
        except Exception as e:
            self.issues.append(f"Rust environment check failed: {e}")
    
    def _check_gpu_support(self):
        """Check GPU support"""
        print("\n🎮 Checking GPU support...")
        
        # Check for NVIDIA GPU
        try:
            nvidia_output = subprocess.check_output(['nvidia-smi'], text=True)
            print("   ✅ NVIDIA GPU detected")
            
            # Extract GPU info
            lines = nvidia_output.split('\n')
            for line in lines:
                if 'GeForce' in line or 'RTX' in line or 'GTX' in line or 'Tesla' in line:
                    gpu_name = line.split('|')[1].strip()
                    print(f"   ✅ GPU: {gpu_name}")
                    break
                    
        except (FileNotFoundError, subprocess.CalledProcessError):
            print("   ⚠️  NVIDIA GPU not detected")
            self.recommendations.append("Consider GPU acceleration for 10-100x speedup on AI workloads")
        
        # Check for Metal (Apple Silicon)
        if platform.system() == 'Darwin':
            try:
                # Check for Apple Silicon
                arch_output = subprocess.check_output(['uname', '-m'], text=True).strip()
                if 'arm64' in arch_output:
                    print("   ✅ Apple Silicon detected (Metal GPU available)")
                else:
                    print("   ⚠️  Intel Mac (limited GPU acceleration)")
            except:
                pass
    
    def _check_performance_files(self):
        """Check performance and benchmark files"""
        print("\n⚡ Checking performance infrastructure...")
        
        perf_files = {
            'benches/ai_performance_benchmarks.rs': 'Comprehensive benchmarks',
            'scripts/run_benchmarks.sh': 'Benchmark runner',
            'scripts/quick_benchmark.sh': 'Quick test runner', 
            'scripts/performance_monitor.py': 'Performance monitor',
            'test_basic_functionality.py': 'Functionality tests'
        }
        
        for file_path, description in perf_files.items():
            if Path(file_path).exists():
                print(f"   ✅ {description}")
            else:
                self.warnings.append(f"Performance file missing: {file_path}")
                print(f"   ⚠️  {description}")
        
        # Check if benchmark results exist
        if Path('benchmark_results').exists():
            result_dirs = list(Path('benchmark_results').iterdir())
            print(f"   ✅ {len(result_dirs)} benchmark result directories found")
        else:
            print("   ⚠️  No benchmark results found")
            self.recommendations.append("Run benchmarks to establish performance baseline")
    
    def _check_configuration(self):
        """Check configuration files"""
        print("\n⚙️  Checking configuration...")
        
        config_files = [
            'configs/global.toml',
            'configs/chains/solana.toml',
            'configs/logging.toml',
            'config/art_config.toml'
        ]
        
        for config_file in config_files:
            if Path(config_file).exists():
                print(f"   ✅ {config_file}")
            else:
                print(f"   ⚠️  {config_file} (optional)")
        
        # Check environment variables
        important_env_vars = ['RUST_LOG', 'CUDA_VISIBLE_DEVICES']
        for env_var in important_env_vars:
            value = os.environ.get(env_var)
            if value:
                print(f"   ✅ {env_var}={value}")
            else:
                print(f"   ⚠️  {env_var} not set")
    
    def _check_memory_usage(self):
        """Check memory usage"""
        print("\n💾 Checking memory usage...")
        
        try:
            # Get current process memory
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            rss_mb = memory_info.rss / (1024 * 1024)
            vms_mb = memory_info.vms / (1024 * 1024)
            
            print(f"   ✅ Current RSS memory: {rss_mb:.1f}MB")
            print(f"   ✅ Current VMS memory: {vms_mb:.1f}MB")
            
            if rss_mb > 1000:
                self.warnings.append(f"High memory usage: {rss_mb:.1f}MB")
            
        except ImportError:
            print("   ⚠️  psutil not available for memory monitoring")
            self.recommendations.append("Install psutil for memory monitoring: pip install psutil")
        except Exception as e:
            print(f"   ⚠️  Could not check memory usage: {e}")
    
    def _check_disk_space(self):
        """Check disk space"""
        print("\n💽 Checking disk space...")
        
        try:
            # Check current directory disk usage
            if platform.system() == 'Darwin' or platform.system() == 'Linux':
                result = subprocess.check_output(['df', '-h', '.'], text=True)
                lines = result.strip().split('\n')
                if len(lines) > 1:
                    fields = lines[1].split()
                    if len(fields) >= 4:
                        total = fields[1]
                        used = fields[2]
                        available = fields[3]
                        usage = fields[4]
                        
                        print(f"   ✅ Disk usage: {used}/{total} ({usage})")
                        print(f"   ✅ Available: {available}")
                        
                        # Warn if usage is high
                        usage_pct = int(usage.rstrip('%'))
                        if usage_pct > 90:
                            self.issues.append(f"Disk space critically low: {usage}")
                        elif usage_pct > 80:
                            self.warnings.append(f"Disk space getting low: {usage}")
                            
        except Exception as e:
            print(f"   ⚠️  Could not check disk space: {e}")
    
    def _generate_report(self):
        """Generate diagnostic report"""
        print("\n" + "=" * 55)
        print("📋 DIAGNOSTIC REPORT")
        print("=" * 55)
        
        # Summary
        print(f"🕐 Diagnosis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🖥️  System: {self.system_info.get('platform', 'Unknown')} {self.system_info.get('architecture', 'Unknown')}")
        print()
        
        # Issues
        if self.issues:
            print(f"❌ CRITICAL ISSUES ({len(self.issues)}):")
            for i, issue in enumerate(self.issues, 1):
                print(f"   {i}. {issue}")
            print()
        else:
            print("✅ NO CRITICAL ISSUES FOUND")
            print()
        
        # Warnings  
        if self.warnings:
            print(f"⚠️  WARNINGS ({len(self.warnings)}):")
            for i, warning in enumerate(self.warnings, 1):
                print(f"   {i}. {warning}")
            print()
        
        # Recommendations
        if self.recommendations:
            print(f"💡 RECOMMENDATIONS ({len(self.recommendations)}):")
            for i, rec in enumerate(self.recommendations, 1):
                print(f"   {i}. {rec}")
            print()
        
        # Health score
        health_score = 100
        health_score -= len(self.issues) * 20  # Critical issues
        health_score -= len(self.warnings) * 5   # Warnings
        health_score = max(0, health_score)
        
        print(f"🏥 SYSTEM HEALTH SCORE: {health_score}/100")
        
        if health_score >= 90:
            print("   🎉 Excellent! System is in great shape.")
        elif health_score >= 70:
            print("   ✅ Good! Minor issues to address.")
        elif health_score >= 50:
            print("   ⚠️  Fair. Several issues need attention.")
        else:
            print("   ❌ Poor. Critical issues require immediate attention.")
        
        # Quick fixes
        print("\n💊 QUICK FIXES:")
        if not self.issues and not self.warnings:
            print("   🎉 No fixes needed! System is healthy.")
        else:
            print("   1. Run: cargo build --release")
            print("   2. Run: ./scripts/quick_benchmark.sh")  
            print("   3. Check logs for detailed error messages")
            print("   4. Consider GPU setup for performance boost")
        
        # Save report
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'system_info': self.system_info,
            'health_score': health_score,
            'issues': self.issues,
            'warnings': self.warnings,
            'recommendations': self.recommendations
        }
        
        report_file = f"diagnostic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n💾 Full report saved to: {report_file}")
        print()
        
        return health_score

def main():
    doctor = TrenchBotDoctor()
    health_score = doctor.run_diagnosis()
    
    # Exit code based on health
    if health_score >= 70:
        return 0  # Success
    elif health_score >= 50:
        return 1  # Warning
    else:
        return 2  # Critical

if __name__ == '__main__':
    sys.exit(main())