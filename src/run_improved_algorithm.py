#!/usr/bin/env python3
"""
Main script to run the improved REINFORCE algorithm for WSSA
Addresses underestimation issues with comprehensive improvements
"""

import sys
import os
import subprocess
import json

def check_dependencies():
    """Check and install required dependencies"""
    required_packages = [
        'numpy',
        'matplotlib',
        'seaborn', 
        'pandas',
        'tqdm',
        'pyyaml'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Installing missing packages: {missing_packages}")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)

def main():
    print("="*80)
    print("IMPROVED REINFORCE FOR WEIGHTED SSA")
    print("Addressing Underestimation Issues")
    print("="*80)
    
    print("\nKey improvements implemented:")
    print("✓ Ensemble of 3 policies for better state space coverage")
    print("✓ Defensive importance sampling with mixture policy")
    print("✓ Temperature-based exploration with adaptive annealing")
    print("✓ UCB-style exploration bonuses")
    print("✓ Path diversity rewards")
    print("✓ State-dependent baselines for variance reduction")
    print("✓ Adaptive learning rates per state")
    print("✓ Reduced biasing bounds (±1.5 instead of ±2.0)")
    print("✓ Enhanced state representation")
    print("✓ Tighter convergence criteria")
    
    # Check dependencies
    print("\nChecking dependencies...")
    check_dependencies()
    
    # Parse arguments
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python run_improved_algorithm.py <config.json> [options]")
        print("\nOptions:")
        print("  --compare          Run both original and improved algorithms")
        print("  --analyze          Analyze existing policy for underestimation")
        print("  --true-prob <val>  Specify true probability for comparison")
        print("\nExamples:")
        print("  python run_improved_algorithm.py config_single_species.json")
        print("  python run_improved_algorithm.py config.json --compare --true-prob 2.41E-7")
        print("  python run_improved_algorithm.py config.json --analyze reinforce_policy.yaml")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    # Parse options
    compare = '--compare' in sys.argv
    analyze = '--analyze' in sys.argv
    true_prob = None
    
    if '--true-prob' in sys.argv:
        idx = sys.argv.index('--true-prob')
        if idx + 1 < len(sys.argv):
            true_prob = float(sys.argv[idx + 1])
    
    # Validate config file
    if not os.path.exists(config_path):
        print(f"Error: Config file '{config_path}' not found")
        sys.exit(1)
    
    # Load and display config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"\nConfiguration:")
    print(f"  Model: {config['model_path']}")
    print(f"  Target: {config['target_variable']} = {config['target_value']}")
    print(f"  Time limit: {config['max_time']}")
    if true_prob:
        print(f"  True probability: {true_prob:.6E}")
    
    # Ensure results directory exists
    os.makedirs('./results', exist_ok=True)
    
    if analyze:
        print("\nRunning analysis...")
        policy_files = [arg for arg in sys.argv if arg.endswith('.yaml')]
        if not policy_files:
            policy_files = ['reinforce_policy.yaml']
        
        cmd = ['python', 'analyze_underestimation.py', config_path] + policy_files
        if true_prob:
            cmd.extend(['--true-prob', str(true_prob)])
        
        subprocess.run(cmd)
        
    elif compare:
        print("\nRunning comparison...")
        cmd = ['python', 'compare_algorithms.py', config_path]
        if true_prob:
            cmd.extend(['--true-prob', str(true_prob)])
        
        subprocess.run(cmd)
        
    else:
        print("\nRunning improved algorithm...")
        subprocess.run(['python', 'main_improved.py', config_path])
    
    print("\n" + "="*80)
    print("EXECUTION COMPLETE")
    print("="*80)
    print("\nResults saved in './results/' directory:")
    print("  - test_improved.txt: Summary results")
    print("  - test_improved.csv: Detailed convergence data")
    print("  - training_progress_improved.png: Training visualization")
    print("  - reinforce_policy_*.yaml: Learned policies")
    
    if compare:
        print("  - algorithm_comparison.png: Algorithm comparison plots")
    
    if analyze:
        print("  - gamma_analysis.png: Policy analysis plots")
    
    print(f"\nTo analyze your results further:")
    print(f"  python analyze_underestimation.py {config_path} reinforce_policy_*.yaml --true-prob {true_prob or 'X.XXE-X'}")

if __name__ == "__main__":
    main()