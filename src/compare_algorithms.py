import sys
import json
import time
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def run_algorithm(algorithm_type, config_path):
    """Run either original or improved algorithm"""
    
    if algorithm_type == "original":
        cmd = ["python", "main.py", config_path]
        results_file = "./results/test.csv"
    elif algorithm_type == "improved":
        cmd = ["python", "main_improved.py", config_path]
        results_file = "./results/test_improved.csv"
    else:
        raise ValueError("algorithm_type must be 'original' or 'improved'")
    
    print(f"\n{'='*60}")
    print(f"Running {algorithm_type.upper()} algorithm...")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        if result.returncode != 0:
            print(f"Error running {algorithm_type} algorithm:")
            print(result.stderr)
            return None
        
        runtime = time.time() - start_time
        print(f"\n{algorithm_type.capitalize()} algorithm completed in {runtime:.2f} seconds")
        
        # Load results
        try:
            results_df = pd.read_csv(results_file)
            return {
                'algorithm': algorithm_type,
                'runtime': runtime,
                'results': results_df,
                'final_estimate': float(results_df.iloc[-1]['probability_estimate']),
                'final_error': float(results_df.iloc[-1]['error']),
                'episodes': results_df.iloc[-1]['episodes']
            }
        except Exception as e:
            print(f"Error loading results for {algorithm_type}: {e}")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"{algorithm_type.capitalize()} algorithm timed out after 1 hour")
        return None
    except Exception as e:
        print(f"Error running {algorithm_type} algorithm: {e}")
        return None

def compare_results(original_result, improved_result, true_prob=None):
    """Compare results from both algorithms"""
    
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")
    
    if original_result is None or improved_result is None:
        print("Cannot compare - one or both algorithms failed")
        return
    
    # Extract results
    orig_estimate = original_result['final_estimate']
    orig_error = original_result['final_error']
    orig_episodes = original_result['episodes']
    orig_runtime = original_result['runtime']
    
    impr_estimate = improved_result['final_estimate']
    impr_error = improved_result['final_error']
    impr_episodes = improved_result['episodes']
    impr_runtime = improved_result['runtime']
    
    print(f"\nFinal Estimates:")
    print(f"  Original:  {orig_estimate:.6E} ± {orig_error:.6E} ({orig_episodes:,} episodes)")
    print(f"  Improved:  {impr_estimate:.6E} ± {impr_error:.6E} ({impr_episodes:,} episodes)")
    
    if true_prob:
        orig_bias = abs(orig_estimate - true_prob) / true_prob * 100
        impr_bias = abs(impr_estimate - true_prob) / true_prob * 100
        
        print(f"\nBias from true probability ({true_prob:.6E}):")
        print(f"  Original:  {orig_bias:.2f}%")
        print(f"  Improved:  {impr_bias:.2f}%")
        
        orig_covers = (orig_estimate - 1.96*orig_error <= true_prob <= orig_estimate + 1.96*orig_error)
        impr_covers = (impr_estimate - 1.96*impr_error <= true_prob <= impr_estimate + 1.96*impr_error)
        
        print(f"\nConfidence interval coverage:")
        print(f"  Original:  {'✓' if orig_covers else '✗'}")
        print(f"  Improved:  {'✓' if impr_covers else '✗'}")
    
    print(f"\nRelative Error:")
    orig_rel_error = orig_error / (orig_estimate + 1e-10)
    impr_rel_error = impr_error / (impr_estimate + 1e-10)
    print(f"  Original:  {orig_rel_error:.4f}")
    print(f"  Improved:  {impr_rel_error:.4f}")
    
    print(f"\nComputational Efficiency:")
    print(f"  Original:  {orig_runtime:.1f} seconds")
    print(f"  Improved:  {impr_runtime:.1f} seconds")
    print(f"  Speedup:   {orig_runtime/impr_runtime:.2f}x")
    
    # Improvement summary
    estimate_improvement = (impr_estimate - orig_estimate) / orig_estimate * 100
    error_improvement = (orig_error - impr_error) / orig_error * 100
    
    print(f"\nImprovement Summary:")
    print(f"  Estimate change: {estimate_improvement:+.2f}%")
    print(f"  Error reduction: {error_improvement:+.2f}%")
    
    # Plot convergence comparison
    plot_convergence_comparison(original_result['results'], improved_result['results'])

def plot_convergence_comparison(orig_df, impr_df):
    """Plot convergence comparison"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Probability estimates
    axes[0,0].plot(orig_df['episodes'], orig_df['probability_estimate'], 
                   label='Original', alpha=0.8)
    axes[0,0].plot(impr_df['episodes'], impr_df['probability_estimate'], 
                   label='Improved', alpha=0.8)
    axes[0,0].set_xlabel('Episodes')
    axes[0,0].set_ylabel('Probability Estimate')
    axes[0,0].set_title('Convergence of Probability Estimates')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].set_yscale('log')
    
    # Standard error
    axes[0,1].plot(orig_df['episodes'], orig_df['error'], 
                   label='Original', alpha=0.8)
    axes[0,1].plot(impr_df['episodes'], impr_df['error'], 
                   label='Improved', alpha=0.8)
    axes[0,1].set_xlabel('Episodes')
    axes[0,1].set_ylabel('Standard Error')
    axes[0,1].set_title('Convergence of Standard Error')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].set_yscale('log')
    
    # Relative error (if available)
    if 'relative_error' in orig_df.columns and 'relative_error' in impr_df.columns:
        axes[1,0].plot(orig_df['episodes'], orig_df['relative_error'], 
                       label='Original', alpha=0.8)
        axes[1,0].plot(impr_df['episodes'], impr_df['relative_error'], 
                       label='Improved', alpha=0.8)
        axes[1,0].set_xlabel('Episodes')
        axes[1,0].set_ylabel('Relative Error')
        axes[1,0].set_title('Convergence of Relative Error')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
    
    # Variance
    axes[1,1].plot(orig_df['episodes'], orig_df['variance'], 
                   label='Original', alpha=0.8)
    axes[1,1].plot(impr_df['episodes'], impr_df['variance'], 
                   label='Improved', alpha=0.8)
    axes[1,1].set_xlabel('Episodes')
    axes[1,1].set_ylabel('Variance')
    axes[1,1].set_title('Convergence of Variance')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('./results/algorithm_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nComparison plot saved to './results/algorithm_comparison.png'")

def main():
    if len(sys.argv) < 2:
        print("Usage: python compare_algorithms.py <json_config> [--true-prob <value>] [--skip-original] [--skip-improved]")
        sys.exit(1)
    
    config_path = sys.argv[1]
    true_prob = None
    skip_original = False
    skip_improved = False
    
    # Parse arguments
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--true-prob':
            true_prob = float(sys.argv[i+1])
            i += 2
        elif sys.argv[i] == '--skip-original':
            skip_original = True
            i += 1
        elif sys.argv[i] == '--skip-improved':
            skip_improved = True
            i += 1
        else:
            i += 1
    
    # Load config for display
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"Comparing algorithms on model: {config['model_path']}")
    print(f"Target: {config['target_variable']} = {config['target_value']}")
    if true_prob:
        print(f"True probability: {true_prob:.6E}")
    
    # Run algorithms
    original_result = None if skip_original else run_algorithm("original", config_path)
    improved_result = None if skip_improved else run_algorithm("improved", config_path)
    
    # Compare results
    compare_results(original_result, improved_result, true_prob)

if __name__ == "__main__":
    main()