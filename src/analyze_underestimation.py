import numpy as np
import yaml
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
from prism_parser import parser
from suppress import suppress_c_output
import json

def analyze_policy_coverage(policy_file, model_path, target_var):
    """Analyze state coverage and biasing aggressiveness"""
    
    # Load policy
    with open(policy_file, 'r') as f:
        policy = yaml.safe_load(f)
    
    # Parse model to get number of reactions
    with suppress_c_output():
        model = parser(model_path)
    n_reactions = len(model.get_reactions_vector())
    
    print(f"\nAnalyzing policy from {policy_file}")
    print(f"Number of states visited: {len(policy)}")
    
    # Analyze gamma values
    all_gammas = []
    state_gamma_ranges = []
    
    for state_str, log_gammas in policy.items():
        gammas = np.exp(log_gammas)
        all_gammas.extend(gammas)
        state_gamma_ranges.append(np.max(gammas) / (np.min(gammas) + 1e-10))
    
    all_gammas = np.array(all_gammas)
    
    print(f"\nGamma statistics:")
    print(f"  Mean: {np.mean(all_gammas):.3f}")
    print(f"  Median: {np.median(all_gammas):.3f}")
    print(f"  Min: {np.min(all_gammas):.3f}")
    print(f"  Max: {np.max(all_gammas):.3f}")
    print(f"  Std: {np.std(all_gammas):.3f}")
    
    # Analyze biasing aggressiveness
    aggressive_states = sum(1 for r in state_gamma_ranges if r > 10)
    very_aggressive_states = sum(1 for r in state_gamma_ranges if r > 100)
    
    print(f"\nBiasing aggressiveness:")
    print(f"  States with gamma ratio > 10: {aggressive_states} ({100*aggressive_states/len(policy):.1f}%)")
    print(f"  States with gamma ratio > 100: {very_aggressive_states} ({100*very_aggressive_states/len(policy):.1f}%)")
    
    # Plot gamma distribution
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.hist(all_gammas, bins=50, alpha=0.7)
    plt.xlabel('Gamma values')
    plt.ylabel('Count')
    plt.title('Distribution of Gamma Values')
    plt.yscale('log')
    
    plt.subplot(1, 3, 2)
    plt.hist(np.log(all_gammas), bins=50, alpha=0.7)
    plt.xlabel('Log Gamma values')
    plt.ylabel('Count')
    plt.title('Distribution of Log Gamma Values')
    
    plt.subplot(1, 3, 3)
    plt.hist(np.log10(state_gamma_ranges), bins=50, alpha=0.7)
    plt.xlabel('Log10(Max/Min Gamma Ratio)')
    plt.ylabel('Number of States')
    plt.title('Biasing Aggressiveness per State')
    
    plt.tight_layout()
    plt.savefig('./results/gamma_analysis.png')
    plt.close()
    
    return policy, all_gammas, state_gamma_ranges

def analyze_trajectory_weights(weights_file):
    """Analyze the distribution of importance weights"""
    
    # Load weights from evaluation
    weights = np.load(weights_file) if weights_file.endswith('.npy') else []
    
    if len(weights) == 0:
        print("No weights file found. Run evaluation first.")
        return
    
    successful_weights = [w for w in weights if w > 0]
    
    print(f"\nWeight analysis:")
    print(f"  Total trajectories: {len(weights)}")
    print(f"  Successful trajectories: {len(successful_weights)}")
    print(f"  Success rate: {len(successful_weights)/len(weights):.4f}")
    
    if successful_weights:
        print(f"\nSuccessful trajectory weights:")
        print(f"  Mean: {np.mean(successful_weights):.3E}")
        print(f"  Median: {np.median(successful_weights):.3E}")
        print(f"  Min: {np.min(successful_weights):.3E}")
        print(f"  Max: {np.max(successful_weights):.3E}")
        print(f"  Std: {np.std(successful_weights):.3E}")
        
        # Analyze weight distribution
        log_weights = np.log10(successful_weights)
        
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.hist(log_weights, bins=50, alpha=0.7, density=True)
        plt.xlabel('Log10(Weight)')
        plt.ylabel('Density')
        plt.title('Distribution of Log Weights')
        
        plt.subplot(1, 2, 2)
        plt.boxplot(successful_weights)
        plt.ylabel('Weight')
        plt.title('Weight Distribution')
        plt.yscale('log')
        
        plt.tight_layout()
        plt.savefig('./results/weight_analysis.png')
        plt.close()

def diagnose_underestimation(json_path, policy_files, true_prob=None):
    """Comprehensive diagnosis of underestimation issues"""
    
    # Load configuration
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    model_path = json_data['model_path']
    target_var = json_data['target_variable']
    
    print("="*60)
    print("UNDERESTIMATION DIAGNOSIS")
    print("="*60)
    
    # Analyze each policy
    all_policies = []
    for policy_file in policy_files:
        policy, gammas, ranges = analyze_policy_coverage(policy_file, model_path, target_var)
        all_policies.append(policy)
    
    # Compare state coverage between policies
    if len(all_policies) > 1:
        print(f"\nState coverage comparison:")
        all_states = set()
        for policy in all_policies:
            all_states.update(policy.keys())
        
        print(f"  Total unique states across all policies: {len(all_states)}")
        
        coverage_matrix = []
        for i, policy in enumerate(all_policies):
            coverage = [1 if state in policy else 0 for state in all_states]
            coverage_matrix.append(coverage)
            print(f"  Policy {i}: {sum(coverage)}/{len(all_states)} states ({100*sum(coverage)/len(all_states):.1f}%)")
        
        # Calculate overlap
        overlap = sum(all(coverage_matrix[j][i] for j in range(len(all_policies))) 
                     for i in range(len(all_states)))
        print(f"  States covered by all policies: {overlap} ({100*overlap/len(all_states):.1f}%)")
    
    # Diagnosis summary
    print("\n" + "="*60)
    print("DIAGNOSIS SUMMARY:")
    print("="*60)
    
    print("\nPotential causes of underestimation:")
    print("1. State Space Coverage:")
    print("   - Limited state exploration during training")
    print("   - Important rare states might be missed")
    
    print("\n2. Biasing Aggressiveness:")
    print("   - Overly aggressive biasing leads to high-variance weights")
    print("   - Some successful paths might have extremely low probability")
    
    print("\n3. Importance Sampling Issues:")
    print("   - High variance in importance weights")
    print("   - Effective sample size might be very small")
    
    print("\nRecommended solutions implemented:")
    print("✓ Defensive importance sampling with mixture policy")
    print("✓ Temperature-based exploration with slow annealing") 
    print("✓ UCB-style exploration bonuses")
    print("✓ Ensemble of policies for better coverage")
    print("✓ Path diversity rewards")
    print("✓ Reduced biasing bounds (±1.5 instead of ±2.0)")
    print("✓ State-dependent baselines for variance reduction")
    print("✓ Adaptive learning rates per state")
    
    if true_prob:
        print(f"\nTrue probability: {true_prob:.3E}")
        print("Run the improved algorithm to see if it achieves better estimates.")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python analyze_underestimation.py <json_config> <policy_file1> [policy_file2] ... [--true-prob <value>]")
        sys.exit(1)
    
    json_path = sys.argv[1]
    policy_files = []
    true_prob = None
    
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--true-prob':
            true_prob = float(sys.argv[i+1])
            i += 2
        else:
            policy_files.append(sys.argv[i])
            i += 1
    
    diagnose_underestimation(json_path, policy_files, true_prob)