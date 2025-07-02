# Improved REINFORCE for Weighted SSA - Addressing Underestimation

## Problem Statement

The original REINFORCE implementation was underestimating the true probability (1.37E-7 vs true 2.41E-7), indicating that the learned biasing policy was too aggressive and missing important probability mass in the state space.

## Root Causes of Underestimation

1. **Insufficient Exploration**: Policy converging too quickly to suboptimal solutions
2. **Aggressive Biasing**: Extreme gamma values causing high variance in importance weights  
3. **Limited State Coverage**: Missing rare but important states during training
4. **Poor Variance Control**: High variance in importance sampling estimates

## Comprehensive Solution

### 1. Defensive Importance Sampling
- **Mixture Policy**: Combines learned policy with uniform policy (10% weight)
- **Prevents Over-Aggressive Biasing**: Ensures some exploration even with converged policy
- **Implementation**: `gamma_values = (1-0.1)*learned + 0.1*uniform`

### 2. Enhanced Exploration Strategy
- **Temperature-Based Exploration**: 
  - Initial temperature: 1.0
  - Decay rate: 0.995 per batch
  - Minimum temperature: 0.1
- **UCB Exploration Bonuses**: Upper confidence bound style bonuses based on visit counts
- **Adaptive Noise**: Exploration noise decreases with state visits

### 3. Ensemble of Policies
- **Multiple Policies**: Train 3 different policies simultaneously
- **Better Coverage**: Each policy explores different regions of state space
- **Robust Estimates**: Combine estimates from all policies

### 4. Improved Reward Structure
- **Path Diversity Bonus**: Rewards trajectories using diverse reactions
- **Weight Penalty**: Re-enabled to discourage extremely heavy trajectories
- **State-Dependent Baselines**: Variance reduction using learned baselines per state

### 5. Advanced Variance Reduction
- **State-Dependent Learning Rates**: Adapt learning rate based on state visit frequency
- **Gradient Variance Tracking**: Monitor and use gradient variance for exploration
- **Entropy Regularization**: Encourage policy diversity (increased from 0.01 to 0.02)

### 6. Conservative Biasing Bounds
- **Reduced Range**: Log-gamma clipped to [-1.5, 1.5] instead of [-2.0, 2.0]
- **Gamma Range**: [0.22, 4.48] instead of [0.135, 7.39]
- **Less Aggressive**: Prevents extreme biasing that causes underestimation

### 7. Enhanced State Representation
- **3D State Space**: (population, time_pressure, distance_to_target)
- **Better Discrimination**: More informative state features
- **Distance Binning**: Additional state dimension for target proximity

### 8. Adaptive Stopping Criteria
- **Training**: Policy convergence threshold 0.0005 (tighter than 0.001)
- **Evaluation**: Relative error threshold 0.03 (tighter than 0.05)
- **Minimum Episodes**: 5000 before checking stopping (increased from 1000)

## Key Algorithm Changes

### Training Loop
```python
# Ensemble of 3 policies with temperature-based exploration
for policy_idx in range(n_policies):
    # UCB exploration bonus
    ucb_bonus = UCB_CONSTANT * sqrt(log(episode) / visit_count) * uncertainty
    
    # Temperature scaling
    log_gammas = (log_gammas + noise + ucb_bonus) / temperature
    
    # Defensive mixture
    gamma_values = (1-0.1) * learned_gamma + 0.1 * uniform_gamma
```

### Reward Function
```python
reward = progress_reward + weight_penalty + success_bonus + diversity_bonus
```

### State-Dependent Baseline
```python
advantage = reward - baseline[state]
baseline[state] = (1-α) * old_baseline + α * reward
```

## Usage

### Quick Start
```bash
# Run improved algorithm
python run_improved_algorithm.py config_single_species.json

# Compare with original
python run_improved_algorithm.py config.json --compare --true-prob 2.41E-7

# Analyze existing policy
python analyze_underestimation.py config.json policy.yaml --true-prob 2.41E-7
```

### Expected Improvements
1. **Higher Estimates**: Closer to true probability
2. **Lower Variance**: More stable convergence  
3. **Better Coverage**: Confidence intervals containing true value
4. **Robust Performance**: Less sensitive to initialization

## Files Created

- `reinforce_improved.py`: Core improved algorithm
- `main_improved.py`: Main execution script with ensemble evaluation
- `compare_algorithms.py`: Side-by-side comparison tool
- `analyze_underestimation.py`: Diagnostic analysis tool
- `run_improved_algorithm.py`: Unified entry point script

## Technical Details

### Computational Complexity
- **Training**: ~3x slower due to ensemble and enhanced exploration
- **Evaluation**: Similar speed with adaptive stopping
- **Memory**: ~3x memory for multiple policies

### Hyperparameters
- Ensemble size: 3 policies
- Mixture weight: 0.1
- Temperature decay: 0.995
- UCB constant: 2.0
- Diversity importance: 2.0
- Entropy weight: 0.02

### Convergence Criteria
- Policy change < 0.0005
- Relative error < 0.03
- Patience: 30 batches
- Minimum evaluation episodes: 5000

## Expected Results

For the motility problem (true probability: 2.41E-7):
- **Original**: ~1.37E-7 (57% of true value)
- **Improved**: Expected ~2.1-2.4E-7 (85-100% of true value)

The improved algorithm should provide:
1. Estimates within 15% of true probability
2. Confidence intervals containing true value
3. Relative error < 3%
4. Stable convergence

## Monitoring Progress

Watch for these indicators during execution:
- **Temperature Decay**: Should decrease from 1.0 to 0.1
- **Policy Convergence**: Change should decrease over time
- **Success Rate**: Should increase during training
- **Diversity Scores**: Higher scores indicate better exploration
- **Weight Distribution**: More balanced than original

This comprehensive solution addresses the underestimation problem through multiple complementary techniques, ensuring robust and accurate probability estimates.