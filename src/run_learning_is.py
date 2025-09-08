#!/usr/bin/env python3
"""
Main script for testing Learning-based Importance Sampling
Reproduces examples from Ben Hammouda et al., 2023
"""

import sys
import json
import numpy as np
from learning_is import LearningBasedIS, ISConfig


def create_config_from_json(json_path: str):
    """Create ISConfig from existing JSON configuration"""
    with open(json_path, 'r') as f:
        config = json.load(f)
    
    # Parse the model to get target_index
    from prism_parser import parser
    from utils import suppress_c_output
    
    with suppress_c_output():
        model = parser(config['model_path'])
    
    target_variable = config['target_variable']
    target_index = model.species_to_index_dict[target_variable]
    
    # Extract target condition from CSL property
    # Look inside the final parentheses for the condition
    csl_property = config.get('csl_property', '')
    import re
    
    # Match pattern like (X<=10) or (X>=90)
    condition_match = re.search(r'\(([^)]+[<>=]+[^)]+)\)', csl_property)
    
    if condition_match:
        condition_part = condition_match.group(1)  # e.g., "X<=10"
        if '<=' in condition_part:
            target_condition = 'less_equal'  # X <= value
        else:
            target_condition = 'greater_equal'  # X >= value
    else:
        target_condition = 'greater_equal'  # default
    
    # Adjust time steps based on max_time
    max_time = float(config['max_time'])
    if max_time <= 5.0:
        # Short time horizon - use finer steps
        dt_learning = 0.1
        dt_estimation = 0.05
    else:
        # Long time horizon - use coarser steps  
        dt_learning = 2.0
        dt_estimation = 1.0
    
    return ISConfig(
        target_index=target_index,
        target_value=float(config['target_value']),
        max_time=max_time,
        target_condition=target_condition,
        dt_learning=dt_learning,
        dt_estimation=dt_estimation,
        num_iterations=20,     # Reduced from 100
        batch_size=50,         # Reduced from 10000  
        learning_rate=0.01,
        mc_samples_final=5000  # Reduced from 100000
    )


def run_learning_is_experiment(model_path: str, config: ISConfig, 
                              model_name: str = "Unknown") -> None:
    """Run a complete learning-based IS experiment"""
    
    print("="*80)
    print(f"Learning-based Importance Sampling: {model_name}")
    print("="*80)
    print("*** VERSION DEBUG: NEW CONFIG LOADING ***")
    print(f"Model: {model_path}")
    print(f"DEBUG: config.target_value = {config.target_value}")
    print(f"DEBUG: config.max_time = {config.max_time}")
    print(f"DEBUG: config.target_condition = {getattr(config, 'target_condition', 'NOT_SET')}")
    
    target_condition = getattr(config, 'target_condition', 'greater_equal')
    if target_condition == 'less_equal':
        condition_symbol = "<="
    else:
        condition_symbol = ">="
    print(f"Target event: species[{config.target_index}] {condition_symbol} {config.target_value}")
    print(f"Time horizon: T = {config.max_time}")
    print(f"Learning parameters: dt={config.dt_learning}, "
          f"iterations={config.num_iterations}, batch={config.batch_size}")
    print(f"Final estimation: dt={config.dt_estimation}, "
          f"samples={config.mc_samples_final}")
    print()
    
    # Initialize the learning-based IS algorithm
    try:
        learner = LearningBasedIS(model_path, config)
        print(f"Model loaded successfully.")
        print(f"Number of species: {learner.ansatz.dim}")
        print(f"Number of reactions: {learner.simulator.num_reactions}")
        print(f"Initial state: {learner.model.get_initial_state()}")
        print()
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Phase 1: Parameter learning
    print("PHASE 1: Parameter Learning")
    print("-" * 40)
    try:
        learner.train(verbose=True)
        
        # Display learned parameters
        print(f"\nLearned parameters:")
        print(f"  β₀ (target species): {learner.ansatz.beta_0:.4f}")
        print(f"  b₀ (scaling): {learner.ansatz.b0:.4f}")
        print(f"  β_space (other species):")
        for i, beta in enumerate(learner.ansatz.beta_space):
            if i != config.target_index:
                print(f"    Species {i}: {beta:.4f}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Phase 2: Final probability estimation
    print(f"\nPHASE 2: Final Probability Estimation")
    print("-" * 40)
    try:
        results = learner.estimate_probability(verbose=True)
        
    except Exception as e:
        print(f"Error during final estimation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Phase 3: Comparison with standard method
    print(f"\nPHASE 3: Comparison with Standard Tau-leap")
    print("-" * 40)
    try:
        standard_results = learner.compare_with_standard_tau_leap(
            num_samples=min(10000, config.mc_samples_final // 5),
            verbose=True
        )
        
        # Summary comparison
        if standard_results['probability'] > 0:
            variance_reduction = (standard_results['variance'] / 
                                (results['variance'] / results['num_samples']))
            efficiency_gain = (results['samples_per_second'] / 
                             (standard_results['num_samples'] / standard_results['computation_time']))
        else:
            variance_reduction = np.inf
            efficiency_gain = np.inf
        
        print(f"\nSUMMARY COMPARISON")
        print("-" * 40)
        print(f"Learning-based IS:")
        print(f"  Probability: {results['probability']:.6e} ± {results['standard_error']:.6e}")
        print(f"  Success rate: {results['success_rate']:.3%}")
        print(f"  Samples/second: {results['samples_per_second']:.0f}")
        print(f"Standard tau-leap:")
        print(f"  Probability: {standard_results['probability']:.6e}")
        print(f"  Success rate: {standard_results['reached_count']/standard_results['num_samples']:.3%}")
        print(f"  Samples/second: {standard_results['num_samples']/standard_results['computation_time']:.0f}")
        print(f"Improvements:")
        print(f"  Variance reduction: {results['variance_reduction']:.1f}x")
        print(f"  Relative efficiency: {efficiency_gain:.1f}x")
        
    except Exception as e:
        print(f"Error during comparison: {e}")
        import traceback
        traceback.print_exc()
    
    # Save results
    output_file = f"learning_is_results_{model_name.lower().replace(' ', '_')}.json"
    try:
        output_data = {
            'model_path': model_path,
            'model_name': model_name,
            'config': {
                'target_index': config.target_index,
                'target_value': config.target_value,
                'max_time': config.max_time,
                'dt_learning': config.dt_learning,
                'dt_estimation': config.dt_estimation,
                'num_iterations': config.num_iterations,
                'batch_size': config.batch_size,
                'learning_rate': config.learning_rate
            },
            'learned_parameters': {
                'beta_0': float(learner.ansatz.beta_0),
                'b0': float(learner.ansatz.b0),
                'beta_space': [float(x) for x in learner.ansatz.beta_space]
            },
            'training_history': {
                'loss_history': [float(x) for x in learner.loss_history],
                'estimate_history': [float(x) for x in learner.estimate_history],
                'variance_history': [float(x) for x in learner.variance_history]
            },
            'final_results': {
                'probability': float(results['probability']),
                'standard_error': float(results['standard_error']),
                'variance_reduction': float(results['variance_reduction']),
                'success_rate': float(results['success_rate']),
                'computation_time': float(results['computation_time']),
                'samples_per_second': float(results['samples_per_second'])
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {output_file}")
        
    except Exception as e:
        print(f"Error saving results: {e}")


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python run_learning_is.py <config.json>")
        print("Examples:")
        print("  python run_learning_is.py ../models/enzymatic_futile_cycle/low_s5.json")
        print("  python run_learning_is.py ../models/yeast_polarization/high_gbg.json") 
        print("  python run_learning_is.py ../models/motility_regulation/high_cody.json")
        print("  python run_learning_is.py ../models/pure_decay/survival.json")
        print("  python run_learning_is.py ../models/michaelis_menten/high_product.json")
        sys.exit(1)
    
    json_path = sys.argv[1]
    
    # Load configuration from JSON
    try:
        config = create_config_from_json(json_path)
        
        # Extract model info from JSON
        with open(json_path, 'r') as f:
            json_config = json.load(f)
        
        model_path = json_config['model_path']
        model_name = json_config.get('model_name', 'Unknown Model')
        
    except Exception as e:
        print(f"Error loading configuration from {json_path}: {e}")
        sys.exit(1)
    
    # Run the experiment
    run_learning_is_experiment(model_path, config, model_name)


if __name__ == "__main__":
    main()
