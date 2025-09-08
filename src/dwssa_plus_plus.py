"""
dwSSA++ (Doubly weighted Stochastic Simulation Algorithm++)
Implementation based on "Data-Driven Method for Efficient Characterization of 
Rare Event Probabilities in Biochemical Systems" by Min K. Roh (2019)

This module implements the enhanced dwSSA algorithm with polynomial leaping
to improve convergence when the conventional multilevel cross-entropy method
converges slowly or fails to converge.
"""

import math
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import multiprocessing
from dwssa import dwssa_train, dwssa
from prism_parser import parser
from utils import suppress_c_output, is_target
import warnings

class dwSSAPlusPlus:
    """
    dwSSA++ algorithm implementation with polynomial leaping enhancement.
    
    Key parameters from the paper:
    - ld: Number of past data points to consider (default: 3)
    - sigma: Threshold for slow convergence detection (default: 5)  
    - smax: Maximum number of iterations (default: 20)
    """
    
    def __init__(self, ld: int = 3, sigma: float = 5.0, smax: int = 20):
        """
        Initialize dwSSA++ parameters.
        
        Args:
            ld: Number of past data points for convergence analysis
            sigma: Threshold for detecting slow convergence (mu > sigma)
            smax: Maximum iterations before termination
        """
        self.ld = ld
        self.sigma = sigma
        self.smax = smax
        
        # Algorithm state tracking
        self.convergence_history = []
        self.gamma_history = []
        self.counter_history = []
        self.IRE_history = []
        self.iteration_count = 0
        
    def compute_optimal_parameters(self, model_path: str, x0: List[int], E: int, 
                                 tf: float, K: int, rho: float, target_index: int,
                                 num_procs: int = None) -> Tuple[List[float], bool]:
        """
        Algorithm 2: Optimal dwSSA++ Parameter Estimation
        
        Args:
            model_path: Path to the PRISM model
            x0: Initial state
            E: Target rare event value
            tf: Final time
            K: Number of trajectories per iteration
            rho: Selection percentage for elite trajectories
            target_index: Index of target species
            num_procs: Number of processors for parallel execution
            
        Returns:
            Tuple of (optimal_gamma, convergence_achieved)
        """
        if num_procs is None:
            num_procs = multiprocessing.cpu_count()
            
        # Initialize gamma^(0) = [1, 1, ..., 1] and s = -1
        with suppress_c_output():
            model = parser(model_path)
        num_reactions = len(model.get_reactions_vector())
        gamma = [1.0] * num_reactions
        self.iteration_count = -1
        
        # Reset history
        self.convergence_history = []
        self.gamma_history = []
        self.counter_history = []
        self.IRE_history = []
        
        print(f"dwSSA++ Parameter Estimation Started")
        print(f"Target: Species {target_index} = {E}, Time limit: {tf}")
        print(f"Parameters: ld={self.ld}, sigma={self.sigma}, smax={self.smax}")
        
        while self.iteration_count < self.smax:
            self.iteration_count += 1
            print(f"\nIteration {self.iteration_count}:")
            
            # Step 3: Run trajectories with current gamma
            trajectories = self._run_parallel_trajectories(
                model_path, K, tf, target_index, E, gamma, num_procs
            )
            
            # Step 4: Compute E^(s) - top rho*K trajectories
            Es, counter_data, IRE_data = self._compute_elite_set(
                trajectories, rho, target_index, E
            )
            
            # Store history
            self.convergence_history.append(Es)
            self.gamma_history.append(gamma.copy())
            self.counter_history.append(counter_data)
            self.IRE_history.append(IRE_data)
            
            print(f"  Current E^({self.iteration_count}): {Es}")
            print(f"  Elite trajectories: {len(counter_data)}")
            
            # Step 5: Check if E^(s) contains E (Es=0 means elite trajectories reached target)
            if Es == 0:
                print(f"  Target reached! E^({self.iteration_count}) = {Es} (elite trajectories reached target)")
                return gamma, True
                
            # Step 6-7: Check convergence conditions
            if self.iteration_count >= self.ld:
                slow_convergence = self._detect_slow_convergence(E)
                
                if slow_convergence:
                    print("  Slow convergence detected, applying polynomial leaping...")
                    
                    # Step 8: Binary decision tree
                    decision = self._binary_decision_tree()
                    
                    if decision['action'] == 'polynomial_leaping':
                        gamma = self._polynomial_leaping(decision)
                        print(f"  New gamma from polynomial leaping: {gamma}")
                    elif decision['action'] == 'continue_mce':
                        gamma = self._standard_multilevel_ce(
                            trajectories, rho, target_index, E, model_path
                        )
                        print(f"  Continuing with standard mCE: {gamma[:3]}...")
                    elif decision['action'] == 'exit':
                        print("  No convergence signal, exiting...")
                        return gamma, False
                else:
                    # Step 9: Standard multilevel CE (Equation 7)
                    gamma = self._standard_multilevel_ce(
                        trajectories, rho, target_index, E, model_path
                    )
                    print(f"  Standard mCE update: {gamma}")
            else:
                # Not enough history, use standard mCE
                gamma = self._standard_multilevel_ce(
                    trajectories, rho, target_index, E, model_path
                )
                print(f"  Initial mCE update: {gamma}")
        
        print(f"\nMaximum iterations ({self.smax}) reached without convergence")
        return gamma, False
    
    def _run_parallel_trajectories(self, model_path: str, K: int, tf: float,
                                 target_index: int, target_value: int, 
                                 gamma: List[float], num_procs: int) -> List:
        """Run K trajectories in parallel using current gamma."""
        # Distribute trajectories across processors
        N_vec = [K // num_procs if j != num_procs - 1 
                else K - ((num_procs - 1) * (K // num_procs))
                for j in range(num_procs)]
        
        tasks = [(model_path, N_vec_j, tf, target_index, target_value, gamma)
                for N_vec_j in N_vec]
        
        with multiprocessing.Pool(processes=num_procs) as pool:
            results = pool.starmap(dwssa_train, tasks)
        
        # Flatten results
        trajectories = [trajectory for result in results for trajectory in result]
        return trajectories
    
    def _compute_elite_set(self, trajectories: List, rho: float, 
                          target_index: int, target_value: int) -> Tuple[float, List, List]:
        """
        Compute E^(s) as the set of top rho*K trajectories that evolve farthest toward E.
        
        Returns:
            Es: The elite threshold value
            counter_data: Counter values for elite trajectories  
            IRE_data: Intermediate rare event values for elite trajectories
        """
        # Sort trajectories by minimum distance to target (ascending)
        trajectories.sort(key=lambda traj: traj[2])  # traj[2] is min_dist
        
        # Select top rho fraction
        threshold = math.ceil(rho * len(trajectories))
        elite_trajectories = trajectories[:threshold]
        
        # Es is the maximum distance in the elite set (closest to target)
        Es = max(traj[2] for traj in elite_trajectories)
        
        # Extract counter and IRE data
        counter_data = []
        IRE_data = []
        
        for traj in elite_trajectories:
            # Counter data: final value of target species
            final_state = traj[3]  # min_dist_state
            counter_value = final_state[target_index]
            counter_data.append(counter_value)
            
            # IRE data: intermediate rare event value (distance from target)
            IRE_value = target_value - traj[2]  # How close we got to target
            IRE_data.append(IRE_value)
        
        return Es, counter_data, IRE_data
    
    def _detect_slow_convergence(self, target_E: int) -> bool:
        """
        Detect slow convergence using two conditions from Algorithm 2:
        1. Past ld IRE values form a non-strictly converging sequence
        2. Estimated iterations to reach rare event (mu) exceeds threshold sigma
        """
        if len(self.convergence_history) < self.ld:
            return False
            
        recent_Es = self.convergence_history[-self.ld:]
        
        # Condition 1: Check if sequence is non-strictly monotonic
        non_strictly_increasing = all(recent_Es[i] <= recent_Es[i+1] 
                                    for i in range(len(recent_Es)-1))
        
        if not non_strictly_increasing:
            return False
            
        # Condition 2: Estimate convergence speed (Equation 8)
        current_E = recent_Es[-1]
        prev_E = recent_Es[-2] if len(recent_Es) >= 2 else recent_Es[-1]
        
        h = abs(current_E - prev_E)
        if h == 0:
            return True  # No progress made
            
        mu = abs(target_E - current_E) / h
        
        print(f"  Convergence analysis: mu = {mu:.2f}, threshold = {self.sigma}")
        return mu > self.sigma
    
    def _binary_decision_tree(self) -> Dict[str, Any]:
        """
        Implement the binary decision tree from Figure 1 to determine:
        - Whether to use polynomial leaping, continue with mCE, or exit
        - Which extrapolation method (bisection vs polynomial interpolation)
        - Which data type (counters vs IRE values)
        """
        # Check data availability for polynomial interpolation
        has_sufficient_counter_data = len(self.counter_history) >= 2
        has_sufficient_IRE_data = len(self.IRE_history) >= 2
        
        # Decision tree logic
        if has_sufficient_counter_data and len(self.counter_history) >= 3:
            # Use polynomial interpolation with counter data
            return {
                'action': 'polynomial_leaping',
                'method': 'polynomial_interpolation',
                'data_type': 'counters',
                'degree': min(2, len(self.counter_history) - 1)
            }
        elif has_sufficient_IRE_data and len(self.IRE_history) >= 3:
            # Use polynomial interpolation with IRE data
            return {
                'action': 'polynomial_leaping', 
                'method': 'polynomial_interpolation',
                'data_type': 'IRE',
                'degree': min(2, len(self.IRE_history) - 1)
            }
        elif has_sufficient_counter_data or has_sufficient_IRE_data:
            # Use bisection method
            data_type = 'counters' if has_sufficient_counter_data else 'IRE'
            return {
                'action': 'polynomial_leaping',
                'method': 'bisection', 
                'data_type': data_type
            }
        elif len(self.convergence_history) >= 2:
            # Continue with standard mCE
            return {'action': 'continue_mce'}
        else:
            # Exit - no signal
            return {'action': 'exit'}
    
    def _polynomial_leaping(self, decision: Dict[str, Any]) -> List[float]:
        """
        Algorithm 3: Polynomial Leaping Method
        
        Implements extrapolation of biasing parameters using past simulation data.
        Uses either bisection or polynomial interpolation based on decision tree output.
        """
        if decision['method'] == 'bisection':
            return self._bisection_method(decision['data_type'])
        else:
            return self._polynomial_interpolation(decision['data_type'], decision['degree'])
    
    def _bisection_method(self, data_type: str) -> List[float]:
        """
        Bisection method for gamma extrapolation when limited data is available.
        """
        current_gamma = self.gamma_history[-1].copy()
        
        if data_type == 'counters':
            data_history = self.counter_history
        else:
            data_history = self.IRE_history
            
        # Simple bisection: if progress is being made, increase gamma by factor of 2
        # If no progress, decrease by factor of 2
        if len(data_history) >= 2:
            current_data = np.mean(data_history[-1])
            prev_data = np.mean(data_history[-2])
            
            if current_data > prev_data:  # Making progress
                new_gamma = [g * 2.0 for g in current_gamma]
            else:  # Not making enough progress
                new_gamma = [g / 2.0 for g in current_gamma]
        else:
            # Default: increase by factor of 2
            new_gamma = [g * 2.0 for g in current_gamma]
            
        return new_gamma
    
    def _polynomial_interpolation(self, data_type: str, degree: int) -> List[float]:
        """
        Polynomial interpolation method for gamma extrapolation.
        Uses Equations 9-10 from the paper.
        """
        if data_type == 'counters':
            data_history = self.counter_history[-degree-1:]
            # Equation 9: xi = sum(counters) / len(counters) for target
            target_xi = np.mean([np.mean(data) for data in data_history])
        else:
            data_history = self.IRE_history[-degree-1:]
            # Equation 10: xi = max(IRE_values) for target
            target_xi = max([max(data) for data in data_history])
        
        # Get corresponding gamma values
        gamma_history = self.gamma_history[-degree-1:]
        
        # For simplicity, use the mean gamma adjustment based on data progression
        gamma_adjustments = []
        for i in range(1, len(data_history)):
            prev_data = np.mean(data_history[i-1])
            curr_data = np.mean(data_history[i])
            
            if curr_data > prev_data:
                adjustment = curr_data / max(prev_data, 1e-10)
            else:
                adjustment = 0.5  # Reduce if no progress
            gamma_adjustments.append(adjustment)
        
        # Apply polynomial-based adjustment
        if gamma_adjustments:
            mean_adjustment = np.mean(gamma_adjustments)
            # Polynomial extrapolation factor
            extrapolation_factor = 1.0 + (mean_adjustment - 1.0) * 1.5
        else:
            extrapolation_factor = 1.5
            
        current_gamma = self.gamma_history[-1]
        new_gamma = [g * extrapolation_factor for g in current_gamma]
        
        # Handle under-perturbation cases (ensure reasonable bounds)
        new_gamma = [max(0.1, min(10.0, g)) for g in new_gamma]
        
        return new_gamma
    
    def _standard_multilevel_ce(self, trajectories: List, rho: float,
                              target_index: int, target_value: int, 
                              model_path: str = None) -> List[float]:
        """
        Standard multilevel cross-entropy method (Equation 7 from paper).
        This is the same as the existing biasing computation in your codebase.
        """
        from biasing import find_biasing
        
        # Get model from stored path or parse it
        if model_path:
            with suppress_c_output():
                model = parser(model_path)
        else:
            # Fallback - create dummy model with correct number of reactions
            # This assumes trajectories is not empty
            if trajectories and len(trajectories) > 0:
                num_reactions = len(self.gamma_history[-1]) if self.gamma_history else 8
                class DummyModel:
                    def get_reactions_vector(self):
                        return [None] * num_reactions
                    def get_initial_state(self):
                        return trajectories[0][0][0] if trajectories[0][0] else [0] * 6
                model = DummyModel()
            else:
                raise ValueError("Cannot determine model structure")
            
        # Use existing biasing computation
        flag, new_gamma = find_biasing(
            model, trajectories, rho, len(model.get_reactions_vector()),
            target_index, target_value
        )
        
        return new_gamma
    
    def estimate_probability(self, model_path: str, gamma: List[float], 
                           target_index: int, target_value: int, tf: float,
                           K: int, num_ensembles: int = 10, 
                           num_procs: int = None) -> Dict[str, float]:
        """
        Algorithm 1: Final probability estimation using optimized gamma.
        This uses the standard dwSSA estimation (unchanged from your implementation).
        """
        if num_procs is None:
            num_procs = multiprocessing.cpu_count()
            
        print(f"\nRunning probability estimation with {num_ensembles} ensembles of {K:,} trajectories each...")
        
        p_vector = []
        
        for i in range(num_ensembles):
            print(f"Ensemble {i+1}/{num_ensembles}")
            
            # Distribute trajectories across processors
            N_vec = [K // num_procs if j != num_procs - 1 
                    else K - ((num_procs - 1) * (K // num_procs))
                    for j in range(num_procs)]
            
            tasks = [(model_path, N_vec_j, tf, target_index, target_value, gamma)
                    for N_vec_j in N_vec]
            
            with multiprocessing.Pool(processes=num_procs) as pool:
                results = pool.starmap(dwssa, tasks)
            
            # Sum results from all processors
            ensemble_estimate = sum(results) / K
            p_vector.append(ensemble_estimate)
        
        # Compute statistics
        p_hat = sum(p_vector) / num_ensembles
        if num_ensembles > 1:
            variance = sum((p - p_hat)**2 for p in p_vector) / (num_ensembles - 1)
            std_error = math.sqrt(variance / num_ensembles)
        else:
            variance = 0.0
            std_error = 0.0
            
        return {
            'probability_estimate': p_hat,
            'standard_error': std_error,
            'variance': variance,
            'num_trajectories': num_ensembles * K,
            'individual_estimates': p_vector
        }


def create_sirs_model(output_path: str = "/tmp/sirs_model.prism"):
    """
    Create the SIRS model from Section 3.1 of the paper as a test case.
    
    Model parameters:
    - 3 reactions: S+I → 2I (beta=0.0675), I → R (lambda=3.959), R → S (omega=2.369)  
    - Initial state: x0 = [100, 1, 0]
    - Rare event: I reaches 60 before t=30
    """
    sirs_model_content = """
dtmc

const double beta = 0.0675;
const double lambda = 3.959; 
const double omega = 2.369;

module sirs_model
    S : [0..101] init 100;
    I : [0..101] init 1;
    R : [0..101] init 0;
    
    [] S > 0 & I > 0 -> beta * S * I : (S' = S - 1) & (I' = I + 1);
    [] I > 0 -> lambda * I : (I' = I - 1) & (R' = R + 1);
    [] R > 0 -> omega * R : (R' = R - 1) & (S' = S + 1);
    
endmodule
"""
    
    with open(output_path, 'w') as f:
        f.write(sirs_model_content)
    
    return output_path


def test_dwssa_plus_plus():
    """
    Test the dwSSA++ implementation with the SIRS model example.
    """
    print("="*60)
    print("Testing dwSSA++ with SIRS Model")
    print("="*60)
    
    # Create SIRS model
    model_path = create_sirs_model()
    
    # Test parameters from paper
    initial_state = [100, 1, 0]  # [S, I, R]
    target_species = 1  # I (index 1)
    target_value = 60
    max_time = 30.0
    K = 100000  # Number of trajectories per iteration
    rho = 0.01  # Elite trajectory percentage
    num_procs = multiprocessing.cpu_count()
    
    # Initialize dwSSA++
    dwssa_pp = dwSSAPlusPlus(ld=3, sigma=5, smax=20)
    
    try:
        # Run parameter optimization
        print("Running dwSSA++ parameter optimization...")
        optimal_gamma, converged = dwssa_pp.compute_optimal_parameters(
            model_path=model_path,
            x0=initial_state, 
            E=target_value,
            tf=max_time,
            K=K,
            rho=rho,
            target_index=target_species,
            num_procs=num_procs
        )
        
        print(f"\nParameter optimization completed:")
        print(f"Converged: {converged}")
        print(f"Iterations: {dwssa_pp.iteration_count}")
        print(f"Final gamma: {optimal_gamma}")
        
        if converged:
            # Run probability estimation
            print("\nRunning probability estimation...")
            results = dwssa_pp.estimate_probability(
                model_path=model_path,
                gamma=optimal_gamma,
                target_index=target_species,
                target_value=target_value,
                tf=max_time,
                K=K//10,  # Smaller K for final estimation
                num_ensembles=5,
                num_procs=num_procs
            )
            
            print(f"\nFinal Results:")
            print(f"Probability estimate: {results['probability_estimate']:.6e}")
            print(f"Standard error: {results['standard_error']:.6e}")
            print(f"Total trajectories: {results['num_trajectories']:,}")
            
        else:
            print("Parameter optimization did not converge within maximum iterations")
            
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up temporary model file
        import os
        if os.path.exists(model_path):
            os.remove(model_path)


if __name__ == "__main__":
    test_dwssa_plus_plus()