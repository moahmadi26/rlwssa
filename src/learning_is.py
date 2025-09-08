"""
Learning-based Importance Sampling via Stochastic Optimal Control for Stochastic Reaction Networks
Based on Ben Hammouda et al., 2023

This implementation uses a sigmoid ansatz function with only d+1 parameters to learn optimal
importance sampling controls for rare event estimation.
"""

import numpy as np
import math
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from prism_parser import parser
from utils import suppress_c_output
import time


@dataclass
class ISConfig:
    """Configuration for the learning-based IS algorithm"""
    target_index: int
    target_value: float  # γ in the paper
    max_time: float  # T in the paper
    dt_learning: float  # Δt_pl (coarse time step for learning)
    dt_estimation: float  # Δt_f (fine time step for final estimation)
    num_iterations: int  # optimization iterations
    batch_size: int  # trajectories per iteration
    learning_rate: float  # for Adam optimizer
    target_condition: str = 'greater'  # 'greater' or 'equal'
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    mc_samples_final: int = 100000  # samples for final MC estimation
    clip_exp: float = 100.0  # clip exponentials to avoid overflow


class SigmoidAnsatz:
    """
    Implements the sigmoid ansatz function from Eq 2.15:
    u(t,x;β) = 1 / (1 + b0 * exp(-β0*(γ - x_i) - sum_j β_j*(γ - x_j)))
    
    where:
    - β_space ∈ R^d are the spatial parameters (one per species)
    - β_time is absorbed into β0 for simplicity
    - b0, β0 are set to fit the final condition
    """
    
    def __init__(self, dim: int, target_index: int, target_value: float, 
                 initial_value: float, target_condition: str = 'greater', 
                 clip_exp: float = 100.0):
        self.dim = dim
        self.target_index = target_index
        self.target_value = target_value
        self.initial_value = initial_value
        self.target_condition = target_condition
        self.clip_exp = clip_exp
        
        # Initialize parameters
        self.beta_space = np.zeros(dim)  # β_j for j ≠ i
        self.beta_0 = 10.0  # β_i (for target species)
        self.b0 = 1.0  # scaling factor
        
        # Set b0 and β0 based on the target condition
        if target_condition == 'equal':
            # For exact equality, use a narrow sigmoid centered at target
            self.b0 = 1.0
            self.beta_0 = 10.0
        else:
            # For > condition, standard setup
            self.b0 = 1.0
            self.beta_0 = 10.0
        
    def evaluate(self, t: float, x: np.ndarray, T: float) -> float:
        """
        Evaluate u(t,x;β)
        
        Args:
            t: current time
            x: current state vector
            T: final time
        
        Returns:
            Value of u(t,x;β)
        """
        # Time factor (approaches 1 as t→T)
        time_factor = t / T if T > 0 else 1.0
        
        # Compute exponent
        exponent = -self.beta_0 * (self.target_value - x[self.target_index])
        
        # Add contributions from other species
        for j in range(self.dim):
            if j != self.target_index:
                exponent -= self.beta_space[j] * (self.target_value - x[j])
        
        # Clip to avoid overflow
        exponent = np.clip(exponent, -self.clip_exp, self.clip_exp)
        
        # Sigmoid function
        return 1.0 / (1.0 + self.b0 * np.exp(exponent))
    
    def evaluate_batch(self, t: float, x_batch: np.ndarray, T: float) -> np.ndarray:
        """Vectorized evaluation for batch of states"""
        batch_size = x_batch.shape[0]
        results = np.zeros(batch_size)
        
        for i in range(batch_size):
            results[i] = self.evaluate(t, x_batch[i], T)
        
        return results
    
    def gradient(self, t: float, x: np.ndarray, T: float) -> np.ndarray:
        """
        Compute gradient of u with respect to β_space
        
        Returns:
            Gradient vector ∂u/∂β_j for j ≠ target_index
        """
        u_val = self.evaluate(t, x, T)
        
        # For sigmoid u = 1/(1 + b0*exp(-z)), we have ∂u/∂z = u^2 * b0 * exp(-z)
        exponent = -self.beta_0 * (self.target_value - x[self.target_index])
        for j in range(self.dim):
            if j != self.target_index:
                exponent -= self.beta_space[j] * (self.target_value - x[j])
        
        exponent = np.clip(exponent, -self.clip_exp, self.clip_exp)
        exp_val = np.exp(exponent)
        
        # Common factor for all gradients
        common_factor = u_val * u_val * self.b0 * exp_val
        
        # Gradient with respect to each β_j
        grad = np.zeros(self.dim)
        for j in range(self.dim):
            if j != self.target_index:
                # ∂u/∂β_j = common_factor * (γ - x_j)
                grad[j] = common_factor * (self.target_value - x[j])
        
        return grad


class TauLeapIS:
    """
    Tau-leap simulation with importance sampling control
    Implements Algorithm from Section 2.3 of the paper
    """
    
    def __init__(self, model, config: ISConfig):
        self.model = model
        self.config = config
        self.num_reactions = len(model.get_reactions_vector())
        self.num_species = len(model.get_initial_state())
        self.stoichiometry = np.array(model.get_reactions_vector())  # ν matrix
        
    def compute_propensities(self, x: np.ndarray) -> np.ndarray:
        """Compute reaction propensities a_j(x)"""
        a = np.zeros(self.num_reactions)
        x_tuple = tuple(max(0, int(xi)) for xi in x)
        
        for j in range(self.num_reactions):
            a[j] = self.model.get_reaction_rate(x_tuple, j)
        
        return a
    
    def compute_is_control(self, t: float, x: np.ndarray, ansatz: SigmoidAnsatz, dt: float) -> np.ndarray:
        """
        Compute IS control from Eq 2.16:
        δ_j(n,x;β) = a_j(x) * sqrt(u((n+1)Δt, max(0, x+ν_j);β) / u((n+1)Δt, x;β))
        
        With convention that a_j/δ_j = 1 when both are 0
        """
        T = self.config.max_time
        t_next = min(t + dt, T)
        
        # Current u value
        u_current = ansatz.evaluate(t_next, x, T)
        
        # Compute propensities
        a = self.compute_propensities(x)
        
        # Initialize control
        delta = np.zeros(self.num_reactions)
        
        for j in range(self.num_reactions):
            if a[j] > 1e-10:  # Only compute if reaction can fire
                # State after reaction j fires
                x_next = np.maximum(0, x + self.stoichiometry[j])
                u_next = ansatz.evaluate(t_next, x_next, T)
                
                # IS control from Eq 2.16
                if u_current > 1e-10:
                    ratio = u_next / u_current
                    # Clip ratio to avoid extreme values
                    ratio = np.clip(ratio, 1e-6, 1e6)
                    delta[j] = a[j] * np.sqrt(ratio)
                    # Additional clipping for very large delta values
                    delta[j] = min(delta[j], 1000.0 / dt)  # limit λ = δ*dt
                else:
                    delta[j] = a[j]
            else:
                delta[j] = 0.0
        
        return delta
    
    def _check_trajectory_target(self, trajectory_states: List[np.ndarray]) -> bool:
        """
        Check if target was reached at any point during trajectory
        Handles different target conditions
        """
        target_condition = getattr(self.config, 'target_condition', 'greater_equal')
        
        for state in trajectory_states:
            if target_condition == 'less_equal':
                # Enzymatic case: s5 <= 25
                if state[self.config.target_index] <= self.config.target_value:
                    return True
            else:  # 'greater_equal' (default)
                # Standard case: species >= target_value
                if state[self.config.target_index] >= self.config.target_value:
                    return True
        return False
    
    def simulate_trajectory(self, ansatz: SigmoidAnsatz, dt: float, 
                          compute_gradient: bool = False) -> Dict:
        """
        Simulate a single trajectory with tau-leap IS
        
        Returns dictionary with:
            - final_state: final state x(T)
            - likelihood_ratio: W from Eq 2.4
            - reached_target: indicator 1{x_i(T) > γ}
            - gradient: ∂log(W)/∂β if compute_gradient=True
            - trajectory: full state trajectory if needed
        """
        T = self.config.max_time
        x = np.array(self.model.get_initial_state(), dtype=float)
        t = 0.0
        
        # Track likelihood ratio W (Eq 2.4)
        log_W = 0.0
        
        # For gradient computation (Lemma 2.9)
        if compute_gradient:
            # Pathwise gradient accumulator
            grad_log_W = np.zeros(self.num_species)
            trajectory_states = [x.copy()]
            trajectory_times = [t]
        else:
            grad_log_W = None
            trajectory_times = None
        
        # Always track states for target checking (but lighter storage if not computing gradients)
        trajectory_states = [x.copy()]
        
        # Tau-leap simulation loop
        num_steps = int(np.ceil(T / dt))
        
        for step in range(num_steps):
            if t >= T:
                break
            
            # Current time step (may be smaller at the end)
            current_dt = min(dt, T - t)
            
            # Compute propensities and IS control
            a = self.compute_propensities(x)
            delta = self.compute_is_control(t, x, ansatz, current_dt)
            
            # Sample number of firings for each reaction (Poisson)
            # K_j ~ Poisson(δ_j * Δt)
            K = np.zeros(self.num_reactions, dtype=int)
            for j in range(self.num_reactions):
                if delta[j] > 0:
                    lambda_j = delta[j] * current_dt
                    # Clip lambda to avoid overflow in Poisson sampling
                    lambda_j = min(lambda_j, 1000.0)  # numpy.random.poisson limit
                    if lambda_j > 0:
                        K[j] = np.random.poisson(lambda_j)
            
            # Update likelihood ratio (Eq 2.4)
            # W *= prod_j (a_j/δ_j)^K_j * exp((δ_j - a_j)*Δt)
            for j in range(self.num_reactions):
                if delta[j] > 1e-10:
                    # Standard case
                    ratio = a[j] / delta[j] if delta[j] > 0 else 1.0
                    log_W += K[j] * np.log(ratio) if ratio > 0 else 0
                    log_W += (delta[j] - a[j]) * current_dt
                elif a[j] > 1e-10 and K[j] > 0:
                    # This shouldn't happen (K[j] should be 0 if delta[j] = 0)
                    # But handle it gracefully
                    log_W = -np.inf
                    break
            
            # Gradient computation (if needed)
            if compute_gradient and log_W > -np.inf:
                # Compute pathwise derivative using Lemma 2.9
                # This requires tracking how δ depends on β through u
                u_current = ansatz.evaluate(t + current_dt, x, T)
                if u_current > 1e-10:
                    grad_u = ansatz.gradient(t + current_dt, x, T)
                    
                    for j in range(self.num_reactions):
                        if delta[j] > 1e-10 and K[j] > 0:
                            # Contribution from this reaction
                            x_next = np.maximum(0, x + self.stoichiometry[j])
                            u_next = ansatz.evaluate(t + current_dt, x_next, T)
                            grad_u_next = ansatz.gradient(t + current_dt, x_next, T)
                            
                            # Gradient of log(δ_j) w.r.t. β
                            # δ_j ∝ sqrt(u_next/u_current)
                            # log(δ_j) = 0.5*(log(u_next) - log(u_current)) + const
                            if u_next > 1e-10:
                                grad_log_delta = 0.5 * (grad_u_next/u_next - grad_u/u_current)
                                grad_log_W += K[j] * (-grad_log_delta)  # negative because W ∝ (a/δ)^K
                                grad_log_W += current_dt * a[j] * 0.5 * (grad_u_next/u_next - grad_u/u_current) / np.sqrt(u_next/u_current)
            
            # Update state
            for j in range(self.num_reactions):
                if K[j] > 0:
                    x += K[j] * self.stoichiometry[j]
            
            # Ensure non-negative populations
            x = np.maximum(0, x)
            
            # Update time
            t += current_dt
            
            # Store trajectory 
            trajectory_states.append(x.copy())
            if compute_gradient:
                trajectory_times.append(t)
        
        # Check if target was ever reached during the trajectory (within time T)
        # This should check the entire trajectory, not just final state
        reached_target = self._check_trajectory_target(trajectory_states)
        
        # Convert log-likelihood to likelihood
        W = np.exp(np.clip(log_W, -self.config.clip_exp, self.config.clip_exp))
        
        result = {
            'final_state': x,
            'likelihood_ratio': W,
            'log_likelihood_ratio': log_W,
            'reached_target': reached_target,
            'gradient': grad_log_W
        }
        
        if compute_gradient:
            result['trajectory_states'] = trajectory_states
            result['trajectory_times'] = trajectory_times
        
        return result


class AdamOptimizer:
    """Adam optimizer for parameter updates"""
    
    def __init__(self, dim: int, learning_rate: float, beta1: float = 0.9, 
                 beta2: float = 0.999, epsilon: float = 1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # Moment estimates
        self.m = np.zeros(dim)
        self.v = np.zeros(dim)
        self.t = 0
    
    def update(self, params: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """
        Update parameters using Adam
        
        Args:
            params: current parameters
            gradient: gradient estimate
        
        Returns:
            Updated parameters
        """
        self.t += 1
        
        # Update biased moments
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * gradient**2
        
        # Bias correction
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        
        # Update parameters
        params_new = params - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        return params_new


class LearningBasedIS:
    """
    Main class for learning-based importance sampling
    """
    
    def __init__(self, model_path: str, config: ISConfig):
        with suppress_c_output():
            self.model = parser(model_path)
        self.config = config
        
        # Initialize components
        initial_state = self.model.get_initial_state()
        self.ansatz = SigmoidAnsatz(
            dim=len(initial_state),
            target_index=config.target_index,
            target_value=config.target_value,
            initial_value=initial_state[config.target_index],
            target_condition=getattr(config, 'target_condition', 'greater'),
            clip_exp=config.clip_exp
        )
        
        self.simulator = TauLeapIS(self.model, config)
        
        # Only optimize non-target species parameters
        opt_dim = self.ansatz.dim - 1  # exclude target species
        self.optimizer = AdamOptimizer(
            dim=opt_dim,
            learning_rate=config.learning_rate,
            beta1=config.adam_beta1,
            beta2=config.adam_beta2,
            epsilon=config.adam_epsilon
        )
        
        # Training history
        self.loss_history = []
        self.estimate_history = []
        self.variance_history = []
    
    def train(self, verbose: bool = True) -> None:
        """
        Train the ansatz parameters using pathwise gradient estimation
        """
        if verbose:
            print(f"Starting learning-based IS training")
            target_condition = getattr(self.config, 'target_condition', 'greater_equal')
            condition_symbol = "<=" if target_condition == 'less_equal' else ">="
            print(f"Target: species {self.config.target_index} {condition_symbol} {self.config.target_value}")
            print(f"Time horizon: {self.config.max_time}")
            print(f"Learning dt: {self.config.dt_learning}")
            print(f"Batch size: {self.config.batch_size}")
            print(f"Iterations: {self.config.num_iterations}")
            print("-" * 50)
        
        for iteration in range(self.config.num_iterations):
            # Collect batch of trajectories
            batch_gradients = []
            batch_estimators = []
            batch_reached = 0
            
            if iteration == 0:
                print(f"Starting iteration {iteration}, current β_space: {self.ansatz.beta_space}")
            
            for traj_idx in range(self.config.batch_size):
                # Simulate with gradient computation
                result = self.simulator.simulate_trajectory(
                    self.ansatz, 
                    self.config.dt_learning,
                    compute_gradient=True
                )
                
                # Collect statistics
                if result['reached_target']:
                    batch_reached += 1
                    # For rare event, the estimator is W * 1{reached}
                    estimator = result['likelihood_ratio']
                    batch_estimators.append(estimator)
                    
                    # Gradient of log(W * 1{reached}) = gradient of log(W) when reached
                    if result['gradient'] is not None:
                        batch_gradients.append(result['gradient'] * estimator)
                else:
                    batch_estimators.append(0.0)
                    if result['gradient'] is not None:
                        batch_gradients.append(np.zeros_like(result['gradient']))
            
            # Compute batch statistics
            mean_estimate = np.mean(batch_estimators)
            if len(batch_estimators) > 1:
                variance = np.var(batch_estimators, ddof=1)
            else:
                variance = 0.0
            
            self.estimate_history.append(mean_estimate)
            self.variance_history.append(variance)
            
            # Compute gradient estimate (only for non-target parameters)
            if batch_gradients:
                # Average gradient over batch
                mean_gradient_full = np.mean(batch_gradients, axis=0)
                # Extract non-target parameters
                mean_gradient = np.delete(mean_gradient_full, self.config.target_index)
            else:
                mean_gradient = np.zeros(self.ansatz.dim - 1)
            
            # Compute "loss" (negative of rare event probability for minimization)
            loss = -mean_estimate
            self.loss_history.append(loss)
            
            # Update parameters using Adam
            current_params = np.delete(self.ansatz.beta_space, self.config.target_index)
            new_params = self.optimizer.update(current_params, -mean_gradient)  # maximize probability
            
            # Update ansatz parameters (insert back target index with value 0)
            new_beta_space = np.insert(new_params, self.config.target_index, 0.0)
            self.ansatz.beta_space = new_beta_space
            
            # Print progress
            if verbose and (iteration % max(1, self.config.num_iterations // 20) == 0 or iteration == 0):
                success_rate = batch_reached / self.config.batch_size
                rel_error = np.sqrt(variance) / mean_estimate if mean_estimate > 0 else np.inf
                print(f"Iter {iteration:3d}: P≈{mean_estimate:.3e}, "
                      f"Var={variance:.3e}, RelErr={rel_error:.3f}, "
                      f"Success={success_rate:.1%}, Loss={loss:.3e}")
        
        if verbose:
            print("-" * 50)
            print("Training completed!")
            print(f"Final estimate: {self.estimate_history[-1]:.3e}")
            print(f"Final variance: {self.variance_history[-1]:.3e}")
            print(f"Variance reduction: {self.variance_history[0]/self.variance_history[-1]:.1f}x" 
                  if self.variance_history[-1] > 0 else "N/A")
    
    def estimate_probability(self, num_samples: Optional[int] = None, 
                           verbose: bool = True) -> Dict:
        """
        Final probability estimation using fine time step
        
        Returns dictionary with:
            - probability: estimated rare event probability
            - standard_error: standard error of estimate
            - variance: variance of estimator
            - confidence_interval: 95% CI
            - num_samples: number of samples used
            - variance_reduction: compared to standard tau-leap
        """
        if num_samples is None:
            num_samples = self.config.mc_samples_final
        
        if verbose:
            print(f"\nFinal estimation with {num_samples} samples")
            print(f"Estimation dt: {self.config.dt_estimation}")
        
        # Collect samples with fine time step
        estimators = []
        reached_count = 0
        
        start_time = time.time()
        
        for i in range(num_samples):
            result = self.simulator.simulate_trajectory(
                self.ansatz,
                self.config.dt_estimation,
                compute_gradient=False
            )
            
            if result['reached_target']:
                reached_count += 1
                estimators.append(result['likelihood_ratio'])
            else:
                estimators.append(0.0)
            
            # Progress update
            if verbose and (i + 1) % max(1, num_samples // 10) == 0:
                print(f"  Progress: {i+1}/{num_samples} samples "
                      f"({100*(i+1)/num_samples:.0f}%), "
                      f"{reached_count} reached target")
        
        elapsed = time.time() - start_time
        
        # Compute statistics
        probability = np.mean(estimators)
        variance = np.var(estimators, ddof=1) if len(estimators) > 1 else 0.0
        standard_error = np.sqrt(variance / num_samples)
        
        # 95% confidence interval
        ci_lower = probability - 1.96 * standard_error
        ci_upper = probability + 1.96 * standard_error
        
        # Estimate variance reduction compared to standard tau-leap
        # Standard tau-leap variance for rare event ≈ p(1-p)/n ≈ p/n for small p
        standard_variance = probability / num_samples if probability > 0 else 0
        actual_variance = variance / num_samples
        variance_reduction = standard_variance / actual_variance if actual_variance > 0 else np.inf
        
        results = {
            'probability': probability,
            'standard_error': standard_error,
            'variance': variance,
            'confidence_interval': (ci_lower, ci_upper),
            'num_samples': num_samples,
            'reached_count': reached_count,
            'success_rate': reached_count / num_samples,
            'variance_reduction': variance_reduction,
            'computation_time': elapsed,
            'samples_per_second': num_samples / elapsed
        }
        
        if verbose:
            print(f"\n" + "="*50)
            print("FINAL RESULTS")
            print("="*50)
            print(f"Probability estimate: {probability:.6e}")
            print(f"Standard error: {standard_error:.6e}")
            print(f"Relative error: {standard_error/probability:.4f}" if probability > 0 else "N/A")
            print(f"95% CI: [{ci_lower:.6e}, {ci_upper:.6e}]")
            print(f"Success rate: {reached_count}/{num_samples} = {results['success_rate']:.3%}")
            print(f"Variance reduction: {variance_reduction:.1f}x")
            print(f"Computation time: {elapsed:.2f}s ({results['samples_per_second']:.0f} samples/s)")
        
        return results
    
    def compare_with_standard_tau_leap(self, num_samples: int = 10000, 
                                      verbose: bool = True) -> Dict:
        """
        Compare with standard tau-leap (no importance sampling)
        """
        if verbose:
            print(f"\nComparison with standard tau-leap ({num_samples} samples)")
        
        # Create a neutral ansatz (u = 1 everywhere, so δ = a)
        initial_state = self.model.get_initial_state()
        neutral_ansatz = SigmoidAnsatz(
            dim=self.ansatz.dim,
            target_index=self.config.target_index,
            target_value=self.config.target_value,
            initial_value=initial_state[self.config.target_index],
            target_condition=getattr(self.config, 'target_condition', 'greater')
        )
        neutral_ansatz.beta_space[:] = 0
        neutral_ansatz.beta_0 = 0
        neutral_ansatz.b0 = 0  # This makes u = 1
        
        # Run standard tau-leap
        reached = 0
        start_time = time.time()
        
        for i in range(num_samples):
            result = self.simulator.simulate_trajectory(
                neutral_ansatz,
                self.config.dt_estimation,
                compute_gradient=False
            )
            if result['reached_target']:
                reached += 1
        
        elapsed = time.time() - start_time
        probability_standard = reached / num_samples
        variance_standard = probability_standard * (1 - probability_standard) / num_samples
        
        if verbose:
            print(f"Standard tau-leap: P≈{probability_standard:.6e}, "
                  f"reached {reached}/{num_samples}")
            print(f"Time: {elapsed:.2f}s")
        
        return {
            'probability': probability_standard,
            'variance': variance_standard,
            'reached_count': reached,
            'num_samples': num_samples,
            'computation_time': elapsed
        }