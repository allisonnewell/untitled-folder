"""
Macroeconomic Models - Value Function Iteration Solvers

Three canonical models solved using VFI:
1. Consumption-Savings with CES preferences and income shocks
2. Robinson Crusoe production economy with capital accumulation and TFP shocks
3. Labor supply model with wage shocks
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# UTILITY FUNCTIONS & MARKOV CHAIN DISCRETIZATION
# ============================================================================

def ces_utility(c, gamma):
    """CES utility function: U(c) = (c^(1-γ) - 1) / (1-γ)"""
    if gamma == 1.0:
        return np.log(np.maximum(c, 1e-8))
    else:
        return (np.maximum(c, 1e-8) ** (1 - gamma) - 1) / (1 - gamma)


def labor_utility(l, chi, eta):
    """Labor disutility: χ * l^(1+1/η) / (1+1/η)"""
    return -chi * (np.maximum(l, 0) ** (1 + 1/eta)) / (1 + 1/eta)


def tauchen_discretization(rho, sigma, n_points=9, n_std=3):
    """
    Discretize AR(1) process: x_{t+1} = rho * x_t + sigma * eps_{t+1}
    Returns grid points and transition matrix
    """
    # Grid points
    z_max = n_std * sigma / np.sqrt(1 - rho**2)
    z_min = -z_max
    grid = np.linspace(z_min, z_max, n_points)
    
    # Spacing
    step = (z_max - z_min) / (n_points - 1)
    
    # Transition probabilities
    P = np.zeros((n_points, n_points))
    
    for i in range(n_points):
        for j in range(n_points):
            if j == 0:
                P[i, j] = norm_cdf((grid[0] - rho * grid[i] + step/2) / sigma)
            elif j == n_points - 1:
                P[i, j] = 1 - norm_cdf((grid[-1] - rho * grid[i] - step/2) / sigma)
            else:
                P[i, j] = norm_cdf((grid[j] - rho * grid[i] + step/2) / sigma) - \
                         norm_cdf((grid[j] - rho * grid[i] - step/2) / sigma)
    
    # Normalize rows
    P = P / P.sum(axis=1, keepdims=True)
    
    return grid, P


def norm_cdf(x):
    """Standard normal CDF"""
    return 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))


# ============================================================================
# MODEL 1: CONSUMPTION-SAVINGS WITH CES PREFERENCES
# ============================================================================

class ConsumptionSavingsModel:
    """
    Stochastic Consumption-Savings Model
    
    Household maximizes: E_0 sum_{t=0}^∞ β^t U(c_t)
    Subject to: a_{t+1} = (1+r)a_t + y_t - c_t
                a_{t+1} ≥ 0 (no borrowing)
                y_{t+1} = ρ y_t + σ_y ε_{t+1}
    
    Parameters:
    -----------
    beta : float, discount factor
    r : float, interest rate
    gamma : float, risk aversion
    rho : float, AR(1) persistence
    sigma_y : float, income shock std dev
    n_a : int, size of asset grid
    a_min : float, minimum assets (default 0.01)
    a_max : float, maximum assets (default 50)
    """
    
    def __init__(self, beta, r, gamma, rho, sigma_y, n_a=100, a_min=0.01, a_max=50):
        self.beta = beta
        self.r = r
        self.gamma = gamma
        self.rho = rho
        self.sigma_y = sigma_y
        self.n_a = n_a
        self.a_min = a_min
        self.a_max = a_max
        
        # Discretize income process
        self.y_grid, self.P_y = tauchen_discretization(rho, sigma_y, n_points=9)
        self.n_y = len(self.y_grid)
        self.y_grid = np.exp(self.y_grid)  # Convert to levels
        
        # Asset grid (log-spaced for refinement near zero)
        self.a_grid = np.logspace(np.log10(a_min), np.log10(a_max), n_a)
        
        # Initialize value and policy functions
        self.V = np.zeros((n_a, self.n_y))
        self.policy_a = np.zeros((n_a, self.n_y))
        self.policy_c = np.zeros((n_a, self.n_y))
    
    def solve(self, tol=1e-5, max_iter=500, verbose=True):
        """Solve using value function iteration with a discrete grid search instead
        of continuous optimization.  This avoids slow minimize_scalar calls and
        guarantees termination even when the surface is irregular.
        """
        
        V_old = self.V.copy()
        converged = False
        
        # pre‑compute continuation interpolation once per iteration
        for iteration in range(max_iter):
            V_interp = interp1d(self.a_grid, self.V[:, :], kind='linear',
                                axis=0, bounds_error=False, fill_value='extrapolate')
            
            for i_y, y in enumerate(self.y_grid):
                for i_a, a in enumerate(self.a_grid):
                    # resources available next period
                    resource = (1 + self.r) * a + y
                    if resource <= 0:
                        # cannot consume
                        self.V[i_a, i_y] = -1e10
                        self.policy_c[i_a, i_y] = 0.0
                        self.policy_a[i_a, i_y] = 0.0
                        continue
                    
                    # candidate next‑period assets are drawn from grid
                    # ensure a' <= resources (c >=0)
                    feasible_indices = self.a_grid <= resource
                    a_candidates = self.a_grid[feasible_indices]
                    c_candidates = resource - a_candidates
                    
                    # compute utility and continuation for each candidate
                    util_vals = ces_utility(c_candidates, self.gamma)
                    
                    # Vectorized continuation value calculation
                    V_next_all = V_interp(a_candidates)  # Shape: (n_candidates, n_y)
                    cont_vals = V_next_all @ self.P_y[i_y, :]  # Matrix multiplication
                    
                    total_vals = util_vals + self.beta * cont_vals
                    best_idx = np.argmax(total_vals)
                    
                    self.V[i_a, i_y] = total_vals[best_idx]
                    self.policy_c[i_a, i_y] = c_candidates[best_idx]
                    self.policy_a[i_a, i_y] = a_candidates[best_idx]
            
            # check convergence
            diff = np.max(np.abs(self.V - V_old))
            if diff < tol:
                converged = True
                break
            V_old = self.V.copy()
        
        return {
            'converged': converged,
            'iterations': iteration + 1,
            'final_diff': diff
        }
    
    def simulate(self, T=1000, initial_a=1.0, random_seed=42):
        """Simulate the model forward"""
        np.random.seed(random_seed)
        
        # Initialize
        a_path = np.zeros(T + 1)
        c_path = np.zeros(T)
        y_path = np.zeros(T)
        
        a_path[0] = initial_a
        
        # Draw initial income state
        y_idx = np.random.choice(self.n_y)
        
        # Interpolation function for policy
        policy_interp = interp1d(self.a_grid, self.policy_c, kind='linear',
                                axis=0, bounds_error=False, fill_value='extrapolate')
        
        for t in range(T):
            y = self.y_grid[y_idx]
            y_path[t] = y
            
            # Interpolate consumption policy
            c = policy_interp(a_path[t])[y_idx]
            c = np.maximum(c, 1e-6)
            c_path[t] = c
            
            # Next period assets
            a_path[t + 1] = (1 + self.r) * a_path[t] + y - c
            a_path[t + 1] = np.maximum(a_path[t + 1], 0)
            
            # Income transition
            y_idx = np.random.choice(self.n_y, p=self.P_y[y_idx, :])
        
        return {
            'c': c_path,
            'a': a_path,
            'y': y_path
        }


# ============================================================================
# MODEL 2: ROBINSON CRUSOE PRODUCTION ECONOMY
# ============================================================================

class RobinsonCrusoeModel:
    """
    Robinson Crusoe Production Economy
    
    Agent maximizes: E_0 sum_{t=0}^∞ β^t U(c_t)
    Subject to: c_t + k_{t+1} = z_t * k_t^α + (1-δ)k_t
                k_{t+1} ≥ 0
                log(z_t) = rho * log(z_{t-1}) + σ_z ε_t
    
    Parameters:
    -----------
    beta : float, discount factor
    alpha : float, capital share (0 to 1)
    delta : float, depreciation rate
    gamma : float, risk aversion
    rho : float, AR(1) persistence
    sigma_z : float, TFP shock std dev
    n_k : int, capital grid size
    """
    
    def __init__(self, beta, alpha, delta, gamma, rho, sigma_z, n_k=100):
        self.beta = beta
        self.alpha = alpha
        self.delta = delta
        self.gamma = gamma
        self.rho = rho
        self.sigma_z = sigma_z
        self.n_k = n_k
        
        # Discretize TFP process
        self.z_grid, self.P_z = tauchen_discretization(rho, sigma_z, n_points=9)
        self.n_z = len(self.z_grid)
        self.z_grid = np.exp(self.z_grid)  # Convert to levels
        
        # Capital grid
        k_min = 0.1
        k_max = 50
        self.k_grid = np.logspace(np.log10(k_min), np.log10(k_max), n_k)
        
        # Value and policy functions
        self.V = np.zeros((n_k, self.n_z))
        self.policy_k = np.zeros((n_k, self.n_z))
        self.policy_c = np.zeros((n_k, self.n_z))
    
    def production_function(self, k, z):
        """Output: Y = z * k^α"""
        return z * (k ** self.alpha)
    
    def solve(self, tol=1e-5, max_iter=500, verbose=True):
        """Solve using VFI with discrete grid search for k'"""
        
        V_old = self.V.copy()
        
        for iteration in range(max_iter):
            V_interp = interp1d(self.k_grid, self.V[:, :], kind='linear',
                                axis=0, bounds_error=False, fill_value='extrapolate')
            
            for i_z, z in enumerate(self.z_grid):
                for i_k, k in enumerate(self.k_grid):
                    output = self.production_function(k, z)
                    k_max_next = output + (1 - self.delta) * k
                    
                    # feasible k' on grid
                    feasible = self.k_grid <= k_max_next
                    k_candidates = self.k_grid[feasible]
                    c_candidates = k_max_next - k_candidates
                    
                    util_vals = ces_utility(c_candidates, self.gamma)
                    
                    # Vectorized continuation value calculation
                    V_next_all = V_interp(k_candidates)  # Shape: (n_candidates, n_z)
                    cont_vals = V_next_all @ self.P_z[i_z, :]  # Matrix multiplication
                    
                    total_vals = util_vals + self.beta * cont_vals
                    best = np.argmax(total_vals)
                    
                    k_opt = k_candidates[best]
                    c_opt = c_candidates[best]
                    
                    self.V[i_k, i_z] = total_vals[best]
                    self.policy_k[i_k, i_z] = k_opt
                    self.policy_c[i_k, i_z] = c_opt
            
            diff = np.max(np.abs(self.V - V_old))
            if diff < tol:
                return {'converged': True, 'iterations': iteration+1, 'final_diff': diff}
            V_old = self.V.copy()
        
        return {'converged': False, 'iterations': max_iter, 'final_diff': diff}

    
    def simulate(self, T=1000, initial_k=1.0, random_seed=42):
        """Simulate the economy"""
        np.random.seed(random_seed)
        
        k_path = np.zeros(T + 1)
        c_path = np.zeros(T)
        output_path = np.zeros(T)
        investment_path = np.zeros(T)
        z_path = np.zeros(T)
        
        k_path[0] = initial_k
        z_idx = np.random.choice(self.n_z)
        
        policy_interp = interp1d(self.k_grid, self.policy_c, kind='linear',
                                axis=0, bounds_error=False, fill_value='extrapolate')
        
        for t in range(T):
            z = self.z_grid[z_idx]
            z_path[t] = z
            
            output = self.production_function(k_path[t], z)
            output_path[t] = output
            
            c = policy_interp(k_path[t])[z_idx]
            c = np.maximum(c, 1e-6)
            c_path[t] = c
            
            k_next = output + (1 - self.delta) * k_path[t] - c
            k_next = np.maximum(k_next, 0.01)
            k_path[t + 1] = k_next
            
            investment_path[t] = k_next - (1 - self.delta) * k_path[t]
            
            z_idx = np.random.choice(self.n_z, p=self.P_z[z_idx, :])
        
        return {
            'c': c_path,
            'k': k_path,
            'output': output_path,
            'investment': investment_path,
            'z': z_path
        }


# ============================================================================
# MODEL 3: ENDOGENOUS LABOR SUPPLY
# ============================================================================

class LaborSupplyModel:
    """
    Endogenous Labor Supply Model
    
    Agent maximizes: E_0 sum_{t=0}^∞ β^t [U(c_t) + χ * l_t^(1+1/η)/(1+1/η)]
    Subject to: a_{t+1} + c_t = (1+r)a_t + w_t * l_t
                0 ≤ l_t ≤ 1
                a_{t+1} ≥ 0
                log(w_t) = rho * log(w_{t-1}) + σ_w ε_t
    
    Parameters:
    -----------
    beta : float, discount factor
    r : float, interest rate
    gamma : float, consumption risk aversion
    chi : float, labor disutility parameter
    eta : float, Frisch elasticity
    rho : float, wage persistence
    sigma_w : float, wage shock std dev
    n_a : int, asset grid size
    """
    
    def __init__(self, beta, r, gamma, chi, eta, rho, sigma_w, n_a=80):
        self.beta = beta
        self.r = r
        self.gamma = gamma
        self.chi = chi
        self.eta = eta
        self.rho = rho
        self.sigma_w = sigma_w
        self.n_a = n_a
        
        # Discretize wage process
        self.w_grid, self.P_w = tauchen_discretization(rho, sigma_w, n_points=7)
        self.n_w = len(self.w_grid)
        self.w_grid = np.exp(self.w_grid)
        
        # Asset grid
        self.a_grid = np.logspace(np.log10(0.01), np.log10(50), n_a)
        
        # Value and policies
        self.V = np.zeros((n_a, self.n_w))
        self.policy_a = np.zeros((n_a, self.n_w))
        self.policy_l = np.zeros((n_a, self.n_w))
        self.policy_c = np.zeros((n_a, self.n_w))
    
    def solve(self, tol=1e-5, max_iter=500, verbose=True):
        """Solve with VFI"""
        
        V_old = self.V.copy()
        
        for iteration in range(max_iter):
            for i_w, w in enumerate(self.w_grid):
                for i_a, a in enumerate(self.a_grid):
                    # Budget constraint: c = (1+r)*a + w*l - a'
                    
                    V_interp = interp1d(self.a_grid, self.V[:, :], kind='linear',
                                       axis=0, bounds_error=False, fill_value='extrapolate')
                    
                    best_val = -np.inf
                    best_l = 0.5
                    best_a_next = a * 0.9
                    
                    # Reduced grid search for efficiency
                    l_grid = np.linspace(0, 1, 15)
                    a_next_grid = np.linspace(0, (1 + self.r) * a + w, 20)
                    
                    for l in l_grid:
                        for a_next in a_next_grid:
                            c = (1 + self.r) * a + w * l - a_next
                            
                            if c <= 1e-6:
                                continue
                            
                            # Continuation value
                            V_next = V_interp(a_next)
                            EV = self.P_w[i_w, :] @ V_next
                            
                            util = ces_utility(c, self.gamma) + labor_utility(l, self.chi, self.eta)
                            total_val = util + self.beta * EV
                            
                            if total_val > best_val:
                                best_val = total_val
                                best_l = l
                                best_a_next = a_next
                    
                    # Store results
                    c_opt = (1 + self.r) * a + w * best_l - best_a_next
                    V_next = V_interp(best_a_next)
                    EV = self.P_w[i_w, :] @ V_next
                    
                    self.V[i_a, i_w] = ces_utility(c_opt, self.gamma) + \
                                       labor_utility(best_l, self.chi, self.eta) + self.beta * EV
                    self.policy_a[i_a, i_w] = best_a_next
                    self.policy_l[i_a, i_w] = best_l
                    self.policy_c[i_a, i_w] = c_opt
            
            # Convergence
            diff = np.max(np.abs(self.V - V_old))
            if diff < tol:
                return {
                    'converged': True,
                    'iterations': iteration + 1,
                    'final_diff': diff
                }
            V_old = self.V.copy()
        
        return {
            'converged': False,
            'iterations': max_iter,
            'final_diff': diff
        }
    
    def simulate(self, T=1000, initial_a=1.0, random_seed=42):
        """Simulate labor supply decisions"""
        np.random.seed(random_seed)
        
        a_path = np.zeros(T + 1)
        c_path = np.zeros(T)
        l_path = np.zeros(T)
        y_path = np.zeros(T)
        w_path = np.zeros(T)
        
        a_path[0] = initial_a
        w_idx = np.random.choice(self.n_w)
        
        c_interp = interp1d(self.a_grid, self.policy_c, kind='linear',
                           axis=0, bounds_error=False, fill_value='extrapolate')
        l_interp = interp1d(self.a_grid, self.policy_l, kind='linear',
                           axis=0, bounds_error=False, fill_value='extrapolate')
        a_interp = interp1d(self.a_grid, self.policy_a, kind='linear',
                           axis=0, bounds_error=False, fill_value='extrapolate')
        
        for t in range(T):
            w = self.w_grid[w_idx]
            w_path[t] = w
            
            c = c_interp(a_path[t])[w_idx]
            l = l_interp(a_path[t])[w_idx]
            a_next = a_interp(a_path[t])[w_idx]
            
            c = np.maximum(c, 1e-6)
            l = np.clip(l, 0, 1)
            a_next = np.maximum(a_next, 0)
            
            c_path[t] = c
            l_path[t] = l
            y_path[t] = w * l
            a_path[t + 1] = a_next
            
            w_idx = np.random.choice(self.n_w, p=self.P_w[w_idx, :])
        
        return {
            'c': c_path,
            'l': l_path,
            'a': a_path,
            'y': y_path,
            'w': w_path
        }
