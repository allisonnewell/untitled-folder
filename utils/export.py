"""
Utility functions for export and data handling
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime


def export_simulation_to_csv(simulation_dict, model_name):
    """
    Export simulation results to CSV format
    
    Parameters:
    -----------
    simulation_dict : dict, output from model.simulate()
    model_name : str, 'cs', 'rc', or 'ls'
    
    Returns:
    --------
    str : CSV-formatted string
    """
    
    df_dict = {}
    
    if model_name == 'cs':
        df_dict = {
            'Time': np.arange(len(simulation_dict['c'])),
            'Consumption': simulation_dict['c'],
            'Assets': simulation_dict['a'][:-1],
            'Income': simulation_dict['y']
        }
    elif model_name == 'rc':
        df_dict = {
            'Time': np.arange(len(simulation_dict['c'])),
            'Output': simulation_dict['output'],
            'Consumption': simulation_dict['c'],
            'Investment': simulation_dict['investment'],
            'Capital': simulation_dict['k'][:-1],
            'TFP': simulation_dict['z']
        }
    elif model_name == 'ls':
        df_dict = {
            'Time': np.arange(len(simulation_dict['c'])),
            'Consumption': simulation_dict['c'],
            'Labor': simulation_dict['l'],
            'Labor_Income': simulation_dict['y'],
            'Assets': simulation_dict['a'][:-1],
            'Wage': simulation_dict['w']
        }
    
    df = pd.DataFrame(df_dict)
    return df.to_csv(index=False)


def export_policies_to_csv(state_grid, policy_dict, model_name):
    """
    Export policy functions to CSV
    
    Parameters:
    -----------
    state_grid : array, state grid points
    policy_dict : dict, {policy_name: policy_array}
    model_name : str
    
    Returns:
    --------
    str : CSV-formatted string
    """
    
    df_dict = {'State': state_grid}
    
    for policy_name, policy_array in policy_dict.items():
        if policy_array.ndim == 1:
            df_dict[policy_name] = policy_array
        else:
            # Multiple shock states
            for i in range(policy_array.shape[1]):
                df_dict[f'{policy_name}_state_{i+1}'] = policy_array[:, i]
    
    df = pd.DataFrame(df_dict)
    return df.to_csv(index=False)


def create_model_summary_json(model, result, simulation_dict=None):
    """
    Create JSON summary of model results
    
    Parameters:
    -----------
    model : Model class instance
    result : dict, solver result
    simulation_dict : dict, optional simulation results
    
    Returns:
    --------
    str : JSON-formatted string
    """
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'solver_status': {
            'converged': bool(result['converged']),
            'iterations': int(result['iterations']),
            'final_diff': float(result['final_diff'])
        },
        'model_parameters': {}
    }
    
    # Model-specific parameters
    if hasattr(model, 'beta'):
        summary['model_parameters']['beta'] = float(model.beta)
    if hasattr(model, 'r'):
        summary['model_parameters']['r'] = float(model.r)
    if hasattr(model, 'gamma'):
        summary['model_parameters']['gamma'] = float(model.gamma)
    if hasattr(model, 'rho'):
        summary['model_parameters']['rho'] = float(model.rho)
    if hasattr(model, 'sigma_y'):
        summary['model_parameters']['sigma_y'] = float(model.sigma_y)
    if hasattr(model, 'sigma_z'):
        summary['model_parameters']['sigma_z'] = float(model.sigma_z)
    if hasattr(model, 'sigma_w'):
        summary['model_parameters']['sigma_w'] = float(model.sigma_w)
    if hasattr(model, 'alpha'):
        summary['model_parameters']['alpha'] = float(model.alpha)
    if hasattr(model, 'delta'):
        summary['model_parameters']['delta'] = float(model.delta)
    if hasattr(model, 'chi'):
        summary['model_parameters']['chi'] = float(model.chi)
    if hasattr(model, 'eta'):
        summary['model_parameters']['eta'] = float(model.eta)
    
    # Simulation statistics
    if simulation_dict is not None:
        summary['simulation_statistics'] = {}
        
        for key, data in simulation_dict.items():
            if isinstance(data, np.ndarray):
                summary['simulation_statistics'][key] = {
                    'mean': float(np.mean(data)),
                    'std': float(np.std(data)),
                    'min': float(np.min(data)),
                    'max': float(np.max(data)),
                    'median': float(np.median(data))
                }
    
    return json.dumps(summary, indent=2)


def calculate_summary_statistics(simulation_dict):
    """
    Calculate detailed statistics from simulation
    
    Returns dict with moments and correlations
    """
    
    stats = {}
    
    for key, data in simulation_dict.items():
        if isinstance(data, np.ndarray) and len(data) > 1:
            stats[key] = {
                'mean': float(np.mean(data)),
                'variance': float(np.var(data)),
                'std': float(np.std(data)),
                'min': float(np.min(data)),
                'max': float(np.max(data)),
                'median': float(np.median(data)),
                'autocorr_1': float(np.corrcoef(data[:-1], data[1:])[0, 1])
            }
    
    return stats
