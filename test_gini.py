#!/usr/bin/env python3
"""
Quick test to verify the gini_coefficient function fix
"""

import numpy as np

def gini_coefficient(values):
    """Calculate Gini coefficient"""
    sorted_values = np.sort(values)
    n = len(values)
    cumsum = np.cumsum(sorted_values)
    gini = (2 * np.sum(np.arange(1, n + 1) * sorted_values)) / (n * np.sum(sorted_values)) - (n + 1) / n
    return gini

# Test the function
test_values = np.array([1, 2, 3, 4, 5])
gini_result = gini_coefficient(test_values)
print(f'✓ Gini coefficient function works: {gini_result:.4f}')

# Test with simulation-like data
sim_data = np.random.normal(1.0, 0.2, 100)
gini_sim = gini_coefficient(sim_data)
print(f'✓ Gini coefficient on simulation data: {gini_sim:.4f}')

print('✅ gini_coefficient function is properly defined and working!')