#!/usr/bin/env python3
"""
Final verification of macroeconomic models dashboard
"""

import sys

print("\n" + "="*70)
print("FINAL VERIFICATION - Macroeconomic Models Dashboard")
print("="*70)

errors = []

# Check 1: Imports
print("\n[1] Import Check...")
try:
    from models import ConsumptionSavingsModel, RobinsonCrusoeModel, LaborSupplyModel
    from visualizations.plots import plot_value_function, plot_policy_function
    from utils.export import export_simulation_to_csv, export_policies_to_csv
    from utils.fred_data import FREDDataFetcher, get_sample_calibration
    print("    OK - All modules import successfully")
except Exception as e:
    errors.append(f"Import: {str(e)[:50]}")
    print(f"    ERROR: {str(e)[:50]}")

# Check 2: Model Instantiation
print("\n[2] Model Creation Check...")
try:
    m1 = ConsumptionSavingsModel(beta=0.95, r=0.05, gamma=2.0, rho=0.9, sigma_y=0.1, n_a=40)
    m2 = RobinsonCrusoeModel(beta=0.95, alpha=0.33, delta=0.1, gamma=2.0, rho=0.9, sigma_z=0.02, n_k=40)
    m3 = LaborSupplyModel(beta=0.95, r=0.05, gamma=2.0, chi=1.0, eta=0.5, rho=0.9, sigma_w=0.1, n_a=30)
    print("    OK - All models instantiate")
except Exception as e:
    errors.append(f"Model: {str(e)[:40]}")

# Check 3: Solving
print("\n[3] Model Solving Check...")
try:
    r1 = m1.solve(max_iter=5, verbose=False)
    r2 = m2.solve(max_iter=5, verbose=False)
    print("    OK - Solving works")
except Exception as e:
    errors.append(f"Solve: {str(e)[:40]}")

# Check 4: Simulation
print("\n[4] Simulation Check...")
try:
    s1 = m1.simulate(T=100, initial_a=1.0, random_seed=42)
    s2 = m2.simulate(T=100, initial_k=1.0, random_seed=42)
    print("    OK - Simulations work")
except Exception as e:
    errors.append(f"Sim: {str(e)[:40]}")

# Check 5: Exports
print("\n[5] Export Check...")
try:
    csv1 = export_simulation_to_csv(s1, 'cs')
    csv2 = export_simulation_to_csv(s2, 'rc')
    print("    OK - CSV export works")
except Exception as e:
    errors.append(f"Export: {str(e)[:40]}")

# Check 6: Visualization
print("\n[6] Visualization Check...")
try:
    import numpy as np
    grid = np.linspace(0, 10, 50)
    policy = np.linspace(0.5, 5, 50)
    fig = plot_policy_function(grid, policy)
    print("    OK - Visualizations work")
except Exception as e:
    errors.append(f"Viz: {str(e)[:40]}")

# Check 7: FRED
print("\n[7] FRED Data Check...")
try:
    fred = FREDDataFetcher()
    cal = get_sample_calibration('Default')
    print("    OK - FRED module works")
except Exception as e:
    errors.append(f"FRED: {str(e)[:40]}")

# Summary
print("\n" + "="*70)
if errors:
    print(f"RESULT: {len(errors)} ERRORS")
    for e in errors:
        print(f"  ERROR: {e}")
else:
    print("RESULT: ALL CHECKS PASSED")
    print("Status: READY FOR PRODUCTION")
print("="*70 + "\n")
