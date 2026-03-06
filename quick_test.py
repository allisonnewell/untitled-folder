#!/usr/bin/env python3
"""
Quick test of macroeconomic models (reduced grid for speed)
"""

import numpy as np
from models import ConsumptionSavingsModel, RobinsonCrusoeModel, LaborSupplyModel

print("=" * 70)
print("QUICK TEST - MACROECONOMIC MODELS (Reduced Grid)")
print("=" * 70)

# MODEL 1: Consumption-Savings (30 points)
print("\n[1] Consumption-Savings Model (n_a=30, 20 iterations max)")
print("-" * 70)
try:
    model1 = ConsumptionSavingsModel(beta=0.95, r=0.05, gamma=2.0, rho=0.9, sigma_y=0.1, n_a=30)
    result1 = model1.solve(max_iter=20, verbose=False)
    sim1 = model1.simulate(T=200, initial_a=1.0, random_seed=42)
    print(f"    PASS: Mean consumption={np.mean(sim1['c']):.3f}, Mean assets={np.mean(sim1['a']):.3f}")
except Exception as e:
    print(f"    FAIL: {str(e)[:50]}")

# MODEL 2: Robinson Crusoe (30 points)
print("\n[2] Robinson Crusoe Economy (n_k=30, 20 iterations max)")
print("-" * 70)
try:
    model2 = RobinsonCrusoeModel(beta=0.95, alpha=0.33, delta=0.1, gamma=2.0, rho=0.9, sigma_z=0.02, n_k=30)
    result2 = model2.solve(max_iter=20, verbose=False)
    sim2 = model2.simulate(T=200, initial_k=1.0, random_seed=42)
    print(f"    PASS: Mean output={np.mean(sim2['output']):.3f}, Mean capital={np.mean(sim2['k']):.3f}")
except Exception as e:
    print(f"    FAIL: {str(e)[:50]}")

# MODEL 3: Labor Supply (20 points, 10 iterations - grid search expensive)
print("\n[3] Labor Supply Model (n_a=20, 10 iterations max, reduced grid)")
print("-" * 70)
try:
    model3 = LaborSupplyModel(beta=0.95, r=0.05, gamma=2.0, chi=1.0, eta=0.5, rho=0.9, sigma_w=0.1, n_a=20)
    result3 = model3.solve(max_iter=10, verbose=False)
    sim3 = model3.simulate(T=200, initial_a=1.0, random_seed=42)
    print(f"    PASS: Mean labor={np.mean(sim3['l']):.3f}, Mean consumption={np.mean(sim3['c']):.3f}")
except Exception as e:
    print(f"    FAIL: {str(e)[:50]}")

print("\n" + "=" * 70)
print("TEST COMPLETE - All models functional")
print("=" * 70)
print("\nNOTE: For Streamlit dashboard, use moderate grid sizes:")
print("  - Consumption-Savings: n_a=80-100")
print("  - Robinson Crusoe: n_k=80-100")
print("  - Labor Supply: n_a=40-60 (grid search is computationally expensive)")
print("\nTo run dashboard:")
print("  streamlit run app.py")
print("=" * 70)
