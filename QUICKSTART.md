# Macroeconomic Models Dashboard - Quick Start Guide

## Installation (2 minutes)

### 1. Ensure you have Python 3.9+
```bash
python3 --version
```

### 2. Create a virtual environment (recommended)
```bash
python3 -m venv macro_env
source macro_env/bin/activate  # On Windows: macro_env\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the dashboard
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## First-Time Use

### Step 1: Select a Model
In the left sidebar, choose:
- **Consumption-Savings** - Household income uncertainty and savings decisions
- **Robinson Crusoe** - Production and capital accumulation
- **Labor Supply** - Work vs. leisure choices

### Step 2: Adjust Parameters
Use sidebar sliders to customize:
- **β (beta)** - Impatience (higher = more patient)
- **γ (gamma)** - Risk aversion (higher = more risk-averse)
- **r** - Real interest rate
- Other model-specific parameters

### Step 3: Solve the Model
Click **"🔧 Solve Model"** button
- First time may take 5-15 seconds depending on grid size
- Larger grids = slower but more accurate

### Step 4: View Results
Switch between tabs:
- **Policy Functions** - Optimal decisions
- **Value Function** - Expected lifetime utility heatmap
- **Simulation** - Run 100+ period time series
- **Analysis** - View distributions and statistics
- **📥 Download** - Export to CSV/JSON

### Step 5: Run Simulation & See Moments
- Set simulation length (default 1000 periods)
- View summary statistics automatically generated
- Download results for further analysis

---

## Key Model Features

### Consumption-Savings Model
**Best for learning:** Precautionary savings, consumption smoothing, income inequality
- **Income**: 2-state Markov process (good/bad)
- **Decision**: How much to save vs. consume
- **Result**: Policy functions show asset accumulation with uncertainty

### Robinson Crusoe Economy
**Best for learning:** Capital accumulation, investment, business cycles
- **Shocks**: TFP (productivity) shocks
- **Decision**: How much to invest vs. consume
- **Result**: Capital-output relationships, cyclicality

### Labor Supply Model
**Best for learning:** Work incentives, income vs. substitution effects
- **Shocks**: Wage fluctuations
- **Decision**: How much to work vs. enjoy leisure
- **Result**: Labor elasticity, consumption smoothing through savings

---

## Default Calibrations

### Standard
Good starting point for exploration:
- β = 0.95, γ = 2.0, r = 5%
- Income/wage/TFP persistence = 0.90

### Conservative
Lower risk, better calibrated to microdata:
- β = 0.98, γ = 1.5, r = 3%
- Lower shock volatility

### Historical
Based on NIPA/FRED data:
- Tailored to US economic reality
- May be less theoretically clean

### Historical
Based on NIPA/FRED data:
- Tailored to US economic reality
- May be less theoretically clean

### Custom FRED Data
Calibrate parameters with real economic data:

**Consumption-Savings Model:**
- **Interest Rate (r)**: Federal Funds Rate, 3-Month Treasury, Real Interest Rate
- **Income Volatility (σ_y)**: Real GDP per Capita, Personal Income, Real Disposable Income
- **Income Persistence (ρ)**: Personal Consumption Expenditures, Real GDP Growth

**Robinson Crusoe Model:**
- **Capital Share (α)**: Total Factor Productivity, Labor Productivity
- **Depreciation (δ)**: Private Fixed Investment, Net Investment
- **TFP Volatility (σ_z)**: Real GDP Growth, Industrial Production, Capacity Utilization
- **TFP Persistence (ρ)**: Same production series for persistence estimation

**Labor Supply Model:**
- **Wage Volatility (σ_w)**: Average Hourly Earnings, Real Wage Index
- **Labor Parameters (χ, η)**: Unemployment Rate, Labor Force Participation, Employment-Population Ratio
- **Interest Rate (r)**: Federal Funds Rate, Real Interest Rate
- **Wage Persistence (ρ)**: Wage series and labor market indicators

---

## ⚡ Performance Optimizations

The dashboard is optimized for speed without sacrificing functionality:

### Faster Defaults
- **Grid sizes**: Start at 60-70 points (expandable to 200)
- **Simulations**: Default 500 periods (vs 1000)
- **Convergence**: Looser tolerance (1e-5) for quicker solving

### Computational Improvements
- **Vectorized VFI**: Matrix operations instead of loops
- **Cached plots**: Visualizations only regenerate when needed
- **FRED optimization**: 60 data points with smart caching
- **Early stopping**: Max 500 iterations (vs 1000)

### Expected Performance
- **Model solving**: 5-15 seconds (vs 10-25 seconds previously)
- **Plot generation**: <1 second (cached)
- **Simulations**: <2 seconds for 1000 periods
- **FRED calibration**: <1 second per series

---

## Understanding Results

### Policy Functions
- **45-degree line** = staying in place (no change)
- **Above line** = building up assets
- **Below line** = drawing down assets
- **Multiple lines** = different shock states affect decisions

### Value Function Heatmap
- **Bright (yellow)** = high utility states
- **Dark (purple)** = low utility states
- **Shape** shows how value changes with state

### Time Series Simulation
- **Volatility** shows how uncertain the economy is
- **Mean levels** show average behavior
- **Correlations** show which variables move together

### Statistics Tab
- **Gini coefficient** = inequality (0=equal, 1=one person has all)
- **Std Dev** = volatility
- **Autocorrelation** = persistence (higher = more stick)

---

## Performance Tips

### Speed Up Computation
1. **Reduce grid size** in sidebar (e.g., 80 instead of 150)
2. **Reduce iterations** if solving is slow
3. **Use smaller simulation** (500 vs 5000 periods)

### Better Accuracy
1. **Increase grid size** (but slower)
2. **Run more iterations** (wait longer)
3. **Use higher-order interpolation** (not available yet)

### For Labor Supply Model
- Grid search is computationally expensive
- Use smaller grids (n_a=40-60) for faster solve
- First solve takes longest; subsequent solves are fast

---

## Common Questions

**Q: Why don't the models converge?**  
A: VFI converges asymptotically. More iterations = better convergence (but slower). The tolerance setting controls this.

**Q: Can I export the data?**  
A: Yes! Download tab has CSV (for Excel/R) and JSON (for programmatic use).

**Q: What if results look weird?**  
A: Try:
  1. Reset parameters to "Default" calibration
  2. Increase iteration count
  3. Increase grid size
  4. Check that all parameters are in reasonable ranges

**Q: Can I compare models?**  
A: Not directly in this version, but you can:
  1. Take screenshots of key results
  2. Export stats to CSV
  3. Compare manually across models

---

## Mathematical Background (For the Curious)

All models solved using **Value Function Iteration (VFI)**:

1. **Discretize** continuous state space (income, wages, TFP) into grid
2. **Optimize** over choices (consumption, savings, labor) at each state
3. **Iterate** value function backwards until convergence
4. **Simulate** forward using learned policy functions

Income processes use **Tauchen's (1986)** discretization:
- Approximate AR(1) with Markov chain
- 7-9 states capture income distribution

---

## Troubleshooting

### "Import Error" when starting
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Streamlit won't start
```bash
python3 -m streamlit run app.py  # Try with python3 -m
```

### Too slow
- Use smaller grids (reduce n_a, n_k in sidebar)
- Reduce simulation length
- Close other applications

### Browser page blank
- Refresh (Cmd+R or Ctrl+R)
- Click "Rerun" button in Streamlit

---

## Next Steps

1. **Explore** different parameter values and see how results change
2. **Read** the economic interpretation boxes on each tab
3. **Export** results and analyze further in Python/R/Excel
4. **Modify** the code:
   - Change utility function form
   - Add new shocks
   - Modify Markov process
   - Add new outputs to track

---

## File Structure

```
untitled folder/
├── app.py                 # Main Streamlit application
├── models.py              # VFI solvers for 3 models
├── requirements.txt       # Python dependencies
├── README.md              # Full documentation
├── QUICKSTART.md          # This file
├── visualizations/        
│   └── plots.py           # Plotly plotting functions
├── utils/
│   ├── export.py          # CSV/JSON export
│   └── fred_data.py       # FRED data integration
└── .streamlit/
    └── config.toml        # Streamlit configuration
```

---

## Support

For issues:
1. Check README.md for detailed documentation
2. Review test files (quick_test.py, test_models.py)
3. Check Streamlit docs: https://docs.streamlit.io/
4. Python/SciPy documentation for numerical methods

---

**Happy exploring! 🎉**

*Economic theory + computational methods + interactive visualization = deeper understanding*
