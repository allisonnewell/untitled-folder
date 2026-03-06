# Macroeconomic Models Dashboard

An interactive Streamlit application featuring three canonical macroeconomic models solved using Value Function Iteration (VFI).

## Features

### Three Models Implemented

#### Model 1: Stochastic Consumption-Savings
- **Theory**: Household maximizes lifetime utility subject to borrowing constraint
- **Utility**: CES preferences - $U(c_t) = \frac{c_t^{1-\gamma} - 1}{1-\gamma}$
- **Shocks**: AR(1) income process with two-state Markov discretization
- **Solutions**: Optimal consumption and savings policies
- **Moments**: Mean, variance, autocorrelation of consumption and assets

#### Model 2: Robinson Crusoe Production Economy
- **Theory**: Agent produces, saves as capital, and consumes
- **Production**: $Y_t = z_t K_t^\alpha$
- **Shocks**: TFP (z_t) follows AR(1) process
- **Capital**: With depreciation - $K_{t+1} = Y_t + (1-\delta)K_t - C_t$
- **Solutions**: Capital accumulation and consumption policies
- **Analysis**: Input-output relationships, capital-output ratio

#### Model 3: Endogenous Labor Supply
- **Decision**: Optimize consumption (c) and labor supply (l) simultaneously
- **Budget**: $c_t + a_{t+1} = (1+r)a_t + w_t l_t$
- **Labor Preference**: Preferences over work vs leisure
- **Shocks**: Wage (w) follows AR(1) process
- **Elasticity**: Frisch elasticity parameter η controls labor supply responsiveness
- **Analysis**: Income vs substitution effects

## Dashboard Features

### Interactive Controls
- **Sidebar Parameter Sliders**: Adjust discount factor (β), risk aversion (γ), interest rate (r), shock persistence (ρ), shock volatility (σ)
- **Markov Chain Controls**: Modify transition probabilities
- **FRED Data Integration**: Optional - calibrate parameters with 22+ Federal Reserve economic indicators including interest rates, productivity, labor market data, investment, and inflation
- **Model Selection**: Toggle between three models seamlessly

### Visualizations
- **Policy Functions**: Optimal decisions across state space
- **Value Functions**: Heatmaps showing expected lifetime utility
- **Time Series**: Simulated paths of key economic variables
- **Distributions**: Histograms of consumption, assets, labor choices
- **Correlation Analysis**: Cross-variable relationships

### Analysis & Summaries
- **Automatic Moments**: Mean, variance, autocorrelation, Gini coefficient
- **Dynamic Text**: AI-generated economic interpretations based on live results
- **Steady-State Analysis**: Theoretical long-run values
- **Simulation Statistics**: Monte Carlo moments from simulated time series

### Export Capabilities
- **CSV Export**: Download policy functions and simulations
- **JSON Export**: Model summaries with all parameters and statistics
- **Charts**: Snapshot visualizations

## Design

### Academic Styling
- **Color Scheme**: Navy/dark background (#001a33) with bright cyan accents (#00d4ff)
- **Typography**: Garamond serif font throughout
- **Chart Boxes**: Crisp white backgrounds with subtle borders
- **Accessibility**: High contrast for readability

## Installation & Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Dashboard
```bash
streamlit run app.py
```

### 3. Interact with Models
1. **Select a model** from the sidebar radio buttons
2. **Adjust parameters** using the sidebar sliders
3. **Click "Solve Model"** to run VFI
4. **View results** in the main panel
5. **Run simulation** from the "Simulation" tab
6. **Download results** from the "Download" tab

## Deployment

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app.py
```

### GitHub Codespaces / Cloud Deployment
The app is configured to run automatically in cloud environments:

```bash
# Using the provided run script
./run.sh

# Or manually
PORT=8501 streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true
```

### Docker Deployment
```bash
# Build the image
docker build -t macro-models-dashboard .

# Run the container
docker run -p 8501:8501 macro-models-dashboard
```

### Environment Variables
- `PORT`: Server port (default: 8501)
- `STREAMLIT_SERVER_HEADLESS`: Run in headless mode (default: true in cloud)
- `GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN`: Auto-detected for Codespaces

### GitHub Actions
The repository includes a GitHub Actions workflow (`.github/workflows/deploy.yml`) that:
- Tests all imports and dependencies
- Verifies the app can start successfully
- Can be extended for full deployment pipelines

## Model Details

### Solving Process: Value Function Iteration

All three models are solved using **Value Function Iteration (VFI)**:

1. **Discretization**: Income/wage/TFP processes discretized using Tauchen's method
2. **Grid Construction**: Log-spaced grids for assets, capital, focusing refinement near lower bounds
3. **Bellman Equation**: Backward iteration from T→0
4. **Optimization**: Discrete grid search over feasible choices for each state (removed previous continuous optimizer to prevent slow/hanging evaluation)
5. **Convergence**: Iteration continues until policy functions stabilize (default tolerance: 1e-6)
6. **Interpolation**: Linear interpolation for continuous state space

### Stochastic Processes

Income (y), TFP (z), and wages (w) follow **AR(1) processes**:
$$x_{t+1} = \rho x_t + \sigma \varepsilon_{t+1}, \quad \varepsilon_t \sim N(0,1)$$

Discretization via Tauchen's method creates 7-9 state Markov chains with transition probabilities estimated from process parameters.

## Parameter Guidance

### Default Calibration
- **β = 0.95**: Standard annual discount factor
- **γ = 2.0**: Log-utility equivalent risk aversion
- **r = 0.05**: 5% real interest rate
- **ρ = 0.90**: Fairly persistent income shocks
- **σ_y = 0.10**: ±10% income volatility
- **ρ_z = 0.90**: persistent TFP shocks
- **σ_z = 0.02**: ±2% TFP volatility
- **α = 0.33**: Capital's share (matches US data)
- **δ = 0.10**: 10% annual depreciation
- **η = 0.5**: Unit elastic labor supply

### Conservative Calibration
Lower risk aversion, higher patience, less volatile shocks

### Historical Average Calibration
Estimates from NIPA, FRED, and household surveys

## Mathematical Foundations

### Consumption-Savings Bellman Equation
$$V(a, y) = \max_c \left\{ U(c) + \beta \mathbb{E}_y[V(a', y')] \right\}$$
subject to: $a' = (1+r)a + y - c$, $a' \geq 0$

### Robinson Crusoe Bellman Equation
$$V(k, z) = \max_c \left\{ U(c) + \beta \mathbb{E}_z[V(k', z')] \right\}$$
subject to: $k' = zk^\alpha + (1-\delta)k - c$, $k' \geq 0$

### Labor Supply Bellman Equation
$$V(a, w) = \max_{c,l} \left\{ U(c) - \chi\frac{l^{1+1/\eta}}{1+1/\eta} + \beta \mathbb{E}[V(a', w')] \right\}$$
subject to: $a' = (1+r)a + wl - c$, $0 \leq l \leq 1$, $a' \geq 0$

## Output Interpretation

### Policy Functions Show
- **Consumption Policy**: How much to consume given current state
- **Savings Policy**: How much to save/invest
- **Labor Policy**: Optimal hours worked given wages

### Heatmaps Reveal
- **Value Gradients**: Marginal value of more assets/better technology
- **Non-linearities**: Policy changes across state space
- **Shock Sensitivity**: How shocks affect household behavior

### Simulations Demonstrate
- **Precautionary Savings**: High uncertainty → more saving (C-S model)
- **Business Cycles**: Correlated output, consumption, investment (RC model)
- **Labor Elasticity**: How hours respond to wage fluctuations (LS model)

## Troubleshooting

### Slow Performance
- Reduce grid sizes (n_a, n_k) in sidebar
- Increase VFI iteration tolerance (less precision)
- Use smaller simulation length first

### Non-Convergence
-Ensure all parameters are in reasonable ranges
- Check that β < 1, 0 < r < 0.10, etc.
- Start with fewer grid points; refine after

### Import Errors
- Verify all files in correct directories:
  - `models.py` (root)
  - `visualizations/plots.py` (subdirectory)
  - `utils/export.py`, `utils/fred_data.py` (subdirectory)
- Run `pip install -r requirements.txt`

## Educational Use

This dashboard is designed for:
- **PhD/Advanced Macroeconomics**: Understanding VFI computation
- **Computational Methods**: Gauss-Legendre integration, discretization
- **Economic Theory**: Precautionary savings, labor supply, business cycles
- **Policy Analysis**: Impact of parameter changes on behavior

## References

1. Tauchen, G. (1986). "Finite State Markov-Chain Approximations to Univariate and Vector Autoregressions." *Economics Letters*, 20(2), 177-181.
2. Huggett, M. (1996). "Wealth Distribution in Life-Cycle Economies." *Journal of Monetary Economics*, 38(3), 469-494.
3. Krusell, P., & Smith, A. A. (1998). "Income and Wealth Heterogeneity in the Macroeconomy." *Journal of Political Economy*, 106(5), 867-896.
4. Ljungqvist, L., & Sargent, T. J. (2018). *Recursive Macroeconomic Theory* (4th ed.). MIT Press.

## License

Educational use - modify freely for classroom purposes.

---

**Dashboard Version**: 1.0  
**Last Updated**: 2026  
**Built with Streamlit, NumPy, SciPy, Plotly**
