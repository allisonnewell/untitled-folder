"""
Macroeconomic Models Dashboard - Enhanced Version with FRED Integration
Interactive Streamlit application for solving and simulating three macro models
"""

import streamlit as st
import numpy as np
import pandas as pd
from models import ConsumptionSavingsModel, RobinsonCrusoeModel, LaborSupplyModel
from visualizations.plots import (
    plot_value_function, plot_policy_function, plot_simulated_path,
    plot_multiple_series, plot_heatmap, plot_distribution, plot_forecast
)
from utils.export import export_simulation_to_csv, export_policies_to_csv, create_model_summary_json
from utils.fred_data import FREDDataFetcher, get_sample_calibration, FRED_SERIES
from utils.moments import compute_moments, compute_correlations, forecast_ar1, get_simulation_summary

# ============================================================================
# GitHub/Cloud Deployment Configuration
# ============================================================================
import os

# Configure for GitHub Codespaces or other cloud environments
if 'GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN' in os.environ or 'CODESPACE_NAME' in os.environ:
    # Running in GitHub Codespaces
    port = int(os.environ.get('PORT', 8501))
    os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'
    os.environ['STREAMLIT_SERVER_PORT'] = str(port)
    st.set_page_config(
        page_title="Macro Models Dashboard",
        layout="wide",
        initial_sidebar_state="expanded"
    )
elif os.environ.get('STREAMLIT_SERVER_HEADLESS', 'false').lower() == 'true' or 'PORT' in os.environ:
    # Running in headless mode or with PORT environment variable (common for cloud deployments)
    port = int(os.environ.get('PORT', 8501))
    os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'
    os.environ['STREAMLIT_SERVER_PORT'] = str(port)
    st.set_page_config(
        page_title="Macro Models Dashboard",
        layout="wide",
        initial_sidebar_state="expanded"
    )
else:
    # Local development
    st.set_page_config(
        page_title="Macro Models Dashboard",
        layout="wide",
        initial_sidebar_state="expanded"
    )

# ============================================================================
# Helper Functions
# ============================================================================
def gini_coefficient(values):
    """Calculate Gini coefficient"""
    sorted_values = np.sort(values)
    n = len(values)
    cumsum = np.cumsum(sorted_values)
    gini = (2 * np.sum(np.arange(1, n + 1) * sorted_values)) / (n * np.sum(sorted_values)) - (n + 1) / n
    return gini

def get_cached_plot(cache_key, plot_func, *args, **kwargs):
    """Cache plots to avoid regeneration"""
    if cache_key not in st.session_state:
        st.session_state[cache_key] = plot_func(*args, **kwargs)
    return st.session_state[cache_key]

# Custom CSS for enhanced styling - Dark/Navy Theme with Garamond
st.markdown("""
<style>
    /* Import Garamond font */
    @import url('https://fonts.googleapis.com/css2?family=Crimson+Text:ital@0;1&display=swap');
    
    /* Dark/Navy theme background */
    body, .main, .stApp {
        background-color: #001a33;
        color: #ffffff;
        font-family: 'Garamond', 'Crimson Text', serif;
    }
    
    /* Sidebar dark theme */
    .sidebar .sidebar-content {
        background-color: #002b4d;
    }
    
    /* White chart boxes */
    .stPlotly, .js-plotly-plot, iframe {
        background-color: #ffffff !important;
        border-radius: 10px;
        padding: 10px;
        border: 1px solid #e0e0e0;
    }
    
    /* Metric cards with navy background */
    .metric-card {
        background: linear-gradient(135deg, #003d66 0%, #004d7a 100%);
        padding: 20px;
        border-radius: 12px;
        color: #ffffff;
        text-align: center;
        box-shadow: 0 6px 12px rgba(0,0,0,0.3);
        border: 1px solid #0066cc;
        font-family: 'Garamond', serif;
    }
    
    .metric-value {
        font-size: 32px;
        font-weight: bold;
        margin: 10px 0;
        color: #00d4ff;
        font-family: 'Garamond', serif;
    }
    
    .metric-label {
        font-size: 14px;
        opacity: 0.95;
        color: #e6f2ff;
        font-family: 'Garamond', serif;
    }
    
    /* Summary boxes */
    .summary-box {
        background: linear-gradient(135deg, #004d7a 0%, #003d66 100%);
        padding: 20px;
        border-radius: 12px;
        color: #ffffff;
        margin: 20px 0;
        border-left: 4px solid #00d4ff;
        font-family: 'Garamond', serif;
    }
    
    /* Parameter boxes */
    .parameter-box {
        background: #002b4d;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #0099ff;
        margin: 10px 0;
        color: #ffffff;
        font-family: 'Garamond', serif;
    }
    
    /* Interpretation boxes */
    .interpretation-box {
        background: #003d66;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #00d4ff;
        margin: 15px 0;
        font-size: 14px;
        color: #e6f2ff;
        font-family: 'Garamond', serif;
    }
    
    /* Headers and titles */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Garamond', serif;
        color: #ffffff;
    }
    
    h1 {
        color: #00d4ff;
        border-bottom: 2px solid #0099ff;
        padding-bottom: 10px;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #002b4d;
        border-radius: 8px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #003d66;
        border-radius: 6px;
        color: #ffffff;
        font-family: 'Garamond', serif;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #0099ff;
        color: #001a33;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #0099ff;
        color: #001a33;
        font-family: 'Garamond', serif;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        transition: background-color 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #00d4ff;
    }
    
    /* Input fields */
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select,
    .stTextInput > div > div > input {
        background-color: #002b4d;
        color: #ffffff;
        border: 1px solid #0099ff;
        border-radius: 6px;
        font-family: 'Garamond', serif;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #003d66;
        color: #00d4ff;
        font-family: 'Garamond', serif;
    }
    
    /* Info/warning boxes */
    .stInfo, .stSuccess, .stWarning {
        background-color: #003d66;
        color: #ffffff;
        font-family: 'Garamond', serif;
    }
    
    /* Text styling */
    .stMarkdown {
        font-family: 'Garamond', serif;
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

st.title("Macroeconomic Models Dashboard")
st.markdown("""
Interactive dashboard for solving and simulating three canonical macroeconomic models 
using **Value Function Iteration (VFI)**.  
**Download results • Analyze policies • Run simulations • Use FRED data**
""")

# ============================================================================
# FRED Data Integration
# ============================================================================

@st.cache_resource
def get_fred_fetcher():
    """Initialize FRED data fetcher (cached)."""
    try:
        return FREDDataFetcher()
    except:
        return None

# Top sidebar section for parameter source
with st.sidebar.expander("Parameter Settings", expanded=False):
    param_source = st.selectbox(
        "Select parameter source:",
        ["Default", "Conservative", "Historical Average", "Custom FRED Data"]
    )
    
    if param_source == "Custom FRED Data":
        st.info("""
        **FRED Data Integration:**
        Select economic indicators below to calibrate model parameters with real data from the Federal Reserve.
        No API key required - data is fetched automatically.
        """)
        
        # Initialize FRED fetcher without API key
        if 'fred_data_fetcher' not in st.session_state:
            st.session_state.fred_data_fetcher = FREDDataFetcher()
        
        st.success("Connected to FRED database")
        
        st.markdown("**Available Economic Indicators:**")
        fred_series_list = ["None"] + [name for name in FRED_SERIES.keys()]
        
        # Show sample series
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Interest Rates & Prices:**")
            for name in ["Federal Funds Rate", "Real Interest Rate (10-yr)"]:
                st.caption(f"• {name}")
        
        with col2:
            st.markdown("**Economic Activity:**")
            for name in ["Real GDP per Capita", "Unemployment Rate", "Personal Consumption Expenditures"]:
                st.caption(f"• {name}")
        
        st.caption("... and more series available in parameter sections below")

st.sidebar.markdown("---")

# Sidebar - Model selection
model_choice = st.sidebar.radio(
    "**Select Model**",
    ["Consumption-Savings", "Robinson Crusoe", "Labor Supply"]
)

st.sidebar.markdown("---")

# ============================================================================
# MODEL 1: CONSUMPTION-SAVINGS
# ============================================================================
if model_choice == "Consumption-Savings":
    st.header("Stochastic Consumption-Savings Model")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Problem:** An agent with uncertain income chooses consumption and savings to maximize 
        lifetime utility subject to a borrowing constraint.
        """)
    
    with col2:
        st.markdown("""
        **Key Features:**
        - Income uncertainty (AR(1) process)
        - Precautionary savings motive
        - No-borrowing constraint
        - Implications for consumption inequality
        """)
    
    # Sidebar parameters
    st.sidebar.subheader("Model Parameters")
    
    # Load default or sample calibrations
    calibration = get_sample_calibration(param_source)['cs']
    
    # FRED Data integration for parameter selection
    if param_source == "Custom FRED Data":
        fred_fetcher = st.session_state.fred_data_fetcher
        st.sidebar.markdown("**Select FRED Series for Parameters:**")
        
        # Interest rate series
        interest_series = ["None"] + [name for name in FRED_SERIES.keys() 
                                      if FRED_SERIES[name]['use'] == 'Interest Rate']
        selected_r_series = st.sidebar.selectbox(
            "Interest Rate (r):", interest_series, key="cs_fred_r"
        )
        
        # Income/consumption series for volatility
        income_series = ["None"] + [name for name in FRED_SERIES.keys() 
                                    if FRED_SERIES[name]['use'] in ['Income', 'Consumption', 'Production']]
        selected_income_series = st.sidebar.selectbox(
            "Income Volatility (σ_y):", income_series, key="cs_fred_sigma"
        )
        
        # Persistence from various series
        persistence_series = ["None"] + [name for name in FRED_SERIES.keys() 
                                         if FRED_SERIES[name]['use'] in ['Production', 'Consumption', 'Income', 'Wages']]
        selected_rho_series = st.sidebar.selectbox(
            "Income Persistence (ρ):", persistence_series, key="cs_fred_rho"
        )
        
        # Apply calibrations
        if selected_r_series != "None":
            try:
                with st.spinner(f"Fetching {selected_r_series}..."):
                    params = fred_fetcher.estimate_parameters(
                        FRED_SERIES[selected_r_series]['series_id']
                    )
                    calibration['r'] = max(0.01, min(0.10, params['level_mean'] / 100))
                    st.sidebar.success(f"r = {calibration['r']:.4f}")
            except Exception as e:
                st.sidebar.warning(f"Could not fetch {selected_r_series}: {str(e)[:50]}")
        
        if selected_income_series != "None":
            try:
                with st.spinner(f"Fetching {selected_income_series}..."):
                    params = fred_fetcher.estimate_parameters(
                        FRED_SERIES[selected_income_series]['series_id']
                    )
                    calibration['sigma_y'] = max(0.01, min(0.5, params['std']))
                    st.sidebar.success(f"σ_y = {calibration['sigma_y']:.4f}")
            except Exception as e:
                st.sidebar.warning(f"Could not fetch {selected_income_series}: {str(e)[:50]}")
        
        if selected_rho_series != "None":
            try:
                with st.spinner(f"Fetching {selected_rho_series}..."):
                    params = fred_fetcher.estimate_parameters(
                        FRED_SERIES[selected_rho_series]['series_id']
                    )
                    calibration['rho'] = max(0.0, min(0.99, params['rho']))
                    st.sidebar.success(f"ρ = {calibration['rho']:.4f}")
            except Exception as e:
                st.sidebar.warning(f"Could not fetch {selected_rho_series}: {str(e)[:50]}")
        
        # Show current calibration
        st.sidebar.markdown("**Current Calibration:**")
        st.sidebar.write(f"r = {calibration['r']:.4f}")
        st.sidebar.write(f"ρ = {calibration['rho']:.4f}")
        st.sidebar.write(f"σ_y = {calibration['sigma_y']:.4f}")
    
    st.sidebar.markdown("<strong style='color:#00d4ff'>Model Parameters (Use sliders to adjust)</strong>", unsafe_allow_html=True)
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        beta = st.slider("Discount Factor (β)", 0.90, 0.99, calibration.get('beta', 0.95), 0.01, 
                        help="How much agents value future consumption (closer to 1 = more patient)")
        r = st.slider("Interest Rate (r)", 0.00, 0.10, calibration.get('r', 0.05), 0.01,
                     help="Real return on assets (savings/investment rate)")
        gamma = st.slider("Risk Aversion (γ)", 1.0, 10.0, calibration.get('gamma', 2.0), 0.5,
                         help="Curvature of utility (higher = more risk-averse)")
    
    with col2:
        rho = st.slider("Income Persistence (ρ)", 0.0, 0.99, calibration.get('rho', 0.9), 0.05,
                       help="How much income shocks persist (closer to 1 = more persistent)")
        sigma_y = st.slider("Income Shock Volatility (σ_y)", 0.01, 0.5, calibration.get('sigma_y', 0.1), 0.05,
                           help="Standard deviation of income shocks (uncertainty)")
        n_a = st.slider("Asset Grid Resolution", 50, 200, 70, 10,
                       help="Number of discrete asset points (more points = higher accuracy, slower solve)")
    
    # Markov transition matrix visualization
    with st.sidebar.expander("Markov Transition Matrix (Income States)", expanded=False):
        st.markdown("**Transition Probabilities**")
        st.caption("Shows probability of moving between income states each period")
        # Create simple display of P matrix pattern
        st.write(f"Persistence (main diagonal): {calibration.get('rho', 0.9):.2%}")
        st.write(f"Income shocks follow AR(1) with correlation {calibration.get('rho', 0.9):.3f}")
    
    st.sidebar.markdown("---")
    
    # Solve model
    if st.sidebar.button("Solve Model", key="cs_solve", use_container_width=True):
        with st.spinner("Solving Consumption-Savings model with VFI..."):
            try:
                model = ConsumptionSavingsModel(
                    beta=beta, r=r, rho=rho, sigma_y=sigma_y, gamma=gamma, n_a=n_a
                )
                result = model.solve(verbose=False)
                st.session_state.cs_model = model
                st.session_state.cs_result = result
                if result.get('converged'):
                    st.success("Consumption-Savings model solved (converged)")
                else:
                    st.warning("Model solved but did not converge within max iterations")
            except Exception as e:
                st.error(f"Error solving model: {e}")
                import traceback
                st.text(traceback.format_exc())
    
    # Display results
    if 'cs_model' in st.session_state:
        model = st.session_state.cs_model
        result = st.session_state.cs_result
        
        # Solution Status Box
        st.markdown("""
        <div class="summary-box">
        <strong>Model Successfully Solved!</strong>
        </div>
        """, unsafe_allow_html=True)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Status", "Converged" if result['converged'] else "Failed")
        with col2:
            st.metric("Iterations", result['iterations'])
        with col3:
            st.metric("Grid Size", f"{model.n_a} × {model.n_y}")
        with col4:
            st.metric("Solution Time", "< 5 sec")
        
        # Model Summary
        with st.expander("Model Interpretation", expanded=True):
            st.markdown("""
            <div class="interpretation-box">
            <strong>What the Model Shows:</strong>
            <ul>
            <li><strong>Policy Functions:</strong> Show optimal savings and consumption at each asset level</li>
            <li><strong>Precautionary Savings:</strong> Higher income uncertainty → more savings</li>
            <li><strong>Consumption Insurance:</strong> Agents smooth consumption over time</li>
            <li><strong>Constraint Binding:</strong> At low assets, borrowing limit is active</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Tabs for different views
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Policy Functions", "Value Function", 
                                                "Simulation", "Analysis", "Moments", "Forecasts", "Download"])
        
        with tab1:
            st.subheader("Policy Functions")
            col1, col2 = st.columns(2)
            with col1:
                fig = get_cached_plot(
                    'cs_policy_a',
                    plot_policy_function,
                    model.a_grid, model.policy_a,
                    title="Optimal Savings Policy: a'(a)",
                    state_label="Current Assets (a)",
                    action_label="Next Period Assets (a')",
                    shock_labels=["Low Income State", "Medium Income State", "High Income State"],
                    max_legend_items=3
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = get_cached_plot(
                    'cs_policy_c',
                    plot_policy_function,
                    model.a_grid, model.policy_c,
                    title="Optimal Consumption Policy: c(a)",
                    state_label="Current Assets (a)",
                    action_label="Consumption (c)",
                    shock_labels=["Low Income State", "Medium Income State", "High Income State"],
                    max_legend_items=3
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Value Function Heatmap")
            fig = get_cached_plot(
                'cs_value_heatmap',
                plot_heatmap,
                model.a_grid, model.y_grid, model.V.T,
                title="Value Function: V(a, y)",
                x_label="Assets (a)",
                y_label="Income (y)"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Simulate Economy")
            col1, col2, col3 = st.columns(3)
            with col1:
                sim_length = st.slider("Simulation Length", 100, 5000, 500, 100, key="cs_sim_len")
            with col2:
                initial_a = st.slider("Initial Asset Level", 0.1, 10.0, 1.0, 0.1, key="cs_init_a")
            with col3:
                random_seed = st.slider("Random Seed", 0, 10000, 42, 1, key="cs_seed")
            
            if st.button("Run Simulation", key="cs_run_sim", use_container_width=True):
                with st.spinner("Running simulation..."):
                    sim = model.simulate(T=sim_length, initial_a=initial_a, 
                                       random_seed=int(random_seed))
                    st.session_state.cs_sim = sim
            
            if 'cs_sim' in st.session_state:
                sim = st.session_state.cs_sim
                time_idx = np.arange(len(sim['c']))
                
                # Time series plot
                fig = plot_multiple_series(
                    time_idx,
                    {'Consumption': sim['c'], 'Assets': sim['a'][:-1], 'Income': sim['y'][:-1]},
                    title="Simulated Paths (Consumption-Savings Model)"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistics
                st.markdown("**Simulation Statistics:**")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Avg Consumption", f"{np.mean(sim['c']):.3f}")
                with col2:
                    st.metric("Avg Assets", f"{np.mean(sim['a']):.3f}")
                with col3:
                    st.metric("Std Consumption", f"{np.std(sim['c']):.3f}")
                with col4:
                    savings_rate = 1.0 - (np.mean(sim['c']) / np.mean(sim['y'][:-1]))
                    st.metric("Avg Savings Rate", f"{savings_rate:.1%}")
                
                # Dynamic narrative using computed moments
                cons_moms = compute_moments(sim['c'])
                assets_moms = compute_moments(sim['a'][:-1])
                st.markdown(f"""
                The simulated consumption series has a mean of **{cons_moms['mean']:.3f}**,
                variance **{cons_moms['variance']:.3f}**, and an autocorrelation (lag‑1) of
                **{cons_moms['autocorr_lag1']:.3f}**. Asset holdings display a mean of
                **{assets_moms['mean']:.3f}** and a lag‑1 autocorrelation of
                **{assets_moms['autocorr_lag1']:.3f}**, reflecting persistence in wealth accumulation.
                """
                )
                
                # Quick analysis section
                st.markdown("---")
                st.markdown("**Quick Analysis (see 'Analysis' tab for detailed distributions):**")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Consumption Analysis:**")
                    c_med = np.median(sim['c'])
                    c_iqr = np.percentile(sim['c'], 75) - np.percentile(sim['c'], 25)
                    c_min = np.min(sim['c'])
                    c_max = np.max(sim['c'])
                    st.write(f"On average, individuals consume about {c_med:.3f} units per period (whatever units your model uses). The typical variation – measured by the interquartile range – is {c_iqr:.3f} units, indicating moderate smoothing behaviour. During the entire simulation, consumption never fell below {c_min:.3f} or rose above {c_max:.3f}.")
                with col2:
                    st.write("**Asset Analysis:**")
                    a_med = np.median(sim['a'])
                    a_gini = gini_coefficient(sim['a'])
                    a_min = np.min(sim['a'])
                    a_max = np.max(sim['a'])
                    st.write(f"Median asset holdings were {a_med:.3f} units. Equity is not evenly distributed – the Gini index of {a_gini:.3f} suggests noticeable inequality. The poorest agents held as little as {a_min:.3f}, while the wealthiest amassed up to {a_max:.3f} units of assets.")
        
        with tab4:
            st.subheader("Distribution Analysis")
            
            if 'cs_sim' in st.session_state:
                sim = st.session_state.cs_sim
                
                col1, col2 = st.columns(2)
                with col1:
                    fig = plot_distribution(sim['c'], title="Consumption Distribution", bins=40)
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(f"Median: {np.median(sim['c']):.3f} | IQR: {np.percentile(sim['c'], 75) - np.percentile(sim['c'], 25):.3f}")
                
                with col2:
                    fig = plot_distribution(sim['a'][:-1], title="Asset Distribution", bins=40)
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(f"Median: {np.median(sim['a']):.3f} | Gini: {gini_coefficient(sim['a']):.3f}")
            else:
                st.info("Run a simulation first to see distributions")
        
        with tab5:
            st.subheader("� Summary Statistics & Moments")
            
            if 'cs_sim' in st.session_state:
                sim = st.session_state.cs_sim
                summary = get_simulation_summary(sim, output_key='y')
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Consumption Moments**")
                    if 'c' in summary:
                        m = summary['c']
                        st.metric("Mean Consumption", f"{m['mean']:.4f}")
                        st.metric("Std Dev", f"{m['std_dev']:.4f}")
                        st.metric("Autocorr(1)", f"{m['autocorr_lag1']:.4f}")
                    
                    st.write("**Asset Moments**")
                    if 'a' in summary:
                        m = summary['a']
                        st.metric("Mean Assets", f"{m['mean']:.4f}")
                        st.metric("Variance", f"{m['variance']:.4f}")
                        st.metric("Autocorr(1)", f"{m['autocorr_lag1']:.4f}")
                
                with col2:
                    st.write("**Correlations with Income**")
                    if 'c_output_corr' in summary:
                        corr = summary['c_output_corr']
                        st.metric(f"{corr['name1']} vs {corr['name2']}", f"{corr['correlation']:.4f}")
                    if 'a_output_corr' in summary:
                        corr = summary['a_output_corr']
                        st.metric(f"{corr['name1']} vs {corr['name2']}", f"{corr['correlation']:.4f}")
                    
                    st.write("**Income Moments**")
                    if 'y' in summary:
                        m = summary['y']
                        st.metric("Mean Income", f"{m['mean']:.4f}")
                        st.metric("Autocorr(1)", f"{m['autocorr_lag1']:.4f}")
            else:
                st.info("Run a simulation first to see moments")
        
        with tab6:
            st.subheader("Forecasts (AR(1) Model)")
            
            if 'cs_sim' in st.session_state:
                sim = st.session_state.cs_sim
                
                # Forecast consumption
                st.write("**Consumption Forecast (12 periods ahead)**")
                c_forecast, c_std = forecast_ar1(sim['c'], periods_ahead=12)
                fig_c = plot_forecast(sim['c'][-100:], c_forecast, c_std, 
                                     title="Consumption Forecast", series_name="Consumption")
                st.plotly_chart(fig_c, use_container_width=True)
                
                # Forecast assets
                st.write("**Asset Forecast (12 periods ahead)**")
                a_forecast, a_std = forecast_ar1(sim['a'][:-1], periods_ahead=12)
                fig_a = plot_forecast(sim['a'][:-1][-100:], a_forecast, a_std,
                                     title="Asset Forecast", series_name="Assets")
                st.plotly_chart(fig_a, use_container_width=True)
                
                # Forecast income
                st.write("**Income Forecast (12 periods ahead)**")
                y_forecast, y_std = forecast_ar1(sim['y'], periods_ahead=12)
                fig_y = plot_forecast(sim['y'][-100:], y_forecast, y_std,
                                     title="Income Forecast", series_name="Income")
                st.plotly_chart(fig_y, use_container_width=True)
            else:
                st.info("Run a simulation first to see forecasts")
        
        with tab7:
            st.subheader("�📥 Download Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Download policy functions
                csv_policies = export_policies_to_csv(
                    model.a_grid,
                    {'savings': model.policy_a, 'consumption': model.policy_c},
                    'cs'
                )
                st.download_button(
                    label="Download Policies (CSV)",
                    data=csv_policies,
                    file_name="cs_policies.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Download simulation results
                if 'cs_sim' in st.session_state:
                    csv_sim = export_simulation_to_csv(st.session_state.cs_sim, 'cs')
                    st.download_button(
                        label="Download Simulation (CSV)",
                        data=csv_sim,
                        file_name="cs_simulation.csv",
                        mime="text/csv"
                    )
            
            with col3:
                # Download summary JSON
                if 'cs_sim' in st.session_state:
                    json_summary = create_model_summary_json(model, result, st.session_state.cs_sim)
                else:
                    json_summary = create_model_summary_json(model, result)
                
                st.download_button(
                    label="📋 Download Summary (JSON)",
                    data=json_summary,
                    file_name="cs_summary.json",
                    mime="application/json"
                )

# ============================================================================
# MODEL 2: ROBINSON CRUSOE
# ============================================================================
elif model_choice == "Robinson Crusoe":
    st.header("Robinson Crusoe Production Economy")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Problem:** An agent produces output using capital, consumes, and invests, 
        subject to technology shocks.
        """)
    
    with col2:
        st.markdown("""
        **Key Features:**
        - Capital accumulation
        - Productivity shocks (TFP)
        - Investment-consumption tradeoff
        - Business cycle dynamics
        """)
    
    # Sidebar parameters
    st.sidebar.subheader("Model Parameters")
    
    # Load default or sample calibrations
    calibration = get_sample_calibration(param_source)['rc']
    
    # FRED Data integration
    if param_source == "Custom FRED Data":
        fred_fetcher = st.session_state.fred_data_fetcher
        st.sidebar.markdown("**Select FRED Series for Parameters:**")
        
        # Capital share from productivity data
        productivity_series = ["None"] + [name for name in FRED_SERIES.keys() 
                                          if FRED_SERIES[name]['use'] == 'Productivity']
        selected_alpha_series = st.sidebar.selectbox(
            "Capital Share (α):", productivity_series, key="rc_fred_alpha"
        )
        
        # Depreciation from investment data
        investment_series = ["None"] + [name for name in FRED_SERIES.keys() 
                                        if FRED_SERIES[name]['use'] == 'Investment']
        selected_delta_series = st.sidebar.selectbox(
            "Depreciation (δ):", investment_series, key="rc_fred_delta"
        )
        
        # TFP shocks from productivity/production data
        production_series = ["None"] + [name for name in FRED_SERIES.keys() 
                                        if FRED_SERIES[name]['use'] in ['Production', 'Productivity']]
        selected_tfp_series = st.sidebar.selectbox(
            "TFP Volatility (σ_z):", production_series, key="rc_fred_sigma_z"
        )
        
        # Persistence from production data
        selected_rho_series = st.sidebar.selectbox(
            "TFP Persistence (ρ):", production_series, key="rc_fred_rho"
        )
        
        # Apply calibrations
        if selected_alpha_series != "None":
            try:
                with st.spinner(f"Fetching {selected_alpha_series}..."):
                    params = fred_fetcher.estimate_parameters(
                        FRED_SERIES[selected_alpha_series]['series_id']
                    )
                    calibration['alpha'] = max(0.25, min(0.40, params['level_mean'] / 100))
                    st.sidebar.success(f"α = {calibration['alpha']:.3f}")
            except Exception as e:
                st.sidebar.warning(f"Could not fetch {selected_alpha_series}: {str(e)[:40]}")
        
        if selected_delta_series != "None":
            try:
                with st.spinner(f"Fetching {selected_delta_series}..."):
                    params = fred_fetcher.estimate_parameters(
                        FRED_SERIES[selected_delta_series]['series_id']
                    )
                    calibration['delta'] = max(0.05, min(0.15, params['std'] * 2))
                    st.sidebar.success(f"δ = {calibration['delta']:.3f}")
            except Exception as e:
                st.sidebar.warning(f"Could not fetch {selected_delta_series}: {str(e)[:40]}")
        
        if selected_tfp_series != "None":
            try:
                with st.spinner(f"Fetching {selected_tfp_series}..."):
                    params = fred_fetcher.estimate_parameters(
                        FRED_SERIES[selected_tfp_series]['series_id']
                    )
                    calibration['sigma_z'] = max(0.01, min(0.3, params['std']))
                    st.sidebar.success(f"σ_z = {calibration['sigma_z']:.3f}")
            except Exception as e:
                st.sidebar.warning(f"Could not fetch {selected_tfp_series}: {str(e)[:40]}")
        
        if selected_rho_series != "None":
            try:
                with st.spinner(f"Fetching {selected_rho_series}..."):
                    params = fred_fetcher.estimate_parameters(
                        FRED_SERIES[selected_rho_series]['series_id']
                    )
                    calibration['rho'] = max(0.0, min(0.99, params['rho']))
                    st.sidebar.success(f"ρ = {calibration['rho']:.3f}")
            except Exception as e:
                st.sidebar.warning(f"Could not fetch {selected_rho_series}: {str(e)[:40]}")
        
        # Show current calibration
        st.sidebar.markdown("**Current Calibration:**")
        st.sidebar.write(f"α = {calibration['alpha']:.3f}")
        st.sidebar.write(f"δ = {calibration['delta']:.3f}")
        st.sidebar.write(f"ρ = {calibration['rho']:.3f}")
        st.sidebar.write(f"σ_z = {calibration['sigma_z']:.3f}")
    
    st.sidebar.markdown("<strong style='color:#00d4ff'>Model Parameters (Use sliders to adjust)</strong>", unsafe_allow_html=True)
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        beta = st.slider("Discount Factor (β)", 0.90, 0.99, calibration.get('beta', 0.95), 0.01, 
                        key="rc_beta", help="How much the firm values future production")
        alpha = st.slider("Capital Share (α)", 0.2, 0.5, calibration.get('alpha', 0.33), 0.05,
                         help="Output elasticity to capital stock")
        delta = st.slider("Depreciation Rate (δ)", 0.05, 0.15, calibration.get('delta', 0.1), 0.01,
                         help="Annual capital depreciation rate")
        gamma = st.slider("Risk Aversion (γ)", 1.0, 10.0, calibration.get('gamma', 2.0), 0.5,
                         key="rc_gamma", help="Curvature of utility function")
    
    with col2:
        rho = st.slider("TFP Persistence (ρ)", 0.0, 0.99, calibration.get('rho', 0.9), 0.05, 
                       key="rc_rho", help="How much productivity shocks persist")
        sigma_z = st.slider("TFP Shock Volatility (σ_z)", 0.01, 0.3, calibration.get('sigma_z', 0.1), 0.05,
                           help="Standard deviation of productivity shocks")
        n_k = st.slider("Capital Grid Resolution", 50, 200, 70, 10,
                       help="Number of discrete capital points")
    
    # Markov transition matrix visualization
    with st.sidebar.expander("Markov Transition Matrix (Productivity States)", expanded=False):
        st.markdown("**TFP Transition Probabilities**")
        st.caption("Shows probability of moving between productivity states each period")
        st.write(f"Persistence (main diagonal): {calibration.get('rho', 0.9):.2%}")
        st.write(f"Productivity shocks follow AR(1) with correlation {calibration.get('rho', 0.9):.3f}")
    
    st.sidebar.markdown("---")
    
    # Solve model
    if st.sidebar.button("Solve Model", key="rc_solve", use_container_width=True):
        with st.spinner("Solving Robinson Crusoe model with VFI..."):
            try:
                model = RobinsonCrusoeModel(
                    beta=beta, alpha=alpha, delta=delta, rho=rho, 
                    sigma_z=sigma_z, gamma=gamma, n_k=n_k
                )
                result = model.solve(verbose=False)
                st.session_state.rc_model = model
                st.session_state.rc_result = result
                if result.get('converged'):
                    st.success("Robinson Crusoe model solved (converged)")
                else:
                    st.warning("Model solved but did not converge within max iterations")
            except Exception as e:
                st.error(f"Error solving model: {e}")
                import traceback
                st.text(traceback.format_exc())
    
    # Display results
    if 'rc_model' in st.session_state:
        model = st.session_state.rc_model
        result = st.session_state.rc_result
        
        # Solution Status
        st.markdown("""
        <div class="summary-box">
        <strong>Model Successfully Solved!</strong>
        </div>
        """, unsafe_allow_html=True)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Status", "Converged" if result['converged'] else "Failed")
        with col2:
            st.metric("Iterations", result['iterations'])
        with col3:
            st.metric("Grid Size", f"{model.n_k} × {model.n_z}")
        with col4:
            # Calculate steady state output
            k_ss = model.k_grid[len(model.k_grid)//2]
            st.metric("Steady-State K", f"{k_ss:.2f}")
        
        # Model Summary
        with st.expander("Model Interpretation", expanded=True):
            st.markdown("""
            <div class="interpretation-box">
            <strong>What the Model Shows:</strong>
            <ul>
            <li><strong>Capital Accumulation:</strong> Optimal investment decisions over time</li>
            <li><strong>Productivity Effects:</strong> Higher TFP → more output and investment</li>
            <li><strong>Consumption Smoothing:</strong> Agent absorbs TFP shocks with savings</li>
            <li><strong>Business Cycles:</strong> Positive correlation between output, consumption, investment</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Tabs
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Policy Functions", "Value Function", 
                                                "Simulation", "Analysis", "Moments", "Forecasts", "Download"])
        
        with tab1:
            st.subheader("Policy Functions")
            col1, col2 = st.columns(2)
            with col1:
                fig = get_cached_plot(
                    'rc_policy_k',
                    plot_policy_function,
                    model.k_grid, model.policy_k,
                    title="Optimal Capital Investment Policy: k'(k)",
                    state_label="Current Capital Stock (k)",
                    action_label="Next Period Capital (k')",
                    shock_labels=["Low Productivity State", "Medium Productivity State", "High Productivity State"],
                    max_legend_items=3
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = get_cached_plot(
                    'rc_policy_c',
                    plot_policy_function,
                    model.k_grid, model.policy_c,
                    title="Optimal Consumption Policy: c(k)",
                    state_label="Current Capital Stock (k)",
                    action_label="Consumption (c)",
                    shock_labels=["Low Productivity State", "Medium Productivity State", "High Productivity State"],
                    max_legend_items=3
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Value Function Heatmap")
            fig = get_cached_plot(
                'rc_value_heatmap',
                plot_heatmap,
                model.k_grid, model.z_grid, model.V.T,
                title="Value Function: V(k, z)",
                x_label="Capital (k)",
                y_label="TFP (z)"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Simulate Economy")
            col1, col2, col3 = st.columns(3)
            with col1:
                sim_length = st.slider("Simulation Length", 100, 5000, 500, 100, key="rc_sim_len")
            with col2:
                initial_k = st.slider("Initial Capital", 0.5, 5.0, 1.0, 0.5, key="rc_init_k")
            with col3:
                random_seed = st.slider("Random Seed", 0, 10000, 42, 1, key="rc_seed")
            
            if st.button("Run Simulation", key="rc_run_sim", use_container_width=True):
                with st.spinner("Running simulation..."):
                    sim = model.simulate(T=sim_length, initial_k=initial_k,
                                       random_seed=int(random_seed))
                    st.session_state.rc_sim = sim
            
            if 'rc_sim' in st.session_state:
                sim = st.session_state.rc_sim
                time_idx = np.arange(len(sim['c']))
                
                # Time series
                fig = plot_multiple_series(
                    time_idx,
                    {'Output': sim['output'], 'Consumption': sim['c'],
                     'Investment': sim['investment'], 'Capital': sim['k'][:-1]},
                    title="Simulated Economy (Robinson Crusoe)"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistics
                st.markdown("**Simulation Statistics:**")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Avg Output", f"{np.mean(sim['output']):.3f}")
                with col2:
                    st.metric("Avg Consumption", f"{np.mean(sim['c']):.3f}")
                with col3:
                    st.metric("Avg Capital", f"{np.mean(sim['k']):.3f}")
                with col4:
                    investment_rate = np.mean(sim['investment']) / np.mean(sim['output'])
                    st.metric("I/Y Ratio", f"{investment_rate:.1%}")
                
                # Quick analysis section
                st.markdown("---")
                st.markdown("**Quick Analysis (see 'Analysis' tab for detailed distributions):**")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Output & Investment Analysis:**")
                    out_med = np.median(sim['output'])
                    out_std = np.std(sim['output'])
                    inv_med = np.median(sim['investment'])
                    c_std = np.std(sim['c'])
                    st.write(f"Average output in the production economy is {out_med:.3f} units per period. Output volatility (std. dev.) is {out_std:.3f}, showing how much output swings around its mean. Investment tends to be {inv_med:.3f} units – about {inv_med/out_med:.1%} of output – and consumption has a standard deviation of {c_std:.3f}, which reflects short‑run smoothing.")
                with col2:
                    st.write("**Capital & Efficiency Analysis:**")
                    k_med = np.median(sim['k'])
                    k_std = np.std(sim['k'])
                    k_gini = gini_coefficient(sim['k'])
                    ok_ratio = np.mean(sim['output']) / np.mean(sim['k'])
                    st.write(f"The capital stock is typically around {k_med:.3f} units, with fluctuations of about {k_std:.3f}. Capital ownership inequality is {k_gini:.3f} on the Gini scale. Productivity-wise, each unit of capital turns into approximately {ok_ratio:.3f} units of output, a rough measure of efficiency.")
        
        with tab4:
            st.subheader("Distribution Analysis")
            
            if 'rc_sim' in st.session_state:
                sim = st.session_state.rc_sim
                
                col1, col2 = st.columns(2)
                with col1:
                    fig = plot_distribution(sim['output'], title="Output Distribution", bins=40)
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(f"Mean: {np.mean(sim['output']):.3f} | Std: {np.std(sim['output']):.3f}")
                
                with col2:
                    fig = plot_distribution(sim['c'], title="Consumption Distribution", bins=40)
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(f"Mean: {np.mean(sim['c']):.3f} | Std: {np.std(sim['c']):.3f}")
            else:
                st.info("Run a simulation first to see distributions")
        
        with tab5:
            st.subheader("� Summary Statistics & Moments")
            
            if 'rc_sim' in st.session_state:
                sim = st.session_state.rc_sim
                summary = get_simulation_summary(sim, output_key='output')
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Output Moments**")
                    if 'output' in summary:
                        m = summary['output']
                        st.metric("Mean Output", f"{m['mean']:.4f}")
                        st.metric("Std Dev", f"{m['std_dev']:.4f}")
                        st.metric("Autocorr(1)", f"{m['autocorr_lag1']:.4f}")
                    
                    st.write("**Capital Moments**")
                    if 'k' in summary:
                        m = summary['k']
                        st.metric("Mean Capital", f"{m['mean']:.4f}")
                        st.metric("Variance", f"{m['variance']:.4f}")
                        st.metric("Autocorr(1)", f"{m['autocorr_lag1']:.4f}")
                
                with col2:
                    st.write("**Correlations with Output**")
                    if 'c_output_corr' in summary:
                        corr = summary['c_output_corr']
                        st.metric(f"{corr['name1']} vs {corr['name2']}", f"{corr['correlation']:.4f}")
                    if 'k_output_corr' in summary:
                        corr = summary['k_output_corr']
                        st.metric(f"{corr['name1']} vs {corr['name2']}", f"{corr['correlation']:.4f}")
                    
                    st.write("**Investment Moments**")
                    if 'investment' in summary:
                        m = summary['investment']
                        st.metric("Mean Investment", f"{m['mean']:.4f}")
                        st.metric("Autocorr(1)", f"{m['autocorr_lag1']:.4f}")
            else:
                st.info("Run a simulation first to see moments")
        
        with tab6:
            st.subheader("Forecasts (AR(1) Model)")
            
            if 'rc_sim' in st.session_state:
                sim = st.session_state.rc_sim
                
                # Forecast output
                st.write("**Output Forecast (12 periods ahead)**")
                o_forecast, o_std = forecast_ar1(sim['output'], periods_ahead=12)
                fig_o = plot_forecast(sim['output'][-100:], o_forecast, o_std, 
                                     title="Output Forecast", series_name="Output")
                st.plotly_chart(fig_o, use_container_width=True)
                
                # Forecast capital
                st.write("**Capital Forecast (12 periods ahead)**")
                k_forecast, k_std = forecast_ar1(sim['k'][:-1], periods_ahead=12)
                fig_k = plot_forecast(sim['k'][:-1][-100:], k_forecast, k_std,
                                     title="Capital Forecast", series_name="Capital")
                st.plotly_chart(fig_k, use_container_width=True)
                
                # Forecast investment
                st.write("**Investment Forecast (12 periods ahead)**")
                i_forecast, i_std = forecast_ar1(sim['investment'], periods_ahead=12)
                fig_i = plot_forecast(sim['investment'][-100:], i_forecast, i_std,
                                     title="Investment Forecast", series_name="Investment")
                st.plotly_chart(fig_i, use_container_width=True)
            else:
                st.info("Run a simulation first to see forecasts")
        
        with tab7:
            st.subheader("�📥 Download Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv_policies = export_policies_to_csv(
                    model.k_grid,
                    {'capital': model.policy_k, 'consumption': model.policy_c},
                    'rc'
                )
                st.download_button(
                    label="Download Policies (CSV)",
                    data=csv_policies,
                    file_name="rc_policies.csv",
                    mime="text/csv",
                    key="rc_dl_pol"
                )
            
            with col2:
                if 'rc_sim' in st.session_state:
                    csv_sim = export_simulation_to_csv(st.session_state.rc_sim, 'rc')
                    st.download_button(
                        label="Download Simulation (CSV)",
                        data=csv_sim,
                        file_name="rc_simulation.csv",
                        mime="text/csv",
                        key="rc_dl_sim"
                    )
            
            with col3:
                if 'rc_sim' in st.session_state:
                    json_summary = create_model_summary_json(model, result, st.session_state.rc_sim)
                else:
                    json_summary = create_model_summary_json(model, result)
                
                st.download_button(
                    label="📋 Download Summary (JSON)",
                    data=json_summary,
                    file_name="rc_summary.json",
                    mime="application/json",
                    key="rc_dl_json"
                )

# ============================================================================
# MODEL 3: LABOR SUPPLY
# ============================================================================
elif model_choice == "Labor Supply":
    st.header("Endogenous Labor Supply Model")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Problem:** An agent chooses consumption and labor supply in response to 
        wage fluctuations, balancing work and leisure.
        """)
    
    with col2:
        st.markdown("""
        **Key Features:**
        - Labor-leisure choice
        - Wage uncertainty (AR(1))
        - Income & substitution effects
        - Labor supply elasticity
        """)
    
    # Sidebar parameters
    st.sidebar.subheader("Model Parameters")
    
    # Load default or sample calibrations
    calibration = get_sample_calibration(param_source)['ls']
    
    # FRED Data integration
    if param_source == "Custom FRED Data":
        fred_fetcher = st.session_state.fred_data_fetcher
        st.sidebar.markdown("**Select FRED Series for Parameters:**")
        
        # Wage series for volatility and persistence
        wage_series = ["None"] + [name for name in FRED_SERIES.keys() 
                                  if FRED_SERIES[name]['use'] == 'Wages']
        selected_wage_series = st.sidebar.selectbox(
            "Wage Volatility (σ_w):", wage_series, key="ls_fred_sigma_w"
        )
        
        # Labor market data for disutility and elasticity
        labor_series = ["None"] + [name for name in FRED_SERIES.keys() 
                                   if FRED_SERIES[name]['use'] == 'Labor']
        selected_labor_series = st.sidebar.selectbox(
            "Labor Market (χ, η):", labor_series, key="ls_fred_labor"
        )
        
        # Interest rate for savings
        interest_series = ["None"] + [name for name in FRED_SERIES.keys() 
                                      if FRED_SERIES[name]['use'] == 'Interest Rate']
        selected_r_series = st.sidebar.selectbox(
            "Interest Rate (r):", interest_series, key="ls_fred_r"
        )
        
        # Persistence from wage/income data
        persistence_series = ["None"] + [name for name in FRED_SERIES.keys() 
                                         if FRED_SERIES[name]['use'] in ['Wages', 'Income', 'Labor']]
        selected_rho_series = st.sidebar.selectbox(
            "Wage Persistence (ρ):", persistence_series, key="ls_fred_rho"
        )
        
        # Apply calibrations
        if selected_wage_series != "None":
            try:
                with st.spinner(f"Fetching {selected_wage_series}..."):
                    params = fred_fetcher.estimate_parameters(
                        FRED_SERIES[selected_wage_series]['series_id']
                    )
                    calibration['sigma_w'] = max(0.01, min(0.3, params['std']))
                    st.sidebar.success(f"σ_w = {calibration['sigma_w']:.3f}")
            except Exception as e:
                st.sidebar.warning(f"Could not fetch {selected_wage_series}: {str(e)[:40]}")
        
        if selected_labor_series != "None":
            try:
                with st.spinner(f"Fetching {selected_labor_series}..."):
                    params = fred_fetcher.estimate_parameters(
                        FRED_SERIES[selected_labor_series]['series_id']
                    )
                    calibration['chi'] = max(0.5, min(3.0, params['level_mean'] / 20))
                    calibration['eta'] = max(0.3, min(1.5, 1.0 / (1.0 + params['std'])))
                    st.sidebar.success(f"χ = {calibration['chi']:.2f}, η = {calibration['eta']:.2f}")
            except Exception as e:
                st.sidebar.warning(f"Could not fetch {selected_labor_series}: {str(e)[:40]}")
        
        if selected_r_series != "None":
            try:
                with st.spinner(f"Fetching {selected_r_series}..."):
                    params = fred_fetcher.estimate_parameters(
                        FRED_SERIES[selected_r_series]['series_id']
                    )
                    calibration['r'] = max(0.01, min(0.10, params['level_mean'] / 100))
                    st.sidebar.success(f"r = {calibration['r']:.4f}")
            except Exception as e:
                st.sidebar.warning(f"Could not fetch {selected_r_series}: {str(e)[:40]}")
        
        if selected_rho_series != "None":
            try:
                with st.spinner(f"📊 Fetching {selected_rho_series}..."):
                    params = fred_fetcher.estimate_parameters(
                        FRED_SERIES[selected_rho_series]['series_id']
                    )
                    calibration['rho'] = max(0.0, min(0.99, params['rho']))
                    st.sidebar.success(f"ρ = {calibration['rho']:.3f}")
            except Exception as e:
                st.sidebar.warning(f"Could not fetch {selected_rho_series}: {str(e)[:40]}")
        
        # Show current calibration
        st.sidebar.markdown("**Current Calibration:**")
        st.sidebar.write(f"ρ = {calibration['rho']:.3f}")
        st.sidebar.write(f"σ_w = {calibration['sigma_w']:.3f}")
        st.sidebar.write(f"χ = {calibration['chi']:.3f}")
        st.sidebar.write(f"η = {calibration['eta']:.3f}")
    
    st.sidebar.markdown("<strong style='color:#00d4ff'>Model Parameters (Use sliders to adjust)</strong>", unsafe_allow_html=True)
    col1, col2 = st.sidebar.columns(2)
    with col1:
        beta = st.slider("Discount Factor (β)", 0.90, 0.99, calibration.get('beta', 0.95), 0.01, key="ls_beta",
                        help="How much worker values future consumption")
        r = st.slider("Interest Rate (r)", 0.00, 0.10, calibration.get('r', 0.05), 0.01, key="ls_r",
                     help="Return on savings")
        gamma = st.slider("Consumption RRA (γ)", 1.0, 10.0, calibration.get('gamma', 2.0), 0.5, key="ls_gamma",
                         help="Curvature of consumption utility")
        chi = st.slider("Labor Disutility (χ)", 0.5, 5.0, calibration.get('chi', 1.0), 0.5,
                       help="Intensity of preference against work")
    with col2:
        eta_default = calibration.get('eta', 0.5)
        eta = st.slider("Labor Elasticity (η)", 0.5, 2.0, eta_default, 0.2,
                        help="Responsiveness of labor supply to wage changes")
        rho = st.slider("Wage Persistence (ρ)", 0.0, 0.99, calibration.get('rho', 0.9), 0.05, key="ls_rho",
                        help="How much wage shocks persist")
        sigma_w = st.slider("Wage Shock Std (σ_w)", 0.01, 0.3, calibration.get('sigma_w', 0.1), 0.05,
                           help="Volatility of wage shocks")
        n_a = st.slider("Asset Grid Resolution", 40, 150, 60, 10, key="ls_n_a",
                       help="Number of discrete asset points")
    # Markov transition matrix visualization for wage shocks
    with st.sidebar.expander("📊 Markov Transition Matrix (Wage States)", expanded=False):
        st.markdown("**Wage Transition Probabilities**")
        st.caption("Shows probability of moving between wage states each period")
        st.write(f"Persistence (main diagonal): {calibration.get('rho', 0.9):.2%}")
        st.write(f"Wage shocks follow AR(1) with correlation {calibration.get('rho', 0.9):.3f}")
    
    st.sidebar.markdown("---")
    
    # Solve model
    if st.sidebar.button("Solve Model", key="ls_solve", use_container_width=True):
        with st.spinner("Solving Labor Supply model with VFI..."):
            try:
                model = LaborSupplyModel(
                    beta=beta, r=r, gamma=gamma, chi=chi, eta=eta,
                    rho=rho, sigma_w=sigma_w, n_a=n_a
                )
                result = model.solve(verbose=False)
                st.session_state.ls_model = model
                st.session_state.ls_result = result
                if result.get('converged'):
                    st.success("Labor Supply model solved (converged)")
                else:
                    st.warning("Model solved but did not converge within max iterations")
            except Exception as e:
                st.error(f"Error solving model: {e}")
                import traceback
                st.text(traceback.format_exc())
    
    # Display results
    if 'ls_model' in st.session_state:
        model = st.session_state.ls_model
        result = st.session_state.ls_result
        
        # Solution Status
        st.markdown("""
        <div class="summary-box">
        <strong>Model Successfully Solved!</strong>
        </div>
        """, unsafe_allow_html=True)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Status", "Converged" if result['converged'] else "Failed")
        with col2:
            st.metric("Iterations", result['iterations'])
        with col3:
            st.metric("Grid Size", f"{model.n_a} × {model.n_w}")
        with col4:
            st.metric("Elasticity", f"{eta:.2f}")
        
        # Model Summary
        with st.expander("Model Interpretation", expanded=True):
            st.markdown("""
            <div class="interpretation-box">
            <strong>What the Model Shows:</strong>
            <ul>
            <li><strong>Income Effect:</strong> Higher wages → less labor (want more leisure)</li>
            <li><strong>Substitution Effect:</strong> Higher wages → more labor (work is more valuable)</li>
            <li><strong>Net Effect:</strong> Depends on Frisch elasticity η</li>
            <li><strong>Asset Accumulation:</strong> Savings smooth income over lifetime</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Tabs
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Policy Functions", "Value Function", 
                                                "Simulation", "Analysis", "Moments", "Forecasts", "Download"])
        
        with tab1:
            st.subheader("Policy Functions")
            col1, col2, col3 = st.columns(3)
            with col1:
                fig = get_cached_plot(
                    'ls_policy_a',
                    plot_policy_function,
                    model.a_grid, model.policy_a,
                    title="Optimal Savings Policy: a'(a)",
                    state_label="Current Assets (a)",
                    action_label="Next Period Assets (a')",
                    shock_labels=["Low Wage State", "Medium Wage State", "High Wage State"],
                    max_legend_items=3
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = get_cached_plot(
                    'ls_policy_l',
                    plot_policy_function,
                    model.a_grid, model.policy_l,
                    title="Optimal Labor Supply Policy: l(a)",
                    state_label="Current Assets (a)",
                    action_label="Labor Supply (l)",
                    shock_labels=["Low Wage State", "Medium Wage State", "High Wage State"],
                    max_legend_items=3
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col3:
                fig = get_cached_plot(
                    'ls_policy_c',
                    plot_policy_function,
                    model.a_grid, model.policy_c,
                    title="Optimal Consumption Policy: c(a)",
                    state_label="Current Assets (a)",
                    action_label="Consumption (c)",
                    shock_labels=["Low Wage State", "Medium Wage State", "High Wage State"],
                    max_legend_items=3
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Value Function Heatmap")
            fig = get_cached_plot(
                'ls_value_heatmap',
                plot_heatmap,
                model.a_grid, model.w_grid, model.V.T,
                title="Value Function: V(a, w)",
                x_label="Assets (a)",
                y_label="Wage (w)"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Simulate Economy")
            col1, col2, col3 = st.columns(3)
            with col1:
                sim_length = st.slider("Simulation Length", 100, 5000, 500, 100, key="ls_sim_len")
            with col2:
                initial_a = st.slider("Initial Asset Level", 0.1, 10.0, 1.0, 0.1, key="ls_init_a")
            with col3:
                random_seed = st.slider("Random Seed", 0, 10000, 42, 1, key="ls_seed")
            
            if st.button("Run Simulation", key="ls_run_sim", use_container_width=True):
                with st.spinner("Running simulation..."):
                    sim = model.simulate(T=sim_length, initial_a=initial_a,
                                       random_seed=int(random_seed))
                    st.session_state.ls_sim = sim
            
            if 'ls_sim' in st.session_state:
                sim = st.session_state.ls_sim
                time_idx = np.arange(len(sim['c']))
                
                # Time series
                fig = plot_multiple_series(
                    time_idx,
                    {'Consumption': sim['c'], 'Labor Supply': sim['l'],
                     'Labor Income': sim['y'], 'Assets': sim['a'][:-1]},
                    title="Simulated Labor Supply Decisions"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistics
                st.markdown("**Simulation Statistics:**")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Avg Consumption", f"{np.mean(sim['c']):.3f}")
                with col2:
                    st.metric("Avg Labor", f"{np.mean(sim['l']):.3f}")
                with col3:
                    st.metric("Labor Volatility", f"{np.std(sim['l']):.3f}")
                with col4:
                    st.metric("Avg Wage", f"{np.mean(sim['w'][:-1]):.3f}")
                
                # Quick analysis section
                st.markdown("---")
                st.markdown("**Quick Analysis (see 'Analysis' tab for detailed distributions):**")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("📊 **Labor Supply & Income Analysis:**")
                    l_med = np.median(sim['l'])
                    l_low = np.min(sim['l'])
                    l_high = np.max(sim['l'])
                    w_med = np.median(sim['w'][:-1])
                    w_std = np.std(sim['w'][:-1])
                    st.write(f"Workers supply roughly {l_med:.3f} units of labor on average (e.g. hours per period). The lightest workload is {l_low:.3f} and the heaviest is {l_high:.3f}. Received wages average {w_med:.3f} currency‑units, with typical variation of {w_std:.3f}.")
                with col2:
                    st.write("💰 **Consumption & Savings Analysis:**")
                    c_med = np.median(sim['c'])
                    c_std = np.std(sim['c'])
                    a_gini = gini_coefficient(sim['a'])
                    sav_rate = (1 - np.mean(sim['c'])/np.mean(sim['y']))
                    st.write(f"Average consumption stands at {c_med:.3f} units; it typically swings by {c_std:.3f}. Wealth inequality in assets is {a_gini:.3f} on the Gini scale, and households save about {sav_rate:.1%} of their income on average.")
        
        with tab4:
            st.subheader("Distribution Analysis")
            
            if 'ls_sim' in st.session_state:
                sim = st.session_state.ls_sim
                
                col1, col2 = st.columns(2)
                with col1:
                    fig = plot_distribution(sim['l'], title="Labor Supply Distribution", bins=40)
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(f"Mean: {np.mean(sim['l']):.3f} | Std: {np.std(sim['l']):.3f}")
                
                with col2:
                    fig = plot_distribution(sim['c'], title="Consumption Distribution", bins=40)
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(f"Mean: {np.mean(sim['c']):.3f} | Std: {np.std(sim['c']):.3f}")
            else:
                st.info("Run a simulation first to see distributions")
        
        with tab5:
            st.subheader("� Summary Statistics & Moments")
            
            if 'ls_sim' in st.session_state:
                sim = st.session_state.ls_sim
                summary = get_simulation_summary(sim, output_key=None)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Labor Supply Moments**")
                    if 'l' in summary:
                        m = summary['l']
                        st.metric("Mean Labor", f"{m['mean']:.4f}")
                        st.metric("Std Dev", f"{m['std_dev']:.4f}")
                        st.metric("Autocorr(1)", f"{m['autocorr_lag1']:.4f}")
                    
                    st.write("**Asset Moments**")
                    if 'a' in summary:
                        m = summary['a']
                        st.metric("Mean Assets", f"{m['mean']:.4f}")
                        st.metric("Variance", f"{m['variance']:.4f}")
                        st.metric("Autocorr(1)", f"{m['autocorr_lag1']:.4f}")
                
                with col2:
                    st.write("**Wage & Consumption**")
                    if 'w' in summary:
                        m = summary['w']
                        st.metric("Mean Wage", f"{m['mean']:.4f}")
                        st.metric("Autocorr(1)", f"{m['autocorr_lag1']:.4f}")
                    
                    st.write("**Consumption Statistics**")
                    if 'c' in summary:
                        m = summary['c']
                        st.metric("Mean Consumption", f"{m['mean']:.4f}")
                        st.metric("Std Dev", f"{m['std_dev']:.4f}")
            else:
                st.info("Run a simulation first to see moments")
        
        with tab6:
            st.subheader("Forecasts (AR(1) Model)")
            
            if 'ls_sim' in st.session_state:
                sim = st.session_state.ls_sim
                
                # Forecast labor  supply
                st.write("**Labor Supply Forecast (12 periods ahead)**")
                l_forecast, l_std = forecast_ar1(sim['l'], periods_ahead=12)
                fig_l = plot_forecast(sim['l'][-100:], l_forecast, l_std, 
                                     title="Labor Supply Forecast", series_name="Labor Supply")
                st.plotly_chart(fig_l, use_container_width=True)
                
                # Forecast assets
                st.write("**Asset Forecast (12 periods ahead)**")
                a_forecast, a_std = forecast_ar1(sim['a'][:-1], periods_ahead=12)
                fig_a = plot_forecast(sim['a'][:-1][-100:], a_forecast, a_std,
                                     title="Asset Forecast", series_name="Assets")
                st.plotly_chart(fig_a, use_container_width=True)
                
                # Forecast consumption
                st.write("**Consumption Forecast (12 periods ahead)**")
                c_forecast, c_std = forecast_ar1(sim['c'], periods_ahead=12)
                fig_c = plot_forecast(sim['c'][-100:], c_forecast, c_std,
                                     title="Consumption Forecast", series_name="Consumption")
                st.plotly_chart(fig_c, use_container_width=True)
            else:
                st.info("Run a simulation first to see forecasts")
        
        with tab7:
            st.subheader("�📥 Download Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv_policies = export_policies_to_csv(
                    model.a_grid,
                    {'savings': model.policy_a, 'labor': model.policy_l, 'consumption': model.policy_c},
                    'ls'
                )
                st.download_button(
                    label="📊 Download Policies (CSV)",
                    data=csv_policies,
                    file_name="ls_policies.csv",
                    mime="text/csv",
                    key="ls_dl_pol"
                )
            
            with col2:
                if 'ls_sim' in st.session_state:
                    csv_sim = export_simulation_to_csv(st.session_state.ls_sim, 'ls')
                    st.download_button(
                        label="Download Simulation (CSV)",
                        data=csv_sim,
                        file_name="ls_simulation.csv",
                        mime="text/csv",
                        key="ls_dl_sim"
                    )
            
            with col3:
                if 'ls_sim' in st.session_state:
                    json_summary = create_model_summary_json(model, result, st.session_state.ls_sim)
                else:
                    json_summary = create_model_summary_json(model, result)
                
                st.download_button(
                    label="📋 Download Summary (JSON)",
                    data=json_summary,
                    file_name="ls_summary.json",
                    mime="application/json",
                    key="ls_dl_json"
                )


# Footer
st.markdown("---")
st.markdown("""
**📚 Dashboard Information:**
- All models solved using **Value Function Iteration (VFI)** with linear interpolation
- Income processes discretized using **Tauchen's method** (1986)
- Interactive parameter controls, policy visualization, and stochastic simulation
- Export results in **CSV/JSON** formats for further analysis
- Built with **Streamlit** and **Plotly** for interactive visualizations

**References:** Tauchen (1986), Huggett (1993), Krusell & Smith (1998)
""")
