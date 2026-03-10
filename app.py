"""
Macroeconomic Models Dashboard
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
from utils.moments import compute_moments, compute_correlations, forecast_ar1, get_simulation_summary

# ============================================================================
# GitHub/Cloud Deployment Configuration
# ============================================================================
import os

# Configure for GitHub Codespaces or other cloud environments
if 'GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN' in os.environ or 'CODESPACE_NAME' in os.environ:
    # Running in GitHub Codespaces
    port = int(os.environ.get('PORT', 8502))
    os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'
    os.environ['STREAMLIT_SERVER_PORT'] = str(port)
    st.set_page_config(
        page_title="Macro Models Dashboard",
        layout="wide",
        initial_sidebar_state="expanded"
    )
elif os.environ.get('STREAMLIT_SERVER_HEADLESS', 'false').lower() == 'true' or 'PORT' in os.environ:
    # Running in headless mode or with PORT environment variable (common for cloud deployments)
    port = int(os.environ.get('PORT', 8502))
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


def parse_two_state_shock_plan(plan_text, horizon):
    """Parse comma-separated Low/High plan into 0/1 state indices."""
    mapping = {
        'l': 0,
        'low': 0,
        'h': 1,
        'high': 1,
        '0': 0,
        '1': 1,
    }
    tokens = [t.strip().lower() for t in str(plan_text).split(',') if t.strip()]
    indices = [mapping[t] for t in tokens if t in mapping]

    if not indices:
        indices = [0]

    while len(indices) < horizon:
        indices.append(indices[-1])

    return np.array(indices[:horizon], dtype=int)


def render_static_downloads(fig, data_frame, name_prefix, key_prefix):
    """Render CSV and PNG download buttons for static intuition outputs."""
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Download Static Data (CSV)",
            data=data_frame.to_csv(index=False),
            file_name=f"{name_prefix}_static_intuition.csv",
            mime="text/csv",
            key=f"{key_prefix}_csv",
        )
    with col2:
        try:
            image_bytes = fig.to_image(format="png")
            st.download_button(
                label="Download Static Plot (PNG)",
                data=image_bytes,
                file_name=f"{name_prefix}_static_intuition.png",
                mime="image/png",
                key=f"{key_prefix}_png",
            )
        except Exception:
            st.caption("PNG export unavailable in this environment (CSV download still available).")


def queue_model_reset(model_key):
    """Queue a model reset; applied at top of model block before widgets are created."""
    st.session_state.pending_model_reset = model_key


def apply_pending_reset_if_needed(model_key):
    """Apply pending reset before widgets are instantiated to avoid Streamlit key mutation errors."""
    if st.session_state.get('pending_model_reset') != model_key:
        return

    for key, value in MODEL_SIDEBAR_DEFAULTS[model_key].items():
        st.session_state[key] = value

    for suffix in ('model', 'result', 'sim', 'solve_signature', 'sim_signature'):
        st.session_state.pop(f'{model_key}_{suffix}', None)

    st.session_state.pop('pending_model_reset', None)


def render_model_intro(problem_text, feature_items):
    """Render the top two-column model intro block in a consistent format.

    Notes:
    - Keeps copy/paste UI markup out of each model section.
    - Makes future text edits safer: change one helper instead of three blocks.
    """
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        **The Problem:** {problem_text}
        """)

    with col2:
        bullet_text = "\n".join([f"- {item}" for item in feature_items])
        st.markdown(f"""
        **Key Features:**
        {bullet_text}
        """)


def render_interpretation_box(items):
    """Render the interpretation list with shared HTML wrapper styling.

    Notes:
    - Input format: list of (title, description) tuples.
    - Uses one renderer to avoid drift in wording/style across models.
    """
    list_html = "".join([f"<li><strong>{title}:</strong> {desc}</li>" for title, desc in items])
    st.markdown(
        f"""
        <div class="interpretation-box">
        <strong>What the Model Shows:</strong>
        <ul>
        {list_html}
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_moments_table(series_specs, benchmark_series, benchmark_name, corr_col_label):
    """Build and render the moments table for any model.

    Notes:
    - `series_specs`: list of (display_name, np_array-like).
    - Correlation is computed against `benchmark_series` except for the benchmark row itself,
      where correlation is set to 1.0 by definition.
    - Centralizing this avoids repeating almost identical loops in each model section.
    """
    rows = []
    for name, series in series_specs:
        moms = compute_moments(series)
        corr_val = 1.0 if name == benchmark_name else compute_correlations(series, benchmark_series, name, benchmark_name)['correlation']
        rows.append({
            'Series': name,
            'Mean': round(moms['mean'], 4),
            'Variance': round(moms['variance'], 4),
            'Autocorr(1)': round(moms['autocorr_lag1'], 4),
            corr_col_label: round(corr_val, 4),
        })
    st.table(pd.DataFrame(rows))


def render_run_controls(model_key):
    """Render standardized sidebar run controls and return click states.

    Notes:
    - Keeps widget keys deterministic by model key (`cs`, `rc`, `ls`).
    - Reset is handled via callback to avoid Streamlit state mutation errors.
    """
    with st.sidebar.expander("Section 6: Run Controls", expanded=True):
        solve_clicked = st.button("Solve model", key=f"{model_key}_solve", use_container_width=True)
        simulate_clicked = st.button("Simulate path", key=f"{model_key}_simulate_sidebar", use_container_width=True)
        st.button(
            "Reset to defaults",
            key=f"{model_key}_reset_defaults",
            use_container_width=True,
            on_click=queue_model_reset,
            args=(model_key,),
        )
    return solve_clicked, simulate_clicked


def should_auto_solve(model_key, solve_clicked, current_signature):
    """Return True if model should be solved based on click/state/signature changes."""
    prev_solve_signature = st.session_state.get(f'{model_key}_solve_signature')
    return (
        solve_clicked
        or f'{model_key}_model' not in st.session_state
        or prev_solve_signature != current_signature
    )


def should_auto_simulate(model_key, simulate_clicked, current_signature):
    """Return True if simulation should be rerun based on click/state/signature changes."""
    prev_sim_signature = st.session_state.get(f'{model_key}_sim_signature')
    return (
        simulate_clicked
        or (f'{model_key}_model' in st.session_state and f'{model_key}_sim' not in st.session_state)
        or prev_sim_signature != current_signature
    )


def store_solve_result(model_key, model, result, solve_signature):
    """Persist solved model + metadata and invalidate stale simulations."""
    st.session_state[f'{model_key}_model'] = model
    st.session_state[f'{model_key}_result'] = result
    st.session_state.pop(f'{model_key}_sim', None)
    st.session_state[f'{model_key}_solve_signature'] = solve_signature
    st.session_state.pop(f'{model_key}_sim_signature', None)


def store_sim_result(model_key, sim_data, sim_signature):
    """Persist simulation output and signature for change detection."""
    st.session_state[f'{model_key}_sim'] = sim_data
    st.session_state[f'{model_key}_sim_signature'] = sim_signature


def render_solution_status(metrics):
    """Render the standard solved banner and metric row.

    Notes:
    - `metrics` should be a list of `(label, value)` tuples.
    - Shared renderer keeps status row structure identical across models.
    """
    st.markdown(
        """
        <div class="summary-box">
        <strong>Model Successfully Solved!</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics):
        with col:
            st.metric(label, value)

DEFAULT_CALIBRATION = {
    'cs': {'beta': 0.95, 'r': 0.03, 'gamma': 2.0},
    'rc': {'beta': 0.95, 'alpha': 0.33, 'delta': 0.08, 'gamma': 2.0},
    'ls': {'beta': 0.95, 'r': 0.05, 'gamma': 2.0, 'chi': 1.5, 'eta': 0.8},
}

MODEL_SIDEBAR_DEFAULTS = {
    'cs': {
        'cs_beta': 0.95,
        'cs_sigma': 2.0,
        'cs_r': 0.03,
        'cs_a_min': 0.01,
        'cs_a_max': 50.0,
        'cs_n_a': 90,
        'cs_y_low': 0.8,
        'cs_y_high': 1.2,
        'cs_p_stay_low': 0.90,
        'cs_p_stay_high': 0.90,
        'cs_sim_len': 100,
        'cs_init_a': 1.0,
        'cs_init_income': 'Low',
        'cs_seed': 42,
        'cs_forecast_horizon': 10,
    },
    'rc': {
        'rc_beta': 0.95,
        'rc_sigma': 2.0,
        'rc_alpha': 0.33,
        'rc_delta': 0.08,
        'rc_init_k': 1.0,
        'rc_k_min': 0.1,
        'rc_k_max': 50.0,
        'rc_n_k': 90,
        'rc_z_low': 0.9,
        'rc_z_high': 1.1,
        'rc_p_stay_low': 0.90,
        'rc_p_stay_high': 0.90,
        'rc_sim_len': 100,
        'rc_seed': 42,
        'rc_forecast_horizon': 10,
    },
    'ls': {
        'ls_beta': 0.95,
        'ls_sigma': 2.0,
        'ls_phi': 1.5,
        'ls_eta': 0.8,
        'ls_r': 0.05,
        'ls_n_a': 60,
        'ls_w_low': 0.8,
        'ls_w_high': 1.2,
        'ls_p_stay_low': 0.90,
        'ls_p_stay_high': 0.90,
        'ls_sim_len': 100,
        'ls_init_a': 1.0,
        'ls_init_wage_state': 'Low',
        'ls_seed': 42,
        'ls_forecast_horizon': 10,
        'ls_high_accuracy': False,
    },
}

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
        gap: 10px;
        background-color: #002b4d;
        border-radius: 8px;
        padding: 5px;
        margin-top: 12px;
        margin-bottom: 14px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #003d66;
        border-radius: 6px;
        color: #ffffff;
        font-family: 'Garamond', serif;
        padding: 0.45rem 0.85rem;
    }

    .panel-spacer {
        height: 1rem;
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
runtime_app_path = os.path.abspath(__file__)
runtime_cwd = os.getcwd()
st.caption(f"Running file: `{runtime_app_path}` | CWD: `{runtime_cwd}`")
st.sidebar.caption(f"Running file: `{runtime_app_path}`")
st.markdown("""
Interactive dashboard for exploring three economics models that show how people make decisions about 
**money, work, and savings** when the future is uncertain.  
**Adjust sliders • Watch graphs update • Download results**
""")

st.markdown("""
### How to Use This App
1. **Pick a model** from the dropdown in the sidebar (left).
2. **Adjust the sliders** to change things like income uncertainty or how patient people are.
3. The app **automatically updates** when you move sliders—no buttons needed!
4. Watch graphs and analysis appear showing how people would behave under your settings.
5. Use the **"Reset to defaults"** button if you want to start over.
""")

st.sidebar.markdown("---")

# Sidebar - Model selection
st.sidebar.markdown("**Section 1: Model Selection**")
model_choice = st.sidebar.selectbox(
    "**Step 1: Select Model**",
    [
        "Model 1: Consumption-Savings",
        "Model 2: Robinson Crusoe",
        "Model 3: Endogenous Labor Supply",
    ]
)

st.sidebar.markdown("---")

# ============================================================================
# MODEL 1: CONSUMPTION-SAVINGS
# ============================================================================
if model_choice == "Model 1: Consumption-Savings":
    apply_pending_reset_if_needed('cs')
    st.header("Stochastic Consumption-Savings Model")
    render_model_intro(
        "Imagine someone with an unpredictable paycheck who needs to decide "
        "how much to spend today versus save for tomorrow, but they can't borrow money.",
        [
            "Income varies randomly (like gig work or seasonal jobs)",
            "Saving for 'rainy days' (precautionary savings)",
            "Can't go into debt",
            "Shows why some people save more than others",
        ],
    )
    
    # Sidebar parameters
    st.sidebar.subheader("Step 2: Adjust Economic Parameters")
    
    # Load baseline calibration values
    calibration = DEFAULT_CALIBRATION['cs']

    with st.sidebar.expander("Section 2: Preferences", expanded=False):
        beta = st.slider(
            "Discount factor (beta)",
            0.90,
            0.99,
            calibration.get('beta', 0.95),
            0.01,
            key="cs_beta",
        )
        sigma = st.slider(
            "Risk aversion (sigma)",
            1.0,
            10.0,
            calibration.get('gamma', 2.0),
            0.5,
            key="cs_sigma",
        )

    with st.sidebar.expander("Section 3: Technology or Budget", expanded=False):
        r = st.slider("Interest rate (r)", 0.00, 0.10, calibration.get('r', 0.03), 0.005, key="cs_r")
        a_min = st.slider("Borrowing limit / asset minimum", -2.0, 1.0, 0.01, 0.01, key="cs_a_min")
        a_max = st.slider("Asset grid maximum", 5.0, 100.0, 50.0, 1.0, key="cs_a_max")
        n_a = st.slider("Asset grid resolution", 50, 250, 90, 10, key="cs_n_a")

    with st.sidebar.expander("Section 4: Markov Shock Process (Income)", expanded=False):
        y_low = st.slider("Low income value", 0.2, 2.0, 0.8, 0.05, key="cs_y_low")
        y_high = st.slider("High income value", 0.2, 3.0, 1.2, 0.05, key="cs_y_high")
        if y_high <= y_low:
            st.sidebar.warning("High income should be greater than low income. Adjusting automatically.")
            y_high = y_low + 0.05

        p_stay_low = st.slider("P(stay in low state)", 0.01, 0.99, 0.90, 0.01, key="cs_p_stay_low")
        p_stay_high = st.slider("P(stay in high state)", 0.01, 0.99, 0.90, 0.01, key="cs_p_stay_high")

        transition_matrix = np.array([
            [p_stay_low, 1.0 - p_stay_low],
            [1.0 - p_stay_high, p_stay_high],
        ])
        st.sidebar.caption("Transition matrix")
        st.sidebar.dataframe(
            pd.DataFrame(
                transition_matrix,
                index=["Low", "High"],
                columns=["Low", "High"],
            ),
            use_container_width=True,
        )

    with st.sidebar.expander("Section 5: Simulation", expanded=False):
        sim_length = st.slider("Number of periods", 100, 200, 100, 10, key="cs_sim_len")
        initial_a = st.slider("Initial assets", a_min, a_max, 1.0, 0.1, key="cs_init_a")
        initial_income_state = st.selectbox("Initial income state", ["Low", "High"], key="cs_init_income")
        random_seed = st.number_input("Random seed", min_value=0, max_value=100000, value=42, step=1, key="cs_seed")
        forecast_horizon = st.slider("Forecast horizon", 5, 30, 10, 1, key="cs_forecast_horizon")

    # Keep legacy args for backward compatibility but use custom 2-state process
    rho = 0.0
    sigma_y = 0.1

    solve_clicked, simulate_clicked = render_run_controls('cs')

    current_cs_solve_signature = (
        round(float(beta), 8),
        round(float(r), 8),
        round(float(sigma), 8),
        int(n_a),
        round(float(a_min), 8),
        round(float(a_max), 8),
        round(float(y_low), 8),
        round(float(y_high), 8),
        round(float(p_stay_low), 8),
        round(float(p_stay_high), 8),
    )
    cs_should_solve = should_auto_solve('cs', solve_clicked, current_cs_solve_signature)

    # Solve model (auto-runs when sidebar parameters change)
    if cs_should_solve:
        with st.spinner("Solving Consumption-Savings model with VFI..."):
            try:
                model = ConsumptionSavingsModel(
                    beta=beta,
                    r=r,
                    rho=rho,
                    sigma_y=sigma_y,
                    gamma=sigma,
                    n_a=n_a,
                    a_min=a_min,
                    a_max=a_max,
                    y_grid=np.array([y_low, y_high]),
                    P_y=transition_matrix,
                )
                result = model.solve(verbose=False)
                store_solve_result('cs', model, result, current_cs_solve_signature)
                if result.get('converged'):
                    st.success("Consumption-Savings model solved (converged)")
                else:
                    st.warning(
                        f"Model solved but did not converge within max iterations. "
                        f"Final convergence gap: {result.get('final_diff', float('nan')):.6g}"
                    )
            except Exception as e:
                st.error(f"Error solving model: {e}")
                import traceback
                st.text(traceback.format_exc())

    current_cs_sim_signature = (
        int(sim_length),
        round(float(initial_a), 8),
        str(initial_income_state),
        int(random_seed),
        current_cs_solve_signature,
    )
    cs_should_simulate = should_auto_simulate('cs', simulate_clicked, current_cs_sim_signature)

    if cs_should_simulate:
        if 'cs_model' not in st.session_state:
            st.warning("Solve the model first before simulating.")
        else:
            with st.spinner("Simulating path..."):
                init_idx = 0 if initial_income_state == "Low" else 1
                st.session_state.cs_sim = st.session_state.cs_model.simulate(
                    T=sim_length,
                    initial_a=initial_a,
                    random_seed=int(random_seed),
                    initial_y_state=init_idx,
                )
                store_sim_result('cs', st.session_state.cs_sim, current_cs_sim_signature)
            st.success("Simulation complete.")
    
    # Display results
    if 'cs_model' in st.session_state:
        model = st.session_state.cs_model
        result = st.session_state.cs_result
        
        render_solution_status([
            ("Status", "Converged" if result['converged'] else "Failed"),
            ("Iterations", result['iterations']),
            ("Grid Size", f"{model.n_a} × {model.n_y}"),
            ("Solution Time", "< 5 sec"),
        ])
        
        # Model Summary
        with st.expander("Model Interpretation", expanded=True):
            render_interpretation_box([
                ("Decision Rules", "How much to save or spend based on your current wealth"),
                ("Safety Net Behavior", "When income is more uncertain, people save more for emergencies"),
                ("Spending Smoothness", "People try to keep spending steady even when income jumps around"),
                ("Can't Borrow", "When you have little saved, you can't spend more than you earn"),
            ])
        
        st.subheader("Model Output")
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "Policy Functions", "Value Function", "Simulation", "Analysis", "Summary Statistics", "Forecast Panel", "Download"
        ])

        st.markdown("### 1. Model Description")
        st.markdown(
            "**What you start with:** Current savings and whether income is high or low today. "
            "**What you choose:** How much to spend now and how much to save for later. "
            "**What's uncertain:** Income randomly switches between high and low states. "
            "**The tradeoff:** Enjoy spending money today vs. having a cushion for uncertain future income."
        )

        st.markdown("### 2. Policy Functions")
        shock_labels = ["Low Income State", "High Income State"] if model.n_y == 2 else None
        col1, col2 = st.columns(2)
        with col1:
            fig = get_cached_plot(
                'cs_policy_a_ordered',
                plot_policy_function,
                model.a_grid, model.policy_a,
                title="Savings Policy a'(a, z)",
                state_label="Assets (a)",
                action_label="Next Assets (a')",
                shock_labels=shock_labels,
                max_legend_items=3,
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = get_cached_plot(
                'cs_policy_c_ordered',
                plot_policy_function,
                model.a_grid, model.policy_c,
                title="Consumption Policy c(a, z)",
                state_label="Assets (a)",
                action_label="Consumption (c)",
                shock_labels=shock_labels,
                max_legend_items=3,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Static Intuition")
        beta_grid = np.linspace(0.90, 0.99, 25)
        savings_share = (beta_grid * (1.0 + r)) / (1.0 + beta_grid * (1.0 + r))
        fig_static = plot_multiple_series(
            beta_grid,
            {'Implied Savings Share': savings_share},
            title="Patience and Savings Tendency (Static Intuition)",
        )
        st.plotly_chart(fig_static, use_container_width=True)
        st.caption("Higher beta increases the value of future utility, which raises the savings propensity in this static mapping.")
        static_df = pd.DataFrame({
            'beta': beta_grid,
            'implied_savings_share': savings_share,
        })
        render_static_downloads(fig_static, static_df, "model1_consumption_savings", "cs_static")

        st.markdown("### 3. Simulation Plots")
        if 'cs_sim' in st.session_state:
            sim = st.session_state.cs_sim
            t_idx = np.arange(len(sim['c']))
            fig = plot_multiple_series(
                t_idx,
                {
                    'Consumption': sim['c'],
                    'Savings/Assets': sim['a'][:-1],
                    'Income': sim['y'],
                },
                title="Simulated Paths"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Dynamic economic analysis based on live moments
            st.markdown("#### 📊 Dynamic Economic Analysis")
            c_moms = compute_moments(sim['c'])
            a_moms = compute_moments(sim['a'][:-1])
            y_moms = compute_moments(sim['y'])
            cy_corr = compute_correlations(sim['c'], sim['y'], "Consumption", "Income")['correlation']
            ay_corr = compute_correlations(sim['a'][:-1], sim['y'], "Assets", "Income")['correlation']
            
            volatility_ratio = np.sqrt(c_moms['variance']) / np.sqrt(y_moms['variance']) if y_moms['variance'] > 0 else 0
            smoothing_quality = "strong" if volatility_ratio < 0.5 else "moderate" if volatility_ratio < 0.8 else "weak"
            
            persistence_desc = "highly persistent" if c_moms['autocorr_lag1'] > 0.7 else "moderately persistent" if c_moms['autocorr_lag1'] > 0.4 else "weakly persistent"
            
            comovement_desc = "tightly procyclical" if cy_corr > 0.7 else "moderately procyclical" if cy_corr > 0.4 else "weakly procyclical" if cy_corr > 0 else "countercyclical"
            
            st.markdown(
                f"This person shows **{smoothing_quality} spending stability** (volatility score: **{volatility_ratio:.3f}**). "
                f"{'They do a great job keeping spending steady by using savings when income drops and saving when income rises' if volatility_ratio < 0.6 else 'Their spending still bounces around quite a bit when income changes'}. "
                f"Spending patterns are **{persistence_desc}** (stickiness: {c_moms['autocorr_lag1']:.3f}), meaning "
                f"{'they maintain similar spending habits from month to month' if c_moms['autocorr_lag1'] > 0.6 else 'they quickly adjust spending when circumstances change'}. "
                f"The link between spending and income is **{cy_corr:.3f}**—"
                f"{'spending closely follows income ups and downs' if abs(cy_corr) > 0.5 else 'spending stays fairly steady regardless of income fluctuations'}. "
                f"On average, they keep **{a_moms['mean']:.3f}** in savings, which {'gives them a strong financial cushion' if a_moms['mean'] > 2.0 else 'provides some protection but not a huge buffer' if a_moms['mean'] > 0.5 else 'leaves them living paycheck-to-paycheck'}. "
                f"Savings trend (correlation: **{ay_corr:.3f}**): {'they build savings when income is good' if ay_corr > 0.3 else 'savings stay fairly constant regardless of income' if ay_corr > -0.3 else 'surprisingly, savings drop when income rises'}."
            )
        else:
            st.info("Run simulation from sidebar Section 6 to view simulation plots.")

        st.markdown("### 4. Moments Table")
        if 'cs_sim' in st.session_state:
            sim = st.session_state.cs_sim
            render_moments_table(
                series_specs=[
                    ("Consumption", sim['c']),
                    ("Assets", sim['a'][:-1]),
                    ("Income", sim['y']),
                ],
                benchmark_series=sim['y'],
                benchmark_name="Income",
                corr_col_label="Corr with Income",
            )
        else:
            st.info("Run simulation to compute moments table.")

        st.markdown("### 5. Forecast Panel")
        if 'cs_sim' in st.session_state:
            sim = st.session_state.cs_sim
            forecast_len = int(forecast_horizon)
            plan_text = st.text_input(
                "Enter next income shocks (comma-separated Low/High)",
                value="Low,High,High,Low",
                key="cs_forecast_shock_plan",
            )
            shock_idx = parse_two_state_shock_plan(plan_text, forecast_len)
            a_now = float(sim['a'][-1])
            c_fore, a_fore, y_fore = [], [], []

            for idx in shock_idx:
                y_t = float(model.y_grid[idx])
                c_t = float(np.interp(a_now, model.a_grid, model.policy_c[:, idx]))
                c_t = max(c_t, 1e-6)
                a_next = (1 + model.r) * a_now + y_t - c_t
                a_next = max(a_next, model.a_min)
                c_fore.append(c_t)
                a_fore.append(a_next)
                y_fore.append(y_t)
                a_now = a_next

            fig = plot_multiple_series(
                np.arange(forecast_len),
                {'Consumption Forecast': c_fore, 'Assets Forecast': a_fore, 'Income Shock Path': y_fore},
                title=f"{forecast_len}-Period Conditional Forecast"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run simulation first; forecast starts from latest simulated state.")

        st.markdown("### 6. Economic Summary Text")
        if 'cs_sim' in st.session_state:
            sim = st.session_state.cs_sim
            c_mean = np.mean(sim['c'])
            a_mean = np.mean(sim['a'][:-1])
            c_ac = compute_moments(sim['c'])['autocorr_lag1']
            cy = compute_correlations(sim['c'], sim['y'], "Consumption", "Income")['correlation']
            st.markdown(
                f"With current parameters, average consumption is **{c_mean:.3f}** and average assets are **{a_mean:.3f}**. "
                f"Consumption persistence is **{c_ac:.3f}** at lag 1, and the correlation between consumption and income is **{cy:.3f}**. "
                f"This indicates {'strong' if abs(cy) > 0.5 else 'moderate'} co-movement with income under the selected shock process."
            )
        else:
            st.info("Run simulation to generate dynamic economic summary text.")

        st.markdown("<div class='panel-spacer'></div>", unsafe_allow_html=True)

        with tab1:
            st.subheader("Policy Functions")
            shock_labels = ["Low Income State", "High Income State"] if model.n_y == 2 else None
            col1, col2 = st.columns(2)
            with col1:
                fig = get_cached_plot(
                    'cs_policy_a',
                    plot_policy_function,
                    model.a_grid, model.policy_a,
                    title="Optimal Savings Policy: a'(a)",
                    state_label="Current Assets (a)",
                    action_label="Next Period Assets (a')",
                    shock_labels=shock_labels,
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
                    shock_labels=shock_labels,
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
            st.subheader("Step 4: Simulate")
            init_idx = 0 if initial_income_state == "Low" else 1
            st.caption(
                f"Setup: T={sim_length}, initial assets={initial_a:.2f}, "
                f"initial income state={initial_income_state}, seed={int(random_seed)}"
            )
            
            if 'cs_sim' in st.session_state:
                sim = st.session_state.cs_sim
                time_idx = np.arange(len(sim['c']))

                st.markdown("**Simulated Consumption Path**")
                fig_c_path = plot_simulated_path(
                    time_idx,
                    {'Consumption': sim['c']},
                    title="Simulated Consumption Path",
                    y_label="Level of Consumption"
                )
                st.plotly_chart(fig_c_path, use_container_width=True)

                st.markdown("**Simulated Asset Path**")
                fig_a_path = plot_simulated_path(
                    time_idx,
                    {'Assets': sim['a'][:-1]},
                    title="Simulated Asset Path"
                )
                st.plotly_chart(fig_a_path, use_container_width=True)

                # Combined path for context
                fig = plot_multiple_series(
                    time_idx,
                    {'Consumption': sim['c'], 'Assets': sim['a'][:-1], 'Income': sim['y'][:-1]},
                    title="Simulated Paths (Consumption-Savings Model)"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Assignment-aligned statistics
                st.markdown("**Required Summary Statistics:**")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean Consumption", f"{np.mean(sim['c']):.3f}")
                with col2:
                    st.metric("Variance Consumption", f"{np.var(sim['c']):.3f}")
                with col3:
                    st.metric("Mean Assets", f"{np.mean(sim['a'][:-1]):.3f}")
                with col4:
                    st.metric("Variance Assets", f"{np.var(sim['a'][:-1]):.3f}")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Autocorr c(1)", f"{compute_moments(sim['c'])['autocorr_lag1']:.3f}")
                with col2:
                    st.metric("Autocorr a(1)", f"{compute_moments(sim['a'][:-1])['autocorr_lag1']:.3f}")
                with col3:
                    corr_ca = compute_correlations(sim['c'], sim['y'], "Consumption", "Income")
                    st.metric("Corr(c, y)", f"{corr_ca['correlation']:.3f}")
                with col4:
                    corr_ay = compute_correlations(sim['a'][:-1], sim['y'], "Assets", "Income")
                    st.metric("Corr(a, y)", f"{corr_ay['correlation']:.3f}")
                
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
            st.subheader("Summary Statistics & Moments")
            
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
            st.subheader("Forecast Panel (AR(1) Model)")
            
            if 'cs_sim' in st.session_state:
                sim = st.session_state.cs_sim
                
                # Forecast consumption
                st.write(f"**Consumption Forecast ({forecast_horizon} periods ahead)**")
                c_forecast, c_std = forecast_ar1(sim['c'], periods_ahead=int(forecast_horizon))
                fig_c = plot_forecast(sim['c'][-100:], c_forecast, c_std, 
                                     title="Consumption Forecast", series_name="Consumption")
                st.plotly_chart(fig_c, use_container_width=True)
                
                # Forecast assets
                st.write(f"**Asset Forecast ({forecast_horizon} periods ahead)**")
                a_forecast, a_std = forecast_ar1(sim['a'][:-1], periods_ahead=int(forecast_horizon))
                fig_a = plot_forecast(sim['a'][:-1][-100:], a_forecast, a_std,
                                     title="Asset Forecast", series_name="Assets")
                st.plotly_chart(fig_a, use_container_width=True)
                
                # Forecast income
                st.write(f"**Income Forecast ({forecast_horizon} periods ahead)**")
                y_forecast, y_std = forecast_ar1(sim['y'], periods_ahead=int(forecast_horizon))
                fig_y = plot_forecast(sim['y'][-100:], y_forecast, y_std,
                                     title="Income Forecast", series_name="Income")
                st.plotly_chart(fig_y, use_container_width=True)
            else:
                st.info("Run a simulation first to see forecasts")
        
        with tab7:
            st.subheader("Download Results")
            
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
                    label="Download Summary (JSON)",
                    data=json_summary,
                    file_name="cs_summary.json",
                    mime="application/json"
                )

# ============================================================================
# MODEL 2: ROBINSON CRUSOE
# ============================================================================
elif model_choice == "Model 2: Robinson Crusoe":
    apply_pending_reset_if_needed('rc')
    st.header("Robinson Crusoe Production Economy")
    render_model_intro(
        "Like Robinson Crusoe on an island, someone uses tools and equipment "
        "(capital) to produce goods, then decides whether to consume them or build better tools.",
        [
            "Building up equipment over time",
            "Random good days and bad days (productivity shocks)",
            "Choosing between enjoying today vs. investing for tomorrow",
            "Shows how economies expand and contract",
        ],
    )
    
    # Sidebar parameters
    st.sidebar.subheader("Step 2: Adjust Economic Parameters")

    calibration = DEFAULT_CALIBRATION['rc']

    with st.sidebar.expander("Section 2: Preferences", expanded=False):
        beta = st.slider("Discount factor (beta)", 0.90, 0.99, calibration.get('beta', 0.95), 0.01, key="rc_beta")
        sigma = st.slider("Risk aversion (sigma)", 1.0, 10.0, calibration.get('gamma', 2.0), 0.5, key="rc_sigma")

    with st.sidebar.expander("Section 3: Technology or Budget", expanded=False):
        alpha = st.slider("Capital share (alpha)", 0.20, 0.50, calibration.get('alpha', 0.33), 0.01, key="rc_alpha")
        delta = st.slider("Depreciation rate (delta)", 0.03, 0.20, calibration.get('delta', 0.08), 0.01, key="rc_delta")
        initial_k = st.slider("Initial capital", 0.1, 20.0, 1.0, 0.1, key="rc_init_k")
        k_min = st.slider("Capital grid minimum", 0.05, 5.0, 0.1, 0.05, key="rc_k_min")
        k_max = st.slider("Capital grid maximum", 10.0, 150.0, 50.0, 1.0, key="rc_k_max")
        if k_max <= k_min:
            st.sidebar.warning("Capital grid maximum must exceed minimum. Adjusting maximum.")
            k_max = k_min + 1.0
        n_k = st.slider("Capital grid resolution", 50, 250, 90, 10, key="rc_n_k")

    with st.sidebar.expander("Section 4: Markov Shock Process (TFP)", expanded=False):
        z_low = st.slider("Low TFP", 0.3, 1.5, 0.9, 0.05, key="rc_z_low")
        z_high = st.slider("High TFP", 0.5, 2.5, 1.1, 0.05, key="rc_z_high")
        if z_high <= z_low:
            st.sidebar.warning("High TFP should exceed low TFP. Adjusting automatically.")
            z_high = z_low + 0.05

        p_stay_low = st.slider("Probability of staying in low TFP", 0.01, 0.99, 0.90, 0.01, key="rc_p_stay_low")
        p_stay_high = st.slider("Probability of staying in high TFP", 0.01, 0.99, 0.90, 0.01, key="rc_p_stay_high")

        tfp_transition = np.array([
            [p_stay_low, 1.0 - p_stay_low],
            [1.0 - p_stay_high, p_stay_high],
        ])
        st.sidebar.caption("TFP transition matrix")
        st.sidebar.dataframe(
            pd.DataFrame(
                tfp_transition,
                index=["Low", "High"],
                columns=["Low", "High"],
            ),
            use_container_width=True,
        )

    with st.sidebar.expander("Section 5: Simulation", expanded=False):
        sim_length = st.slider("Number of periods", 100, 300, 100, 10, key="rc_sim_len")
        random_seed = st.number_input("Random seed", min_value=0, max_value=100000, value=42, step=1, key="rc_seed")
        forecast_horizon = st.slider("Forecast horizon", 5, 30, 10, 1, key="rc_forecast_horizon")

    # Kept for compatibility with constructor signature when using custom TFP chain
    rho = 0.0
    sigma_z = 0.1

    solve_clicked, simulate_clicked = render_run_controls('rc')

    current_rc_solve_signature = (
        round(float(beta), 8),
        round(float(sigma), 8),
        round(float(alpha), 8),
        round(float(delta), 8),
        int(n_k),
        round(float(k_min), 8),
        round(float(k_max), 8),
        round(float(z_low), 8),
        round(float(z_high), 8),
        round(float(p_stay_low), 8),
        round(float(p_stay_high), 8),
    )
    rc_should_solve = should_auto_solve('rc', solve_clicked, current_rc_solve_signature)

    # Solve model (auto-runs when sidebar parameters change)
    if rc_should_solve:
        with st.spinner("Solving Robinson Crusoe model with VFI..."):
            try:
                model = RobinsonCrusoeModel(
                    beta=beta,
                    alpha=alpha,
                    delta=delta,
                    rho=rho,
                    sigma_z=sigma_z,
                    gamma=sigma,
                    n_k=n_k,
                    k_min=k_min,
                    k_max=k_max,
                    z_grid=np.array([z_low, z_high]),
                    P_z=tfp_transition,
                )
                result = model.solve(verbose=False)
                store_solve_result('rc', model, result, current_rc_solve_signature)
                if result.get('converged'):
                    st.success("Robinson Crusoe model solved (converged)")
                else:
                    st.warning(
                        f"Model solved but did not converge within max iterations. "
                        f"Final convergence gap: {result.get('final_diff', float('nan')):.6g}"
                    )
            except Exception as e:
                st.error(f"Error solving model: {e}")
                import traceback
                st.text(traceback.format_exc())

    current_rc_sim_signature = (
        int(sim_length),
        round(float(initial_k), 8),
        int(random_seed),
        current_rc_solve_signature,
    )
    rc_should_simulate = should_auto_simulate('rc', simulate_clicked, current_rc_sim_signature)

    if rc_should_simulate:
        if 'rc_model' not in st.session_state:
            st.warning("Solve the model first before simulating.")
        else:
            with st.spinner("Simulating path..."):
                rc_sim = st.session_state.rc_model.simulate(
                    T=sim_length,
                    initial_k=initial_k,
                    random_seed=int(random_seed),
                    initial_z_state=0,
                )
                store_sim_result('rc', rc_sim, current_rc_sim_signature)
            st.success("Simulation complete.")
    
    # Display results
    if 'rc_model' in st.session_state:
        model = st.session_state.rc_model
        result = st.session_state.rc_result
        
        k_ss = model.k_grid[len(model.k_grid)//2]
        render_solution_status([
            ("Status", "Converged" if result['converged'] else "Failed"),
            ("Iterations", result['iterations']),
            ("Grid Size", f"{model.n_k} × {model.n_z}"),
            ("Steady-State K", f"{k_ss:.2f}"),
        ])
        
        # Model Summary
        with st.expander("Model Interpretation", expanded=True):
            render_interpretation_box([
                ("Building Equipment", "How much to invest in better tools each period"),
                ("Good Days & Bad Days", "When you're more productive, you make more stuff and can invest more"),
                ("Dealing with Ups and Downs", "Using savings to maintain stable consumption despite productivity swings"),
                ("Economic Booms & Busts", "Output, spending, and investment all tend to rise and fall together"),
            ])
        
        st.subheader("Model Output")
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "Policy Functions", "Value Function", "Simulation", "Analysis", "Summary Statistics", "Forecast Panel", "Download"
        ])

        st.markdown("### 1. Model Description")
        st.markdown(
            "**What you start with:** Current equipment level and whether today is a 'good productivity day' or 'bad productivity day'. "
            "**What you choose:** How much output to consume now vs. invest in better equipment. "
            "**What's uncertain:** Productivity randomly switches between good days and bad days. "
            "**The tradeoff:** Enjoy consumption today vs. build better equipment for more production tomorrow."
        )

        st.markdown("### 2. Policy Functions")
        shock_labels = ["Low TFP", "High TFP"] if model.n_z == 2 else None
        col1, col2 = st.columns(2)
        with col1:
            fig = get_cached_plot(
                'rc_policy_k_ordered',
                plot_policy_function,
                model.k_grid, model.policy_k,
                title="Capital Policy k'(k, z)",
                state_label="Capital (k)",
                action_label="Next Capital (k')",
                shock_labels=shock_labels,
                max_legend_items=3,
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = get_cached_plot(
                'rc_policy_c_ordered',
                plot_policy_function,
                model.k_grid, model.policy_c,
                title="Consumption Policy c(k, z)",
                state_label="Capital (k)",
                action_label="Consumption (c)",
                shock_labels=shock_labels,
                max_legend_items=3,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Static Intuition")
        beta_grid = np.linspace(0.90, 0.99, 25)
        denom = np.maximum(1.0 - beta_grid * (1.0 - delta), 1e-6)
        k_ss = np.power((alpha * beta_grid) / denom, 1.0 / (1.0 - alpha))
        fig_static = plot_multiple_series(
            beta_grid,
            {'Steady-State Capital k*': k_ss},
            title="Steady-State Capital vs Patience (Static Intuition)",
        )
        st.plotly_chart(fig_static, use_container_width=True)
        st.caption("With higher beta, agents are more patient, so the steady-state capital stock rises in this deterministic benchmark.")
        static_df = pd.DataFrame({
            'beta': beta_grid,
            'steady_state_capital': k_ss,
        })
        render_static_downloads(fig_static, static_df, "model2_robinson_crusoe", "rc_static")

        st.markdown("### 3. Simulation Plots")
        if 'rc_sim' in st.session_state:
            sim = st.session_state.rc_sim
            t_idx = np.arange(len(sim['c']))
            fig = plot_multiple_series(
                t_idx,
                {
                    'Consumption': sim['c'],
                    'Capital': sim['k'][:-1],
                    'Output': sim['output'],
                },
                title="Simulated Paths"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Dynamic economic analysis based on live moments
            st.markdown("#### 📊 Dynamic Economic Analysis")
            y_moms = compute_moments(sim['output'])
            c_moms = compute_moments(sim['c'])
            k_moms = compute_moments(sim['k'][:-1])
            cy_corr = compute_correlations(sim['c'], sim['output'], "Consumption", "Output")['correlation']
            ky_corr = compute_correlations(sim['k'][:-1], sim['output'], "Capital", "Output")['correlation']
            
            # Calculate investment
            investment = np.diff(sim['k']) + delta * sim['k'][:-1]
            i_moms = compute_moments(investment)
            iy_corr = compute_correlations(investment, sim['output'], "Investment", "Output")['correlation']
            
            output_volatility = np.sqrt(y_moms['variance'])
            c_volatility = np.sqrt(c_moms['variance'])
            i_volatility = np.sqrt(i_moms['variance'])
            
            cycle_strength = "pronounced" if output_volatility > 0.15 else "moderate" if output_volatility > 0.08 else "mild"
            business_cycle_pattern = "standard RBC predictions" if (cy_corr > 0.5 and iy_corr > 0.5) else "non-standard co-movement"
            
            persistence_desc = "highly persistent shocks" if y_moms['autocorr_lag1'] > 0.7 else "moderately persistent shocks" if y_moms['autocorr_lag1'] > 0.4 else "transitory shocks"
            
            st.markdown(
                f"This economy shows **{cycle_strength} boom-bust swings** (output jumpiness: **{output_volatility:.4f}**). "
                f"Spending-output link is **{cy_corr:.3f}** and investment-output link is **{iy_corr:.3f}**. "
                f"{'Both spending and investment go up when output goes up and down when output drops—this matches real-world patterns' if (cy_corr > 0.5 and iy_corr > 0.5) else 'The patterns here differ from typical real economies, possibly due to the parameter settings'}. "
                f"Investment swings are **{'bigger than output swings' if i_volatility > output_volatility else 'smaller than output swings'}** ({i_volatility/output_volatility:.2f}x), "
                f"{'which matches reality where investment is the most volatile part of the economy' if i_volatility > output_volatility else 'which is unusual—normally investment swings wildly'}. "
                f"Output stickiness (**{y_moms['autocorr_lag1']:.3f}**) shows **{persistence_desc}**, meaning "
                f"{'good times and bad times tend to last for a while' if y_moms['autocorr_lag1'] > 0.6 else 'conditions change quickly from period to period'}. "
                f"Average equipment level is **{k_moms['mean']:.3f}**, which {'is quite high, suggesting more good days than bad' if k_moms['mean'] > 5.0 else 'is fairly normal'}. "
                f"Equipment-output relationship (**{ky_corr:.3f}**): {'as expected, more equipment means more output' if ky_corr > 0.3 else 'surprisingly weak connection' if ky_corr > -0.3 else 'unexpected negative relationship'}."
            )
        else:
            st.info("Run simulation from sidebar Section 6 to view simulation plots.")

        st.markdown("### 4. Moments Table")
        if 'rc_sim' in st.session_state:
            sim = st.session_state.rc_sim
            render_moments_table(
                series_specs=[
                    ("Output", sim['output']),
                    ("Consumption", sim['c']),
                    ("Capital", sim['k'][:-1]),
                ],
                benchmark_series=sim['output'],
                benchmark_name="Output",
                corr_col_label="Corr with Output",
            )
        else:
            st.info("Run simulation to compute moments table.")

        st.markdown("### 5. Forecast Panel")
        if 'rc_sim' in st.session_state:
            sim = st.session_state.rc_sim
            forecast_len = int(forecast_horizon)
            plan_text = st.text_input(
                "Enter next TFP shocks (comma-separated Low/High)",
                value="Low,High,High,Low",
                key="rc_forecast_shock_plan",
            )
            shock_idx = parse_two_state_shock_plan(plan_text, forecast_len)
            k_now = float(sim['k'][-1])
            c_fore, k_fore, y_fore = [], [], []

            for idx in shock_idx:
                z_t = float(model.z_grid[idx])
                y_t = float(model.production_function(k_now, z_t))
                c_t = float(np.interp(k_now, model.k_grid, model.policy_c[:, idx]))
                k_next = float(np.interp(k_now, model.k_grid, model.policy_k[:, idx]))
                c_t = max(c_t, 1e-6)
                k_next = max(k_next, model.k_min)
                c_fore.append(c_t)
                k_fore.append(k_next)
                y_fore.append(y_t)
                k_now = k_next

            fig = plot_multiple_series(
                np.arange(forecast_len),
                {'Consumption Forecast': c_fore, 'Capital Forecast': k_fore, 'Output Forecast': y_fore},
                title=f"{forecast_len}-Period Conditional Forecast"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run simulation first; forecast starts from latest simulated state.")

        st.markdown("### 6. Economic Summary Text")
        if 'rc_sim' in st.session_state:
            sim = st.session_state.rc_sim
            y_mean = np.mean(sim['output'])
            c_mean = np.mean(sim['c'])
            k_mean = np.mean(sim['k'][:-1])
            cy = compute_correlations(sim['c'], sim['output'], "Consumption", "Output")['correlation']
            ky = compute_correlations(sim['k'][:-1], sim['output'], "Capital", "Output")['correlation']
            st.markdown(
                f"Average output is **{y_mean:.3f}**, with consumption at **{c_mean:.3f}** and capital at **{k_mean:.3f}**. "
                f"The consumption-output correlation is **{cy:.3f}** and capital-output correlation is **{ky:.3f}**, "
                f"which indicates {'strong' if (abs(cy) > 0.5 or abs(ky) > 0.5) else 'moderate'} business-cycle co-movement."
            )
        else:
            st.info("Run simulation to generate dynamic economic summary text.")

        st.markdown("<div class='panel-spacer'></div>", unsafe_allow_html=True)

        with tab1:
            st.subheader("Policy Functions")
            shock_labels = ["Low TFP", "High TFP"] if model.n_z == 2 else None
            col1, col2 = st.columns(2)
            with col1:
                fig = get_cached_plot(
                    'rc_policy_k',
                    plot_policy_function,
                    model.k_grid, model.policy_k,
                    title="Optimal Capital Investment Policy: k'(k)",
                    state_label="Current Capital Stock (k)",
                    action_label="Next Period Capital (k')",
                    shock_labels=shock_labels,
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
                    shock_labels=shock_labels,
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
            st.subheader("Step 4: Simulate")
            st.caption(f"Setup: T={sim_length}, initial capital={initial_k:.2f}, seed={int(random_seed)}")
            
            if 'rc_sim' in st.session_state:
                sim = st.session_state.rc_sim
                time_idx = np.arange(len(sim['c']))
                
                st.markdown("**Simulated Output**")
                fig_out = plot_simulated_path(
                    time_idx,
                    {'Output': sim['output']},
                    title="Simulated Output"
                )
                st.plotly_chart(fig_out, use_container_width=True)

                st.markdown("**Simulated Consumption**")
                fig_cons = plot_simulated_path(
                    time_idx,
                    {'Consumption': sim['c']},
                    title="Simulated Consumption",
                    y_label="Level of Consumption"
                )
                st.plotly_chart(fig_cons, use_container_width=True)

                st.markdown("**Simulated Capital Accumulation**")
                fig_cap = plot_simulated_path(
                    time_idx,
                    {'Capital': sim['k'][:-1]},
                    title="Simulated Capital Accumulation"
                )
                st.plotly_chart(fig_cap, use_container_width=True)

                fig = plot_multiple_series(
                    time_idx,
                    {'Output': sim['output'], 'Consumption': sim['c'],
                     'Investment': sim['investment'], 'Capital': sim['k'][:-1]},
                    title="Simulated Economy (Robinson Crusoe)"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("**Required Moments and Correlations:**")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean Output", f"{np.mean(sim['output']):.3f}")
                with col2:
                    st.metric("Mean Consumption", f"{np.mean(sim['c']):.3f}")
                with col3:
                    st.metric("Mean Capital", f"{np.mean(sim['k'][:-1]):.3f}")
                with col4:
                    st.metric("Std Output", f"{np.std(sim['output']):.3f}")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Autocorr y(1)", f"{compute_moments(sim['output'])['autocorr_lag1']:.3f}")
                with col2:
                    st.metric("Autocorr c(1)", f"{compute_moments(sim['c'])['autocorr_lag1']:.3f}")
                with col3:
                    st.metric("Autocorr k(1)", f"{compute_moments(sim['k'][:-1])['autocorr_lag1']:.3f}")
                with col4:
                    corr = compute_correlations(sim['c'], sim['output'], "Consumption", "Output")
                    st.metric("Corr(c, y)", f"{corr['correlation']:.3f}")

                corr_k = compute_correlations(sim['k'][:-1], sim['output'], "Capital", "Output")
                st.metric("Corr(k, y)", f"{corr_k['correlation']:.3f}")
                
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
            st.subheader("Summary Statistics & Moments")
            
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
            st.subheader("Forecast Panel (AR(1) Model)")
            
            if 'rc_sim' in st.session_state:
                sim = st.session_state.rc_sim
                
                # Forecast output
                st.write(f"**Output Forecast ({forecast_horizon} periods ahead)**")
                o_forecast, o_std = forecast_ar1(sim['output'], periods_ahead=int(forecast_horizon))
                fig_o = plot_forecast(sim['output'][-100:], o_forecast, o_std, 
                                     title="Output Forecast", series_name="Output")
                st.plotly_chart(fig_o, use_container_width=True)
                
                # Forecast capital
                st.write(f"**Capital Forecast ({forecast_horizon} periods ahead)**")
                k_forecast, k_std = forecast_ar1(sim['k'][:-1], periods_ahead=int(forecast_horizon))
                fig_k = plot_forecast(sim['k'][:-1][-100:], k_forecast, k_std,
                                     title="Capital Forecast", series_name="Capital")
                st.plotly_chart(fig_k, use_container_width=True)
                
                # Forecast investment
                st.write(f"**Investment Forecast ({forecast_horizon} periods ahead)**")
                i_forecast, i_std = forecast_ar1(sim['investment'], periods_ahead=int(forecast_horizon))
                fig_i = plot_forecast(sim['investment'][-100:], i_forecast, i_std,
                                     title="Investment Forecast", series_name="Investment")
                st.plotly_chart(fig_i, use_container_width=True)
            else:
                st.info("Run a simulation first to see forecasts")
        
        with tab7:
            st.subheader("Download Results")
            
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
                    label="Download Summary (JSON)",
                    data=json_summary,
                    file_name="rc_summary.json",
                    mime="application/json",
                    key="rc_dl_json"
                )

# ============================================================================
# MODEL 3: LABOR SUPPLY
# ============================================================================
elif model_choice == "Model 3: Endogenous Labor Supply":
    apply_pending_reset_if_needed('ls')
    st.header("Endogenous Labor Supply Model")
    render_model_intro(
        "Someone decides how many hours to work and how much to spend "
        "as their hourly wage changes, balancing the desire for money vs. free time.",
        [
            "Choosing between work and relaxation",
            "Wages that change unpredictably",
            "Two competing forces: 'work more when paid more' vs. 'I can afford to work less'",
            "How responsive work hours are to wage changes",
        ],
    )
    
    # Sidebar parameters
    st.sidebar.subheader("Step 2: Adjust Economic Parameters")

    calibration = DEFAULT_CALIBRATION['ls']

    with st.sidebar.expander("Section 2: Preferences", expanded=False):
        beta = st.slider("Discount factor (beta)", 0.90, 0.99, calibration.get('beta', 0.95), 0.01, key="ls_beta")
        sigma = st.slider("Risk aversion on consumption (sigma)", 1.0, 10.0, calibration.get('gamma', 2.0), 0.5, key="ls_sigma")
        phi = st.slider("Leisure weight / labor disutility parameter (phi)", 0.2, 6.0, calibration.get('chi', 1.5), 0.1, key="ls_phi")
        eta = st.slider("Frisch-style leisure curvature (optional)", 0.4, 2.5, calibration.get('eta', 0.8), 0.1, key="ls_eta")

    with st.sidebar.expander("Section 3: Technology or Budget", expanded=False):
        r = st.slider("Interest rate (r)", 0.00, 0.10, calibration.get('r', 0.05), 0.005, key="ls_r")
        n_a = st.slider("Asset grid resolution", 40, 180, 60, 10, key="ls_n_a")

    with st.sidebar.expander("Section 4: Markov Shock Process (Wage Rate)", expanded=False):
        w_low = st.slider("Low wage", 0.2, 2.0, 0.8, 0.05, key="ls_w_low")
        w_high = st.slider("High wage", 0.3, 3.0, 1.2, 0.05, key="ls_w_high")
        if w_high <= w_low:
            st.sidebar.warning("High wage should exceed low wage. Adjusting automatically.")
            w_high = w_low + 0.05

        p_stay_low = st.slider("Probability of staying in low wage", 0.01, 0.99, 0.90, 0.01, key="ls_p_stay_low")
        p_stay_high = st.slider("Probability of staying in high wage", 0.01, 0.99, 0.90, 0.01, key="ls_p_stay_high")

        wage_transition = np.array([
            [p_stay_low, 1.0 - p_stay_low],
            [1.0 - p_stay_high, p_stay_high],
        ])
        st.sidebar.caption("Wage transition matrix")
        st.sidebar.dataframe(
            pd.DataFrame(
                wage_transition,
                index=["Low", "High"],
                columns=["Low", "High"],
            ),
            use_container_width=True,
        )

    with st.sidebar.expander("Section 5: Simulation", expanded=False):
        sim_length = st.slider("Number of periods", 100, 300, 100, 10, key="ls_sim_len")
        initial_a = st.slider("Initial assets", 0.01, 20.0, 1.0, 0.1, key="ls_init_a")
        initial_wage_state = st.selectbox("Initial wage state", ["Low", "High"], key="ls_init_wage_state")
        random_seed = st.number_input("Random seed", min_value=0, max_value=100000, value=42, step=1, key="ls_seed")
        forecast_horizon = st.slider("Forecast horizon", 5, 30, 10, 1, key="ls_forecast_horizon")
        high_accuracy = st.checkbox("High accuracy mode (slower)", value=False, key="ls_high_accuracy")

    # Compatibility parameters when custom wage process is provided
    rho = 0.0
    sigma_w = 0.1

    solve_clicked, simulate_clicked = render_run_controls('ls')

    current_ls_solve_signature = (
        round(float(beta), 8),
        round(float(r), 8),
        round(float(sigma), 8),
        round(float(phi), 8),
        round(float(eta), 8),
        int(n_a),
        round(float(w_low), 8),
        round(float(w_high), 8),
        round(float(p_stay_low), 8),
        round(float(p_stay_high), 8),
    )
    ls_should_solve = should_auto_solve('ls', solve_clicked, current_ls_solve_signature)

    # Solve model (auto-runs when sidebar parameters change)
    if ls_should_solve:
        with st.spinner("Solving Labor Supply model with VFI..."):
            try:
                model = LaborSupplyModel(
                    beta=beta,
                    r=r,
                    gamma=sigma,
                    chi=phi,
                    eta=eta,
                    rho=rho,
                    sigma_w=sigma_w,
                    n_a=n_a,
                    w_grid=np.array([w_low, w_high]),
                    P_w=wage_transition,
                )
                ls_tol = 5e-5 if high_accuracy else 1e-4
                ls_max_iter = 350 if high_accuracy else 250
                result = model.solve(tol=ls_tol, max_iter=ls_max_iter, verbose=False)
                store_solve_result('ls', model, result, current_ls_solve_signature)
                if result.get('converged'):
                    st.success("Labor Supply model solved (converged)")
                else:
                    st.warning(
                        f"Model solved but did not converge within max iterations. "
                        f"Final convergence gap: {result.get('final_diff', float('nan')):.6g}"
                    )
            except Exception as e:
                st.error(f"Error solving model: {e}")
                import traceback
                st.text(traceback.format_exc())

    current_ls_sim_signature = (
        int(sim_length),
        round(float(initial_a), 8),
        str(initial_wage_state),
        int(random_seed),
        current_ls_solve_signature,
    )
    ls_should_simulate = should_auto_simulate('ls', simulate_clicked, current_ls_sim_signature)

    if ls_should_simulate:
        if 'ls_model' not in st.session_state:
            st.warning("Solve the model first before simulating.")
        else:
            with st.spinner("Simulating path..."):
                init_w_idx = 0 if initial_wage_state == "Low" else 1
                ls_sim = st.session_state.ls_model.simulate(
                    T=sim_length,
                    initial_a=initial_a,
                    random_seed=int(random_seed),
                    initial_w_state=init_w_idx,
                )
                store_sim_result('ls', ls_sim, current_ls_sim_signature)
            st.success("Simulation complete.")
    
    # Display results
    if 'ls_model' in st.session_state:
        model = st.session_state.ls_model
        result = st.session_state.ls_result
        
        render_solution_status([
            ("Status", "Converged" if result['converged'] else "Failed"),
            ("Iterations", result['iterations']),
            ("Grid Size", f"{model.n_a} × {model.n_w}"),
            ("Elasticity", f"{eta:.2f}"),
        ])
        
        # Model Summary
        with st.expander("Model Interpretation", expanded=True):
            render_interpretation_box([
                ("Wealth Effect", "When wages rise, you might work less because you're richer"),
                ("Opportunity Cost Effect", "When wages rise, each hour of leisure is more 'expensive,' so you might work more"),
                ("Which Wins?", "The balance between these two forces (controlled by the elasticity parameter)"),
                ("Savings Buffer", "Building savings helps keep spending stable despite changing wages"),
            ])
        
        st.subheader("Model Output")
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "Policy Functions", "Value Function", "Simulation", "Analysis", "Summary Statistics", "Forecast Panel", "Download"
        ])

        st.markdown("### 1. Model Description")
        st.markdown(
            "**What you start with:** Current savings and whether wages are high or low today. "
            "**What you choose:** How much to spend, how many hours to work, and how much to save. "
            "**What's uncertain:** Wages randomly switch between high and low. "
            "**The tradeoff:** Enjoying consumption and free time today vs. building savings for an uncertain wage future."
        )

        st.markdown("### 2. Policy Functions")
        shock_labels = ["Low Wage State", "High Wage State"] if model.n_w == 2 else None
        col1, col2, col3 = st.columns(3)
        with col1:
            fig = get_cached_plot(
                'ls_policy_c_ordered',
                plot_policy_function,
                model.a_grid, model.policy_c,
                title="Consumption Policy c(a, w)",
                state_label="Assets (a)",
                action_label="Consumption (c)",
                shock_labels=shock_labels,
                max_legend_items=3,
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = get_cached_plot(
                'ls_policy_l_ordered',
                plot_policy_function,
                model.a_grid, model.policy_l,
                title="Labor Policy l(a, w)",
                state_label="Assets (a)",
                action_label="Labor (l)",
                shock_labels=shock_labels,
                max_legend_items=3,
            )
            st.plotly_chart(fig, use_container_width=True)
        with col3:
            fig = get_cached_plot(
                'ls_policy_a_ordered',
                plot_policy_function,
                model.a_grid, model.policy_a,
                title="Savings Policy a'(a, w)",
                state_label="Assets (a)",
                action_label="Next Assets (a')",
                shock_labels=shock_labels,
                max_legend_items=3,
            )
            st.plotly_chart(fig, use_container_width=True)

        fig_leisure = get_cached_plot(
            'ls_policy_leisure_ordered',
            plot_policy_function,
            model.a_grid,
            1.0 - model.policy_l,
            title="Leisure Implied from Labor: 1 - l(a, w)",
            state_label="Assets (a)",
            action_label="Leisure",
            shock_labels=shock_labels,
            max_legend_items=3,
        )
        st.plotly_chart(fig_leisure, use_container_width=True)

        st.markdown("### Static Intuition")
        wage_grid = np.linspace(max(0.1, w_low * 0.7), w_high * 1.3, 40)
        c_ref = max(np.mean(model.policy_c), 1e-4)
        labor_static = np.clip(((wage_grid * (c_ref ** (-sigma))) / max(phi, 1e-6)) ** eta, 0.0, 1.0)
        fig_static = plot_multiple_series(
            wage_grid,
            {'Static Labor Supply l(w)': labor_static},
            title="Intra-temporal Labor Supply Curve (Static Intuition)",
        )
        st.plotly_chart(fig_static, use_container_width=True)
        st.caption("Holding consumption fixed, higher wages increase labor supply through the substitution effect in this static relation.")
        static_df = pd.DataFrame({
            'wage': wage_grid,
            'static_labor_supply': labor_static,
        })
        render_static_downloads(fig_static, static_df, "model3_labor_supply", "ls_static")

        st.markdown("### 3. Simulation Plots")
        if 'ls_sim' in st.session_state:
            sim = st.session_state.ls_sim
            t_idx = np.arange(len(sim['c']))
            fig = plot_multiple_series(
                t_idx,
                {
                    'Consumption': sim['c'],
                    'Savings/Assets': sim['a'][:-1],
                    'Labor': sim['l'],
                    'Wage': sim['w'],
                    'Income': sim['y'],
                },
                title="Simulated Paths"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Dynamic economic analysis based on live moments
            st.markdown("#### 📊 Dynamic Economic Analysis")
            c_moms = compute_moments(sim['c'])
            l_moms = compute_moments(sim['l'])
            w_moms = compute_moments(sim['w'])
            y_moms = compute_moments(sim['y'])
            lw_corr = compute_correlations(sim['l'], sim['w'], "Labor", "Wage")['correlation']
            cy_corr = compute_correlations(sim['c'], sim['y'], "Consumption", "Income")['correlation']
            cw_corr = compute_correlations(sim['c'], sim['w'], "Consumption", "Wage")['correlation']
            
            # Calculate Frisch elasticity approximation
            leisure = 1.0 - sim['l']
            avg_leisure = np.mean(leisure)
            labor_response = "strong" if abs(lw_corr) > 0.5 else "moderate" if abs(lw_corr) > 0.25 else "weak"
            
            substitution_dominance = "substitution effect dominant" if lw_corr > 0.3 else "income effect dominant" if lw_corr < -0.2 else "offsetting income and substitution effects"
            
            labor_volatility = np.sqrt(l_moms['variance'])
            wage_volatility = np.sqrt(w_moms['variance'])
            labor_wage_ratio = labor_volatility / wage_volatility if wage_volatility > 0 else 0
            
            st.markdown(
                f"Work hours show a **{labor_response} response** to wage changes (correlation: **{lw_corr:.3f}**). "
                f"This reveals **{substitution_dominance}**: "
                f"{'when wages go up, they work more hours, trading free time for extra money' if lw_corr > 0.3 else 'when wages go up, they work fewer hours, enjoying the wealth effect' if lw_corr < -0.2 else 'wealth and opportunity-cost effects roughly cancel out, so hours barely change with wages'}. "
                f"On average, they work **{l_moms['mean']:.3f}** of their time (leaving **{avg_leisure:.3f}** for leisure), "
                f"{'showing they are quite work-focused' if l_moms['mean'] > 0.5 else 'showing they prioritize free time' if l_moms['mean'] < 0.3 else 'showing balanced work-life choices'}. "
                f"Hour flexibility score is **{labor_wage_ratio:.3f}**, meaning "
                f"{'hours adjust a lot when wages change—very flexible' if labor_wage_ratio > 1.0 else 'hours adjust moderately to wage changes' if labor_wage_ratio > 0.5 else 'hours stay fairly constant regardless of wage changes—inflexible'}. "
                f"Spending-wage link (**{cw_corr:.3f}**): {'spending moves closely with wages' if cw_corr > 0.5 else 'spending is somewhat protected from wage swings through savings'}. "
                f"Spending-income link (**{cy_corr:.3f}**) is {'very tight' if cy_corr > 0.7 else 'moderate' if cy_corr > 0.4 else 'loose'}, "
                f"suggesting they {'live paycheck-to-paycheck' if cy_corr > 0.6 else 'use savings to smooth spending when income jumps around'}. "
                f"Work hour stickiness (**{l_moms['autocorr_lag1']:.3f}**): "
                f"{'they stick to consistent work schedules over time' if l_moms['autocorr_lag1'] > 0.5 else 'they adjust hours frequently based on current conditions'}."
            )
        else:
            st.info("Run simulation from sidebar Section 6 to view simulation plots.")

        st.markdown("### 4. Moments Table")
        if 'ls_sim' in st.session_state:
            sim = st.session_state.ls_sim
            render_moments_table(
                series_specs=[
                    ("Consumption", sim['c']),
                    ("Savings/Assets", sim['a'][:-1]),
                    ("Labor", sim['l']),
                    ("Wage", sim['w']),
                    ("Income", sim['y']),
                ],
                benchmark_series=sim['y'],
                benchmark_name="Income",
                corr_col_label="Corr with Income",
            )
        else:
            st.info("Run simulation to compute moments table.")

        st.markdown("### 5. Forecast Panel")
        if 'ls_sim' in st.session_state:
            sim = st.session_state.ls_sim
            forecast_len = int(forecast_horizon)
            plan_text = st.text_input(
                "Enter next wage shocks (comma-separated Low/High)",
                value="Low,High,High,Low",
                key="ls_forecast_shock_plan",
            )
            shock_idx = parse_two_state_shock_plan(plan_text, forecast_len)
            a_now = float(sim['a'][-1])
            c_fore, l_fore, w_fore, y_fore = [], [], [], []

            for idx in shock_idx:
                w_t = float(model.w_grid[idx])
                c_t = float(np.interp(a_now, model.a_grid, model.policy_c[:, idx]))
                l_t = float(np.interp(a_now, model.a_grid, model.policy_l[:, idx]))
                a_next = float(np.interp(a_now, model.a_grid, model.policy_a[:, idx]))
                c_t = max(c_t, 1e-6)
                l_t = float(np.clip(l_t, 0.0, 1.0))
                y_t = w_t * l_t
                c_fore.append(c_t)
                l_fore.append(l_t)
                w_fore.append(w_t)
                y_fore.append(y_t)
                a_now = a_next

            fig = plot_multiple_series(
                np.arange(forecast_len),
                {
                    'Consumption Forecast': c_fore,
                    'Labor Forecast': l_fore,
                    'Wage Path': w_fore,
                    'Income Forecast': y_fore,
                },
                title=f"{forecast_len}-Period Conditional Forecast"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run simulation first; forecast starts from latest simulated state.")

        st.markdown("### 6. Economic Summary Text")
        if 'ls_sim' in st.session_state:
            sim = st.session_state.ls_sim
            c_mean = np.mean(sim['c'])
            l_mean = np.mean(sim['l'])
            w_mean = np.mean(sim['w'])
            lw = compute_correlations(sim['l'], sim['w'], "Labor", "Wage")['correlation']
            cy = compute_correlations(sim['c'], sim['y'], "Consumption", "Income")['correlation']
            st.markdown(
                f"Average consumption is **{c_mean:.3f}**, average labor is **{l_mean:.3f}**, and average wage is **{w_mean:.3f}**. "
                f"Labor-wage correlation is **{lw:.3f}** and consumption-income correlation is **{cy:.3f}**, "
                f"suggesting {'elastic' if abs(lw) > 0.4 else 'muted'} labor response under the selected shock persistence."
            )
        else:
            st.info("Run simulation to generate dynamic economic summary text.")

        st.markdown("<div class='panel-spacer'></div>", unsafe_allow_html=True)

        with tab1:
            st.subheader("Policy Functions")
            shock_labels = ["Low Wage State", "High Wage State"] if model.n_w == 2 else None
            col1, col2, col3 = st.columns(3)
            with col1:
                fig = get_cached_plot(
                    'ls_policy_a',
                    plot_policy_function,
                    model.a_grid, model.policy_a,
                    title="Optimal Savings Policy: a'(a)",
                    state_label="Current Assets (a)",
                    action_label="Next Period Assets (a')",
                    shock_labels=shock_labels,
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
                    shock_labels=shock_labels,
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
                    shock_labels=shock_labels,
                    max_legend_items=3
                )
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Leisure Implied from Labor Policy**")
            fig_leisure = get_cached_plot(
                'ls_policy_leisure',
                plot_policy_function,
                model.a_grid,
                1.0 - model.policy_l,
                title="Implied Leisure Policy: 1 - l(a)",
                state_label="Current Assets (a)",
                action_label="Leisure",
                shock_labels=shock_labels,
                max_legend_items=3,
            )
            st.plotly_chart(fig_leisure, use_container_width=True)
        
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
            st.subheader("Step 4: Simulate")
            init_w_idx = 0 if initial_wage_state == "Low" else 1
            st.caption(
                f"Setup: T={sim_length}, initial assets={initial_a:.2f}, "
                f"initial wage state={initial_wage_state}, seed={int(random_seed)}"
            )
            
            if 'ls_sim' in st.session_state:
                sim = st.session_state.ls_sim
                time_idx = np.arange(len(sim['c']))
                
                st.markdown("**Simulated Wage Path**")
                fig_w = plot_simulated_path(
                    time_idx,
                    {'Wage': sim['w']},
                    title="Simulated Wage Path"
                )
                st.plotly_chart(fig_w, use_container_width=True)

                st.markdown("**Simulated Consumption Path**")
                fig_c = plot_simulated_path(
                    time_idx,
                    {'Consumption': sim['c']},
                    title="Simulated Consumption Path",
                    y_label="Level of Consumption"
                )
                st.plotly_chart(fig_c, use_container_width=True)

                st.markdown("**Simulated Labor Path**")
                fig_l = plot_simulated_path(
                    time_idx,
                    {'Labor Supply': sim['l']},
                    title="Simulated Labor Path"
                )
                st.plotly_chart(fig_l, use_container_width=True)

                fig = plot_multiple_series(
                    time_idx,
                    {'Consumption': sim['c'], 'Labor Supply': sim['l'],
                     'Labor Income': sim['y'], 'Assets': sim['a'][:-1]},
                    title="Simulated Labor Supply Decisions"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("**Mean / Variance / Autocorrelation**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean Consumption", f"{np.mean(sim['c']):.3f}")
                    st.metric("Var Consumption", f"{np.var(sim['c']):.3f}")
                    st.metric("Autocorr c(1)", f"{compute_moments(sim['c'])['autocorr_lag1']:.3f}")
                with col2:
                    st.metric("Mean Labor", f"{np.mean(sim['l']):.3f}")
                    st.metric("Var Labor", f"{np.var(sim['l']):.3f}")
                    st.metric("Autocorr l(1)", f"{compute_moments(sim['l'])['autocorr_lag1']:.3f}")
                with col3:
                    st.metric("Mean Wage", f"{np.mean(sim['w']):.3f}")
                    st.metric("Var Wage", f"{np.var(sim['w']):.3f}")
                    st.metric("Autocorr w(1)", f"{compute_moments(sim['w'])['autocorr_lag1']:.3f}")

                st.markdown("**Correlation with Wage / Income**")
                col1, col2 = st.columns(2)
                with col1:
                    corr_cw = compute_correlations(sim['c'], sim['w'], "Consumption", "Wage")
                    corr_lw = compute_correlations(sim['l'], sim['w'], "Labor", "Wage")
                    st.metric("Corr(c, w)", f"{corr_cw['correlation']:.3f}")
                    st.metric("Corr(l, w)", f"{corr_lw['correlation']:.3f}")
                with col2:
                    corr_cy = compute_correlations(sim['c'], sim['y'], "Consumption", "Income")
                    corr_ly = compute_correlations(sim['l'], sim['y'], "Labor", "Income")
                    st.metric("Corr(c, y)", f"{corr_cy['correlation']:.3f}")
                    st.metric("Corr(l, y)", f"{corr_ly['correlation']:.3f}")
                
                # Quick analysis section
                st.markdown("---")
                st.markdown("**Quick Analysis (see 'Analysis' tab for detailed distributions):**")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Labor Supply & Income Analysis:**")
                    l_med = np.median(sim['l'])
                    l_low = np.min(sim['l'])
                    l_high = np.max(sim['l'])
                    w_med = np.median(sim['w'][:-1])
                    w_std = np.std(sim['w'][:-1])
                    st.write(f"Workers supply roughly {l_med:.3f} units of labor on average (e.g. hours per period). The lightest workload is {l_low:.3f} and the heaviest is {l_high:.3f}. Received wages average {w_med:.3f} currency‑units, with typical variation of {w_std:.3f}.")
                with col2:
                    st.write("**Consumption & Savings Analysis:**")
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
            st.subheader("Summary Statistics & Moments")
            
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
            st.subheader("Forecast Panel (AR(1) Model)")
            
            if 'ls_sim' in st.session_state:
                sim = st.session_state.ls_sim
                
                # Forecast labor  supply
                st.write(f"**Labor Supply Forecast ({forecast_horizon} periods ahead)**")
                l_forecast, l_std = forecast_ar1(sim['l'], periods_ahead=int(forecast_horizon))
                fig_l = plot_forecast(sim['l'][-100:], l_forecast, l_std, 
                                     title="Labor Supply Forecast", series_name="Labor Supply")
                st.plotly_chart(fig_l, use_container_width=True)
                
                # Forecast assets
                st.write(f"**Asset Forecast ({forecast_horizon} periods ahead)**")
                a_forecast, a_std = forecast_ar1(sim['a'][:-1], periods_ahead=int(forecast_horizon))
                fig_a = plot_forecast(sim['a'][:-1][-100:], a_forecast, a_std,
                                     title="Asset Forecast", series_name="Assets")
                st.plotly_chart(fig_a, use_container_width=True)
                
                # Forecast consumption
                st.write(f"**Consumption Forecast ({forecast_horizon} periods ahead)**")
                c_forecast, c_std = forecast_ar1(sim['c'], periods_ahead=int(forecast_horizon))
                fig_c = plot_forecast(sim['c'][-100:], c_forecast, c_std,
                                     title="Consumption Forecast", series_name="Consumption")
                st.plotly_chart(fig_c, use_container_width=True)
            else:
                st.info("Run a simulation first to see forecasts")
        
        with tab7:
            st.subheader("Download Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv_policies = export_policies_to_csv(
                    model.a_grid,
                    {'savings': model.policy_a, 'labor': model.policy_l, 'consumption': model.policy_c},
                    'ls'
                )
                st.download_button(
                    label="Download Policies (CSV)",
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
                    label="Download Summary (JSON)",
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
