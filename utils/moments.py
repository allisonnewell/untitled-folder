"""
Economic moments and forecasting utilities
Compute summary statistics, autocorrelations, and generate forecasts
"""

import numpy as np
from scipy import stats


def compute_moments(series, name="Series", lag=1):
    """
    Compute mean, variance, and autocorrelation for a time series.
    
    Parameters:
    -----------
    series : array-like, time series data
    name : str, name of the series for display
    lag : int, lag for autocorrelation (default 1)
    
    Returns:
    --------
    dict with 'mean', 'variance', 'std_dev', 'autocorr', 'name'
    """
    x = np.asarray(series).flatten()
    x = x[~np.isnan(x)]  # Remove NaN
    
    mean = np.mean(x)
    variance = np.var(x)
    std_dev = np.std(x)
    
    # Autocorrelation at lag
    if len(x) > lag + 1:
        acf = np.corrcoef(x[:-lag], x[lag:])[0, 1]
    else:
        acf = np.nan
    
    return {
        'name': name,
        'mean': mean,
        'variance': variance,
        'std_dev': std_dev,
        'autocorr_lag1': acf
    }


def compute_correlations(series1, series2, name1="X", name2="Y"):
    """
    Compute correlation between two time series.
    
    Parameters:
    -----------
    series1, series2 : array-like
    name1, name2 : str, series names
    
    Returns:
    --------
    dict with correlation coefficient and p-value
    """
    x = np.asarray(series1).flatten()
    y = np.asarray(series2).flatten()
    
    # Match lengths
    min_len = min(len(x), len(y))
    x = x[:min_len]
    y = y[:min_len]
    
    # Remove NaN
    valid = ~(np.isnan(x) | np.isnan(y))
    x = x[valid]
    y = y[valid]
    
    if len(x) > 2:
        corr, pval = stats.pearsonr(x, y)
    else:
        corr, pval = np.nan, np.nan
    
    return {
        'name1': name1,
        'name2': name2,
        'correlation': corr,
        'p_value': pval
    }


def forecast_ar1(series, periods_ahead=12, burn_in=50):
    """
    Generate AR(1) forecast using last observation and estimated autocorrelation.
    
    Parameters:
    -----------
    series : array-like, time series to forecast from
    periods_ahead : int, number of periods to forecast
    burn_in : int, periods to skip at start for AR(1) estimation
    
    Returns:
    --------
    tuple (forecast_values, forecast_std_error)
    """
    x = np.asarray(series).flatten()
    x = x[~np.isnan(x)]
    
    if len(x) < burn_in + 2:
        return np.full(periods_ahead, np.mean(x)), np.full(periods_ahead, np.nan)
    
    # Use latter part of series for AR(1) estimation
    x_est = x[burn_in:]
    
    # Estimate AR(1): x_t = c + rho * x_{t-1} + eps
    X = x_est[:-1]
    Y = x_est[1:]
    
    # OLS regression
    X_design = np.column_stack([np.ones(len(X)), X])
    beta = np.linalg.lstsq(X_design, Y, rcond=None)[0]
    
    const = beta[0]
    rho = beta[1]
    
    # Residual std dev
    residuals = Y - (const + rho * X)
    sigma = np.std(residuals)
    
    # Forecast
    forecast = np.zeros(periods_ahead)
    forecast_std = np.zeros(periods_ahead)
    
    x_last = x_est[-1]
    
    for h in range(periods_ahead):
        x_next = const + rho * x_last
        forecast[h] = x_next
        forecast_std[h] = sigma * np.sqrt(1 + rho**(2*h))
        x_last = x_next
    
    return forecast, forecast_std


def get_simulation_summary(sim_dict, output_key=None):
    """
    Compute a comprehensive summary of simulation moments.
    
    Parameters:
    -----------
    sim_dict : dict, simulation output with keys like 'c', 'a', 'y', 'k', 'l', 'w'
    output_key : str, optional key for output variable (default: auto-detect)
    
    Returns:
    --------
    dict with moments for each series
    """
    summary = {}
    
    # Standard keys for each model
    key_mapping = {
        'c': 'Consumption',
        'a': 'Assets',
        'y': 'Income/Income Shock',
        'k': 'Capital',
        'l': 'Labor Supply',
        'w': 'Wage',
        'output': 'Output',
        'investment': 'Investment',
        'z': 'Productivity Shock'
    }
    
    for key in sim_dict:
        if key in key_mapping:
            series = sim_dict[key]
            # Skip last element for accumulated state variables
            if key in ['a', 'k', 'w', 'y', 'z']:
                series = series[:-1] if len(series.shape) == 1 else series[:-1, :]
            
            moments = compute_moments(series, name=key_mapping[key])
            summary[key] = moments
    
    # Compute key correlations with output
    if output_key is None:
        if 'output' in sim_dict:
            output_key = 'output'
        elif 'y' in sim_dict:
            output_key = 'y'
        else:
            output_key = None
    
    if output_key is not None:
        output = sim_dict[output_key]
        if len(output.shape) > 1:
            output = output[:, 0]
        
        for key in ['c', 'a', 'k', 'l']:
            if key in sim_dict and key != output_key:
                series = sim_dict[key]
                if len(series.shape) > 1:
                    series = series[:, 0]
                
                # Trim to same length
                min_len = min(len(output), len(series))
                corr_info = compute_correlations(
                    output[:min_len],
                    series[:min_len],
                    name1='Output',
                    name2=key_mapping.get(key, key)
                )
                summary[f'{key}_output_corr'] = corr_info
    
    return summary


def format_moments_for_display(summary):
    """
    Format moments summary for Streamlit display.
    
    Parameters:
    -----------
    summary : dict, output from get_simulation_summary()
    
    Returns:
    --------
    formatted string for display
    """
    lines = []
    
    for key, info in summary.items():
        if isinstance(info, dict):
            if 'mean' in info:  # Moments blob
                lines.append(f"\n**{info['name']}**")
                lines.append(f"  • Mean: {info['mean']:.4f}")
                lines.append(f"  • Variance: {info['variance']:.4f}")
                lines.append(f"  • Std. Dev: {info['std_dev']:.4f}")
                if not np.isnan(info['autocorr_lag1']):
                    lines.append(f"  • Autocorr(1): {info['autocorr_lag1']:.4f}")
            elif 'correlation' in info:  # Correlation blob
                if not np.isnan(info['correlation']):
                    lines.append(f"\n**{info['name1']} vs {info['name2']}**")
                    lines.append(f"  • Correlation: {info['correlation']:.4f}")
                    if not np.isnan(info['p_value']):
                        lines.append(f"  • p-value: {info['p_value']:.4f}")
    
    return "\n".join(lines)
