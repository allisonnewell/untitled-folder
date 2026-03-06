"""
Visualization Functions for Macroeconomic Models Dashboard
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px


def plot_policy_function(grid, policy, title="Policy Function", 
                         state_label="State Variable", action_label="Policy Variable",
                         shock_labels=None,
                         max_legend_items=None,
                         **kwargs):
    """Plot policy function with multiple shocks
    
    Parameters:
    -----------
    grid : array-like, state variable grid
    policy : array-like, policy function values
    title : str, plot title
    state_label : str, x-axis label
    action_label : str, y-axis label  
    shock_labels : list, optional labels for shock states
    max_legend_items : int or None, maximum number of items to display in legend (extra curves hidden)
    """
    
    fig = go.Figure()
    
    # Handle multiple shock states
    if policy.ndim == 2:
        n_shocks = policy.shape[1]
        colors = px.colors.qualitative.Plotly[:n_shocks]
        
        # Default shock labels if not provided
        if shock_labels is None:
            shock_labels = [f'Income State {i+1}' for i in range(n_shocks)]
        
        for i in range(n_shocks):
            label = shock_labels[i] if i < len(shock_labels) else f'State {i+1}'
            show_leg = True
            if max_legend_items is not None and i >= max_legend_items:
                # hide extra items from legend to keep key concise
                show_leg = False
                label_tooltip = label + ' (hidden in legend)'
            else:
                label_tooltip = label
            fig.add_trace(go.Scatter(
                x=grid, y=policy[:, i],
                mode='lines',
                name=label if show_leg else None,
                showlegend=show_leg,
                line=dict(color=colors[i], width=2.5),
                hovertemplate=f'<b>{state_label}</b>: %{{x:.2f}}<br>' +
                             f'<b>{action_label}</b>: %{{y:.3f}}<extra>{label_tooltip}</extra>'
            ))
    else:
        fig.add_trace(go.Scatter(
            x=grid, y=policy,
            mode='lines',
            name=action_label,
            line=dict(color='#1f77b4', width=2.5),
            hovertemplate=f'<b>{state_label}</b>: %{{x:.2f}}<br>' +
                         f'<b>{action_label}</b>: %{{y:.3f}}<extra></extra>'
        ))
    
    # 45-degree line if applicable (for savings/capital policy)
    if any(keyword in title.lower() for keyword in ['savings', 'capital', 'assets']):
        fig.add_trace(go.Scatter(
            x=grid, y=grid,
            mode='lines',
            name='45° Line (No Change)',
            line=dict(color='red', width=1.5, dash='dash'),
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(family='Garamond, serif', size=14, color='#1f77b4')
        ),
        xaxis_title=dict(
            text=state_label,
            font=dict(family='Garamond, serif', size=12)
        ),
        yaxis_title=dict(
            text=action_label,
            font=dict(family='Garamond, serif', size=12)
        ),
        xaxis=dict(showline=False, showgrid=False, zeroline=False, mirror=False, showspikes=False),
        yaxis=dict(showline=False, showgrid=False, zeroline=False, mirror=False, showspikes=False),
        hovermode='x unified',
        height=400,
        showlegend=True,
        font=dict(family='Garamond, serif', size=11, color='#ffffff'),
        margin=dict(l=50, r=30, t=50, b=50),
        paper_bgcolor='#001a33',
        plot_bgcolor='#001a33'
    )
    
    return fig


def plot_value_function(grid, value_fn, title="Value Function",
                       state_label="State"):
    """Plot value function"""
    
    fig = go.Figure()
    
    if value_fn.ndim == 2:
        n_shocks = value_fn.shape[1]
        colors = px.colors.qualitative.Plotly[:n_shocks]
        
        for i in range(n_shocks):
            fig.add_trace(go.Scatter(
                x=grid, y=value_fn[:, i],
                mode='lines',
                name=f'Shock State {i+1}',
                line=dict(color=colors[i], width=2.5),
                fill='tozeroy' if i == 0 else None
            ))
    else:
        fig.add_trace(go.Scatter(
            x=grid, y=value_fn,
            mode='lines+markers',
            line=dict(color='#1f77b4', width=2.5),
            fill='tozeroy',
            marker=dict(size=3)
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title=state_label,
        yaxis_title='Value',
        xaxis=dict(showline=False, showgrid=False, zeroline=False, mirror=False, showspikes=False),
        yaxis=dict(showline=False, showgrid=False, zeroline=False, mirror=False, showspikes=False),
        height=350,
        hovermode='x unified',
        font=dict(family='Garamond, serif', size=10, color='#ffffff'),
        margin=dict(l=50, r=30, t=40, b=40),
        paper_bgcolor='#001a33',
        plot_bgcolor='#001a33'
    )
    
    return fig


def plot_heatmap(x_grid, y_grid, heatmap_data, title="Heatmap",
                x_label="X", y_label="Y"):
    """Plot 2D heatmap"""
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=x_grid,
        y=y_grid,
        colorscale='Viridis',
        hovertemplate='<b>' + x_label + '</b>: %{x:.3f}<br>' +
                     '<b>' + y_label + '</b>: %{y:.3f}<br>' +
                     '<b>Value</b>: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=450,
        font=dict(family='Garamond, serif', size=11, color='#ffffff'),
        margin=dict(l=60, r=40, t=60, b=60),
        xaxis=dict(showline=False, showgrid=False, zeroline=False, mirror=False, showspikes=False),
        yaxis=dict(showline=False, showgrid=False, zeroline=False, mirror=False, showspikes=False),
        paper_bgcolor='#001a33',
        plot_bgcolor='#001a33'
    )
    
    return fig


def plot_multiple_series(time_index, data_dict, title="Time Series"):
    """
    Plot multiple time series
    
    Parameters:
    -----------
    time_index : array-like, time indices
    data_dict : dict, {series_name: data_array}
    title : str
    """
    
    fig = go.Figure()
    
    colors = px.colors.qualitative.Plotly
    for idx, (name, data) in enumerate(data_dict.items()):
        color = colors[idx % len(colors)]
        fig.add_trace(go.Scatter(
            x=time_index,
            y=data,
            mode='lines',
            name=name,
            line=dict(color=color, width=2),
            hovertemplate='<b>Period</b>: %{x}<br>' +
                         f'<b>{name}</b>: %{{y:.3f}}<extra></extra>'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Time Period',
        yaxis_title='Value',
        hovermode='x unified',
        height=400,
        showlegend=True,
        font=dict(family='Garamond, serif', size=10, color='#ffffff'),
        xaxis=dict(showline=False, showgrid=False, zeroline=False, mirror=False, showspikes=False),
        yaxis=dict(showline=False, showgrid=False, zeroline=False, mirror=False, showspikes=False),
        margin=dict(l=50, r=30, t=40, b=40),
        paper_bgcolor='#001a33',
        plot_bgcolor='#001a33'
    )
    
    return fig


def plot_distribution(data, title="Distribution", bins=40):
    """Plot histogram of data"""
    
    fig = go.Figure(data=[
        go.Histogram(
            x=data,
            nbinsx=bins,
            marker=dict(color='#1f77b4', opacity=0.7),
            hovertemplate='<b>Range</b>: %{x}<br>' +
                         '<b>Frequency</b>: %{y}<extra></extra>'
        )
    ])
    
    # Add mean line
    mean_val = np.mean(data)
    fig.add_vline(x=mean_val, line_dash="dash", line_color="red",
                 annotation_text=f"Mean: {mean_val:.3f}",
                 annotation_position="top right")
    
    fig.update_layout(
        title=title,
        xaxis_title='Value',
        yaxis_title='Frequency',
        height=350,
        showlegend=False,
        font=dict(family='Garamond, serif', size=11, color='#ffffff'),
        margin=dict(l=50, r=30, t=40, b=40),
        xaxis=dict(showline=False, showgrid=False, zeroline=False, mirror=False, showspikes=False),
        yaxis=dict(showline=False, showgrid=False, zeroline=False, mirror=False, showspikes=False),
        paper_bgcolor='#001a33',
        plot_bgcolor='#001a33'
    )
    
    return fig


def plot_simulated_path(time_index, data_dict, title="Simulated Path"):
    """
    Alias for plot_multiple_series - plots simulated paths
    
    Parameters:
    -----------
    time_index : array-like, time indices
    data_dict : dict, {series_name: data_array}
    title : str
    """
    return plot_multiple_series(time_index, data_dict, title)


def plot_correlation_heatmap(correlation_matrix, labels, title="Correlation Matrix"):
    """Plot correlation matrix heatmap"""
    
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix,
        x=labels,
        y=labels,
        colorscale='RdBu',
        zmid=0,
        zmin=-1,
        zmax=1,
        hovertemplate='<b>%{x}</b> vs <b>%{y}</b><br>Corr: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        height=500,
        font=dict(family='Garamond, serif', size=10, color='#ffffff'),
        margin=dict(l=50, r=30, t=40, b=40),
        xaxis=dict(showline=False, showgrid=False, zeroline=False, mirror=False, showspikes=False),
        yaxis=dict(showline=False, showgrid=False, zeroline=False, mirror=False, showspikes=False),
        paper_bgcolor='#001a33',
        plot_bgcolor='#001a33'
    )
    
    return fig


def plot_forecast(historical, forecast, forecast_std, title="Forecast", series_name="Series"):
    """Plot historical time series with forecast and confidence interval
    
    Parameters:
    -----------
    historical : array, historical data
    forecast : array, forecasted values
    forecast_std : array, forecast standard errors
    title : str, plot title
    series_name : str, name of the series
    """
    
    n_hist = len(historical)
    n_fore = len(forecast)
    
    # Time indices
    hist_time = np.arange(n_hist)
    fore_time = np.arange(n_hist - 1, n_hist + n_fore)
    
    fig = go.Figure()
    
    # Historical series
    fig.add_trace(go.Scatter(
        x=hist_time,
        y=historical,
        mode='lines',
        name='Historical',
        line=dict(color='#1f77b4', width=2.5),
        hovertemplate='<b>Period</b>: %{x}<br><b>Value</b>: %{y:.4f}<extra></extra>'
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=fore_time,
        y=forecast,
        mode='lines+markers',
        name='Forecast',
        line=dict(color='#ff7f0e', width=2, dash='dash'),
        marker=dict(size=6),
        hovertemplate='<b>Period</b>: %{x}<br><b>Forecast</b>: %{y:.4f}<extra></extra>'
    ))
    
    # Confidence interval (±1 std dev)
    upper_ci = forecast + forecast_std
    lower_ci = forecast - forecast_std
    
    fig.add_trace(go.Scatter(
        x=fore_time,
        y=upper_ci,
        mode='none',
        name='Upper CI (±1σ)',
        hovertemplate='<b>Upper</b>: %{y:.4f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=fore_time,
        y=lower_ci,
        mode='none',
        name='Lower CI (±1σ)',
        fill='tonexty',
        fillcolor='rgba(255, 127, 14, 0.2)',
        line=dict(width=0),
        hovertemplate='<b>Lower</b>: %{y:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(family='Garamond, serif', size=14, color='#1f77b4')
        ),
        xaxis_title=dict(
            text='Period',
            font=dict(family='Garamond, serif', size=12)
        ),
        yaxis_title=dict(
            text=series_name,
            font=dict(family='Garamond, serif', size=12)
        ),
        xaxis=dict(showline=False, showgrid=False, zeroline=False, mirror=False, showspikes=False),
        yaxis=dict(showline=False, showgrid=False, zeroline=False, mirror=False, showspikes=False),
        hovermode='x unified',
        height=400,
        font=dict(family='Garamond, serif', size=11, color='#ffffff'),
        margin=dict(l=50, r=30, t=50, b=50),
        paper_bgcolor='#001a33',
        plot_bgcolor='#001a33'
    )
    
    return fig
