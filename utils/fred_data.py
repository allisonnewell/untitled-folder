"""
FRED Data Integration
Fetches economic data from Federal Reserve Economic Data (FRED) API
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# FRED series definitions
FRED_SERIES = {
    # Interest Rates
    "Federal Funds Rate": {
        "series_id": "FEDFUNDS",
        "use": "Interest Rate",
        "transformation": "level"
    },
    "Real Interest Rate (10-yr)": {
        "series_id": "DFEDTARU",
        "use": "Interest Rate",
        "transformation": "level"
    },
    "3-Month Treasury Bill": {
        "series_id": "TB3MS",
        "use": "Interest Rate",
        "transformation": "level"
    },
    
    # Production and Growth
    "Real GDP per Capita": {
        "series_id": "A939RX0Q048SBEA",
        "use": "Production",
        "transformation": "log difference"
    },
    "Real GDP Growth": {
        "series_id": "A191RL1Q225SBEA",
        "use": "Production",
        "transformation": "level"
    },
    "Total Factor Productivity": {
        "series_id": "RTFPNAUSA632N",
        "use": "Productivity",
        "transformation": "log difference"
    },
    "Labor Productivity": {
        "series_id": "OPHNFB",
        "use": "Productivity",
        "transformation": "log difference"
    },
    
    # Capital and Investment
    "Private Fixed Investment": {
        "series_id": "GPDI",
        "use": "Investment",
        "transformation": "log difference"
    },
    "Capital Stock": {
        "series_id": "KSTOCK",
        "use": "Capital",
        "transformation": "log difference"
    },
    "Net Investment": {
        "series_id": "W790RC1Q027SBEA",
        "use": "Investment",
        "transformation": "log difference"
    },
    
    # Labor Market
    "Unemployment Rate": {
        "series_id": "UNRATE",
        "use": "Labor",
        "transformation": "level"
    },
    "Labor Force Participation": {
        "series_id": "CIVPART",
        "use": "Labor",
        "transformation": "level"
    },
    "Average Hourly Earnings": {
        "series_id": "CES0500000003",
        "use": "Wages",
        "transformation": "log difference"
    },
    "Real Wage Index": {
        "series_id": "AWHNONAG",
        "use": "Wages",
        "transformation": "log difference"
    },
    "Employment-Population Ratio": {
        "series_id": "EMRATIO",
        "use": "Labor",
        "transformation": "level"
    },
    
    # Consumption and Income
    "Personal Consumption Expenditures": {
        "series_id": "PCE",
        "use": "Consumption",
        "transformation": "log difference"
    },
    "Real Disposable Income": {
        "series_id": "DSPIC96",
        "use": "Income",
        "transformation": "log difference"
    },
    "Personal Income": {
        "series_id": "W875RX1",
        "use": "Income",
        "transformation": "log difference"
    },
    
    # Inflation and Prices
    "CPI Inflation": {
        "series_id": "CPIAUCSL",
        "use": "Inflation",
        "transformation": "difference"
    },
    "Core PCE Inflation": {
        "series_id": "PCEPILFE",
        "use": "Inflation",
        "transformation": "difference"
    },
    
    # Business Cycle Indicators
    "Industrial Production": {
        "series_id": "INDPRO",
        "use": "Production",
        "transformation": "log difference"
    },
    "Capacity Utilization": {
        "series_id": "TCU",
        "use": "Production",
        "transformation": "level"
    }
}


class FREDDataFetcher:
    """Fetch and process FRED economic data"""
    
    def __init__(self, api_key=None):
        """
        Initialize FRED fetcher
        
        Parameters:
        -----------
        api_key : str, optional FRED API key
                  If None, uses mock data
        """
        self.api_key = api_key
        self.use_mock_data = api_key is None
        self.cache = {}
        self.n_obs = 60  # Reduced from 120 for faster generation
    
    def fetch_series(self, series_id):
        """
        Fetch time series from FRED
        
        Parameters:
        -----------
        series_id : str, FRED series ID
        
        Returns:
        --------
        array : time series data
        """
        # Check cache first
        if series_id in self.cache:
            return self.cache[series_id]
        
        if self.use_mock_data:
            data = self._generate_mock_data(series_id)
            self.cache[series_id] = data  # Cache the result
            return data
        
        # If API key provided, would fetch from FRED here
        # For now, return mock data
        data = self._generate_mock_data(series_id)
        self.cache[series_id] = data
        return data
    
    def _generate_mock_data(self, series_id):
        """Generate realistic mock data for testing"""
        
        np.random.seed(hash(series_id) % 2**32)
        n_obs = self.n_obs  # Use instance variable
        
        # Interest rates (in percent)
        if series_id in ["FEDFUNDS", "TB3MS"]:
            data = np.cumsum(np.random.normal(0, 0.2, n_obs)) + 2.5
            data = np.clip(data, 0.1, 5.0)
        elif series_id == "DFEDTARU":
            data = np.cumsum(np.random.normal(0, 0.15, n_obs)) + 1.5
            data = np.clip(data, -1.0, 4.0)
        
        # Production/Growth (quarterly rates)
        elif series_id in ["A939RX0Q048SBEA", "RTFPNAUSA632N", "OPHNFB", "GPDI", "KSTOCK", "W790RC1Q027SBEA"]:
            data = np.cumsum(np.random.normal(0.002, 0.005, n_obs)) + 10.0
        
        # GDP Growth (quarterly percent)
        elif series_id == "A191RL1Q225SBEA":
            data = np.cumsum(np.random.normal(0, 0.5, n_obs)) + 2.0
            data = np.clip(data, -5.0, 8.0)
        
        # Labor market (percent)
        elif series_id == "UNRATE":
            data = np.cumsum(np.random.normal(0, 0.1, n_obs)) + 5.0
            data = np.clip(data, 3.0, 10.0)
        elif series_id in ["CIVPART", "EMRATIO"]:
            data = np.cumsum(np.random.normal(0, 0.05, n_obs)) + 63.0
            data = np.clip(data, 58.0, 68.0)
        
        # Wages (hourly earnings)
        elif series_id in ["CES0500000003", "AWHNONAG"]:
            data = np.cumsum(np.random.normal(0.003, 0.005, n_obs)) + 25.0
        
        # Consumption/Income (billions/trillions)
        elif series_id in ["PCE", "DSPIC96", "W875RX1"]:
            data = np.cumsum(np.random.normal(0.01, 0.02, n_obs)) + 15.0
        
        # Inflation (monthly percent)
        elif series_id in ["CPIAUCSL", "PCEPILFE"]:
            data = np.cumsum(np.random.normal(0, 0.1, n_obs)) + 2.5
            data = np.clip(data, -2.0, 8.0)
        
        # Industrial production/capacity (index)
        elif series_id == "INDPRO":
            data = np.cumsum(np.random.normal(0.001, 0.008, n_obs)) + 105.0
        elif series_id == "TCU":
            data = np.cumsum(np.random.normal(0, 0.5, n_obs)) + 78.0
            data = np.clip(data, 70.0, 85.0)
        
        else:
            # Default random walk
            data = np.cumsum(np.random.normal(0.002, 0.01, n_obs))
        
        return data
    
    def estimate_parameters(self, series_id):
        """
        Estimate model parameters from time series
        
        Returns dict with:
        - level_mean: mean of the series
        - std: standard deviation
        - rho: AR(1) coefficient
        """
        
        data = self.fetch_series(series_id)
        
        # Remove perfect collinearity
        data = np.asarray(data).flatten()
        data = data[~np.isnan(data)]
        
        if len(data) < 2:
            return {
                'level_mean': 0.05,
                'std': 0.02,
                'rho': 0.9,
                'series_id': series_id
            }
        
        # AR(1) estimation
        y = data[1:]
        X = data[:-1]
        
        # Add small noise to avoid perfect collinearity
        X = X + np.random.normal(0, 1e-6, len(X))
        
        # Simple AR(1) coefficient
        if np.std(X) > 1e-8:
            rho = np.corrcoef(X, y)[0, 1]
            rho = np.clip(rho, 0.0, 0.99)
        else:
            rho = 0.9
        
        # Calculate residual std
        residuals = y - rho * X
        sigma = np.std(residuals)
        
        return {
            'level_mean': float(np.mean(data)),
            'std': float(sigma),
            'rho': float(rho),
            'series_id': series_id
        }
    
    def calibrate_model_parameters(self, model_type, series_selections):
        """
        Calibrate model parameters based on selected FRED series
        
        Parameters:
        -----------
        model_type : str, 'cs', 'rc', or 'ls'
        series_selections : dict, mapping parameter names to series names
        
        Returns:
        --------
        dict : calibrated parameters
        """
        calibrated = {}
        
        for param, series_name in series_selections.items():
            if series_name == "None" or series_name not in FRED_SERIES:
                continue
                
            try:
                params = self.estimate_parameters(FRED_SERIES[series_name]['series_id'])
                
                if model_type == 'cs':  # Consumption-Savings
                    if param == 'r':
                        calibrated['r'] = max(0.01, min(0.10, params['level_mean'] / 100))
                    elif param == 'rho':
                        calibrated['rho'] = max(0.0, min(0.99, params['rho']))
                    elif param == 'sigma_y':
                        calibrated['sigma_y'] = max(0.01, min(0.5, params['std']))
                        
                elif model_type == 'rc':  # Robinson Crusoe
                    if param == 'alpha':
                        # Capital share from productivity data
                        calibrated['alpha'] = max(0.25, min(0.40, params['level_mean'] / 100))
                    elif param == 'delta':
                        # Depreciation from investment data (rough estimate)
                        calibrated['delta'] = max(0.05, min(0.15, params['std'] * 2))
                    elif param == 'rho':
                        calibrated['rho'] = max(0.0, min(0.99, params['rho']))
                    elif param == 'sigma_z':
                        calibrated['sigma_z'] = max(0.01, min(0.3, params['std']))
                        
                elif model_type == 'ls':  # Labor Supply
                    if param == 'r':
                        calibrated['r'] = max(0.01, min(0.10, params['level_mean'] / 100))
                    elif param == 'rho':
                        calibrated['rho'] = max(0.0, min(0.99, params['rho']))
                    elif param == 'sigma_w':
                        calibrated['sigma_w'] = max(0.01, min(0.3, params['std']))
                    elif param == 'chi':
                        # Labor disutility from unemployment/participation
                        calibrated['chi'] = max(0.5, min(3.0, params['level_mean'] / 20))
                    elif param == 'eta':
                        # Frisch elasticity from wage variability
                        calibrated['eta'] = max(0.3, min(1.5, 1.0 / (1.0 + params['std'])))
                        
            except Exception as e:
                print(f"Warning: Could not calibrate {param} from {series_name}: {str(e)}")
                
        return calibrated


def get_sample_calibration(param_source='Default'):
    """
    Get sample calibrations for different sources
    
    Parameters:
    -----------
    param_source : str, 'Default', 'Conservative', 'Historical Average', 'Custom FRED Data'
    
    Returns:
    --------
    dict : calibrations for all three models
    """
    
    calibrations = {
        'Default': {
            'cs': {
                'beta': 0.95,
                'r': 0.05,
                'gamma': 2.0,
                'rho': 0.90,
                'sigma_y': 0.10
            },
            'rc': {
                'beta': 0.95,
                'alpha': 0.33,
                'delta': 0.10,
                'gamma': 2.0,
                'rho': 0.90,
                'sigma_z': 0.02
            },
            'ls': {
                'beta': 0.95,
                'r': 0.05,
                'gamma': 2.0,
                'chi': 1.0,
                'eta': 0.5,
                'rho': 0.90,
                'sigma_w': 0.10
            }
        },
        'Conservative': {
            'cs': {
                'beta': 0.98,
                'r': 0.03,
                'gamma': 1.5,
                'rho': 0.95,
                'sigma_y': 0.05
            },
            'rc': {
                'beta': 0.98,
                'alpha': 0.30,
                'delta': 0.08,
                'gamma': 1.5,
                'rho': 0.95,
                'sigma_z': 0.01
            },
            'ls': {
                'beta': 0.98,
                'r': 0.03,
                'gamma': 1.5,
                'chi': 1.5,
                'eta': 0.3,
                'rho': 0.95,
                'sigma_w': 0.05
            }
        },
        'Historical Average': {
            'cs': {
                'beta': 0.96,
                'r': 0.025,
                'gamma': 2.5,
                'rho': 0.92,
                'sigma_y': 0.08
            },
            'rc': {
                'beta': 0.96,
                'alpha': 0.35,
                'delta': 0.05,
                'gamma': 2.5,
                'rho': 0.92,
                'sigma_z': 0.015
            },
            'ls': {
                'beta': 0.96,
                'r': 0.025,
                'gamma': 2.5,
                'chi': 0.8,
                'eta': 0.7,
                'rho': 0.92,
                'sigma_w': 0.08
            }
        }
    }
    
    return calibrations.get(param_source, calibrations['Default'])
