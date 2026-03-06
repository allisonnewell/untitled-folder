#!/bin/bash
# Run script for Macroeconomic Models Dashboard
# This script configures the app for different deployment environments

# Set default port
PORT=${PORT:-8501}

# Export environment variables for Streamlit
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_SERVER_PORT=$PORT
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Run the Streamlit app
streamlit run app.py --server.address 0.0.0.0 --server.port $PORT --server.headless true