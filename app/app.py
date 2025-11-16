"""
SC Labs - Supply Chain Optimization Dashboard
Main application entry point
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Add parent directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path)

import streamlit as st
import pandas as pd
import numpy as np

from utils.config import get_config
from auth import AuthManager, show_user_menu
from navigation import navigate_to, show_navigation_bar
from pages import (
    show_landing_page,
    show_data_page,
    show_forecast_page,
    show_inventory_page,
    show_routing_page
)

# Page configuration
st.set_page_config(
    page_title="Planner - Supply Chain Optimization",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load custom CSS
def load_css():
    css_file = Path(__file__).parent / "styles.css"
    with open(css_file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Initialize authentication
auth_manager = AuthManager()
auth_manager.require_auth()
show_user_menu(auth_manager)

# Sidebar separator
with st.sidebar:
    st.markdown("---")

# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    # Page state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'landing'
    
    # Module completion flags
    if 'data_generated' not in st.session_state:
        st.session_state.data_generated = False
    if 'forecasts_generated' not in st.session_state:
        st.session_state.forecasts_generated = False
    if 'inventory_optimized' not in st.session_state:
        st.session_state.inventory_optimized = False
    if 'routes_optimized' not in st.session_state:
        st.session_state.routes_optimized = False
    
    # Load configuration from config.yaml
    config = get_config()
    
    # Configuration defaults
    if 'n_stores' not in st.session_state:
        st.session_state.n_stores = config.get('data_generation', 'default_n_stores', default=3)
    if 'n_products' not in st.session_state:
        st.session_state.n_products = config.get('data_generation', 'default_n_products', default=2)
    if 'n_weeks' not in st.session_state:
        st.session_state.n_weeks = config.get('data_generation', 'default_n_weeks', default=52)
    if 'forecast_horizon' not in st.session_state:
        st.session_state.forecast_horizon = config.get('forecasting', 'default_forecast_horizon', default=4)
    if 'planning_horizon' not in st.session_state:
        st.session_state.planning_horizon = config.get('inventory', 'default_planning_horizon', default=8)
    if 'service_level' not in st.session_state:
        st.session_state.service_level = config.get('inventory', 'default_service_level', default=0.95)
    if 'lead_time' not in st.session_state:
        st.session_state.lead_time = config.get('inventory', 'default_lead_time', default=2)
    if 'review_period' not in st.session_state:
        st.session_state.review_period = config.get('inventory', 'default_review_period', default=1)
    if 'max_payload' not in st.session_state:
        st.session_state.max_payload = config.get('routing', 'default_max_payload', default=100)
    if 'n_customers' not in st.session_state:
        st.session_state.n_customers = config.get('routing', 'default_n_customers', default=30)

init_session_state()

# Main app router
if st.session_state.current_page == 'landing':
    show_landing_page(navigate_to)
elif st.session_state.current_page == 'data':
    show_data_page(show_navigation_bar, navigate_to)
elif st.session_state.current_page == 'forecast':
    show_forecast_page(show_navigation_bar, navigate_to)
elif st.session_state.current_page == 'inventory':
    show_inventory_page(show_navigation_bar, navigate_to)
elif st.session_state.current_page == 'routing':
    show_routing_page(show_navigation_bar, navigate_to)
else:
    show_landing_page(navigate_to)
