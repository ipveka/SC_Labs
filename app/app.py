"""
SC Labs - Supply Chain Optimization Dashboard
Beautiful landing page with module navigation
"""

import sys
import os
from pathlib import Path

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

from auxiliar.auxiliar import generate_data
from forecaster.forecaster import Forecaster
from optimizer.optimizer import Optimizer
from router.router import Router
from utils.output_manager import OutputManager
from utils.config import get_config
from auth import AuthManager, show_user_menu
from app_utils import *

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

# Require authentication (shows login page if not authenticated)
auth_manager.require_auth()

# Show user menu in sidebar (only after login)
show_user_menu(auth_manager)

# Sidebar separator (removed system info)
with st.sidebar:
    st.markdown("---")

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'landing'
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

# Configuration defaults from config.yaml
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

# Navigation functions
def navigate_to(page):
    st.session_state.current_page = page
    st.rerun()

def show_footer():
    """Display footer on all pages"""
    st.markdown("""
    <div style='text-align: center; background: white; color: #2d3748; padding: 2rem; margin-top: 3rem; border-top: 1px solid #e2e8f0;'>
        <p style='font-size: 1rem; font-weight: 500; margin: 0;'>SC Labs - Barcelona - 2025</p>
    </div>
    """, unsafe_allow_html=True)


def show_navigation_bar(current_module=None):
    """Display navigation bar with module links"""
    col1, col2, col3, col4, col5 = st.columns([2, 1.2, 1, 1, 1])
    
    with col1:
        if st.button("🏠 Home", width="stretch"):
            navigate_to('landing')
    
    with col2:
        if st.button("⚙️ Data & Settings", width="stretch", disabled=(current_module == 'data')):
            navigate_to('data')
    
    with col3:
        if st.button("📈 Forecast", width="stretch", disabled=(current_module == 'forecast')):
            navigate_to('forecast')
    
    with col4:
        if st.button("📦 Inventory", width="stretch", disabled=(current_module == 'inventory')):
            navigate_to('inventory')
    
    with col5:
        if st.button("🚚 Routing", width="stretch", disabled=(current_module == 'routing')):
            navigate_to('routing')
    
    st.markdown("---")


# ============================================================================
# LANDING PAGE
# ============================================================================
def show_landing_page():
    st.markdown("""
    <div class="hero-section" style="padding: 2rem 2rem; margin-bottom: 1.5rem;">
        <div class="hero-title" style="font-size: 4rem;">📦 Planner</div>
        <div class="hero-subtitle" style="font-size: 1.6rem;">Supply Chain Optimization Platform</div>
        <div class="hero-description" style="font-size: 1.3rem; margin-bottom: 1rem;">
            Forecast demand • Optimize inventory • Route deliveries
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<h2 style="text-align: center; color: white; margin: 1rem 0 1.5rem;">Explore Our Modules</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("""
        <div class="module-card">
            <span class="module-icon">⚙️</span>
            <div class="module-title">Data & Settings</div>
            <div class="module-description">
                Configure system parameters and load data for analysis.
            </div>
            <ul class="module-features">
                <li>✓ Upload your own data</li>
                <li>✓ Generate synthetic demo data</li>
                <li>✓ Configure all module settings</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Launch Data & Settings →", key="btn_data", width="stretch"):
            navigate_to('data')
    
    with col2:
        st.markdown("""
        <div class="module-card">
            <span class="module-icon">📈</span>
            <div class="module-title">Demand Forecasting</div>
            <div class="module-description">
                Predict future demand using LightGBM with automated feature engineering.
            </div>
            <ul class="module-features">
                <li>✓ Gradient boosting model</li>
                <li>✓ Automated features (lag, rolling, temporal)</li>
                <li>✓ No data leakage</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Launch Forecasting →", key="btn_forecast", width="stretch"):
            navigate_to('forecast')
    
    col3, col4 = st.columns(2, gap="large")
    
    with col3:
        st.markdown("""
        <div class="module-card">
            <span class="module-icon">📦</span>
            <div class="module-title">Inventory Optimization</div>
            <div class="module-description">
                Optimize stock levels using reorder point policies with safety stock.
            </div>
            <ul class="module-features">
                <li>✓ Reorder point calculation</li>
                <li>✓ Safety stock optimization</li>
                <li>✓ Stockout prevention</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Launch Inventory →", key="btn_inventory", width="stretch"):
            navigate_to('inventory')
    
    with col4:
        st.markdown("""
        <div class="module-card">
            <span class="module-icon">🚚</span>
            <div class="module-title">Delivery Routing</div>
            <div class="module-description">
                Optimize delivery routes with intelligent truck assignment.
            </div>
            <ul class="module-features">
                <li>✓ Smart truck assignment</li>
                <li>✓ Route optimization</li>
                <li>✓ Payload management</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Launch Routing →", key="btn_routing", width="stretch"):
            navigate_to('routing')
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    show_footer()


# ============================================================================
# DATA & SETTINGS PAGE
# ============================================================================
def show_data_page():
    show_navigation_bar('data')
    
    st.markdown('<div class="content-container">', unsafe_allow_html=True)
    st.markdown("## ⚙️ Data & Settings")
    st.markdown("---")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload Data", "🎲 Generate Demo Data", "⚙️ Configuration"])
    
    # ========================================================================
    # TAB 1: UPLOAD DATA
    # ========================================================================
    with tab1:
        st.markdown("### Upload Your Data")
        st.markdown("Upload a CSV file with your historical sales data to use for forecasting and optimization.")
        
        # Show format requirements
        with st.expander("📋 Required Data Format", expanded=False):
            st.markdown("""
            Your CSV file must contain the following columns:
            
            | Column | Type | Description | Required |
            |--------|------|-------------|----------|
            | `store` | string | Store identifier | ✓ Yes |
            | `product` | string | Product identifier | ✓ Yes |
            | `date` | date | Date (YYYY-MM-DD) | ✓ Yes |
            | `sales` | numeric | Sales quantity | ✓ Yes |
            | `inventory` | numeric | Inventory level | Optional |
            | `customer_id` | string | Customer ID | Optional |
            | `destination` | string | Postal code | Optional |
            
            **Example:**
            ```
            store,product,date,sales,inventory
            A,A,2024-01-07,116,202
            A,A,2024-01-14,140,445
            A,B,2024-01-07,89,150
            ```
            
            **Tips:**
            - Use weekly data for best results
            - Ensure at least 20-30 historical periods
            - Date format must be YYYY-MM-DD
            - Each store-product combination should have continuous data
            """)
        
        # Sample data download
        st.markdown("#### 📥 Download Sample Data")
        sample_path = Path(__file__).parent.parent / 'data' / 'sample_sales_data.csv'
        if sample_path.exists():
            with open(sample_path, 'rb') as f:
                st.download_button(
                    label="⬇️ Download sample_sales_data.csv",
                    data=f,
                    file_name="sample_sales_data.csv",
                    mime="text/csv",
                    help="Download a sample dataset to see the required format"
                )
        
        st.markdown("---")
        st.markdown("#### 📤 Upload Your File")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file with your sales data"
        )
        
        if uploaded_file is not None:
            try:
                # Read the uploaded file
                data = pd.read_csv(uploaded_file)
                
                # Validate required columns
                required_cols = ['store', 'product', 'date', 'sales']
                missing_cols = [col for col in required_cols if col not in data.columns]
                
                if missing_cols:
                    st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                    st.info("Please ensure your CSV has: store, product, date, sales")
                else:
                    # Convert date column
                    data['date'] = pd.to_datetime(data['date'])
                    
                    # Validate data types
                    if not pd.api.types.is_numeric_dtype(data['sales']):
                        st.error("❌ 'sales' column must contain numeric values")
                    else:
                        # Store in session state
                        st.session_state.data = data
                        st.session_state.data_generated = True
                        st.success("✅ Data uploaded successfully!")
                        
                        # Show data summary
                        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                        with col_m1:
                            st.metric("Total Records", f"{len(data):,}")
                        with col_m2:
                            st.metric("Stores", data['store'].nunique())
                        with col_m3:
                            st.metric("Products", data['product'].nunique())
                        with col_m4:
                            st.metric("Date Range", f"{(data['date'].max() - data['date'].min()).days // 7} weeks")
                        
                        st.markdown("#### Data Preview")
                        st.dataframe(data.head(20), width="stretch")
                        
                        st.markdown("#### Sales Over Time")
                        fig = create_sales_trend_chart(data)
                        st.plotly_chart(fig, width="stretch")
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        col_next1, col_next2, col_next3 = st.columns([1, 1, 1])
                        with col_next2:
                            if st.button("Next: Forecasting →", width="stretch", type="primary", key="upload_next"):
                                navigate_to('forecast')
                        
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
                st.info("Please ensure your file is a valid CSV with the correct format")
    
    # ========================================================================
    # TAB 2: GENERATE DEMO DATA
    # ========================================================================
    with tab2:
        st.markdown("### Generate Synthetic Demo Data")
        st.markdown("Create realistic synthetic data for testing and demonstration purposes.")
        
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.metric("Stores", st.session_state.n_stores)
        with col_info2:
            st.metric("Products", st.session_state.n_products)
        with col_info3:
            st.metric("Weeks", st.session_state.n_weeks)
        
        st.info("💡 Adjust these parameters in the Configuration tab")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("🎲 Generate Demo Data", key="gen_data", width="stretch", type="primary"):
            with st.spinner("Generating synthetic data..."):
                st.session_state.data = generate_data(
                    n_stores=st.session_state.n_stores,
                    n_products=st.session_state.n_products,
                    n_weeks=st.session_state.n_weeks,
                    start_date='2024-01-01',
                    seed=42
                )
                st.session_state.data_generated = True
                st.success("✅ Demo data generated successfully!")
        
        if st.session_state.data_generated:
            data = st.session_state.data
            
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("Total Records", f"{len(data):,}")
            with col_m2:
                st.metric("Avg Sales", f"{data['sales'].mean():.1f}")
            with col_m3:
                st.metric("Date Range", f"{st.session_state.n_weeks} weeks")
            with col_m4:
                st.metric("Time Series", st.session_state.n_stores * st.session_state.n_products)
            
            st.markdown("#### Sales Over Time")
            fig = create_sales_trend_chart(data)
            st.plotly_chart(fig, width="stretch")
            
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                fig = create_sales_histogram(data)
                st.plotly_chart(fig, width="stretch")
            
            with col_d2:
                fig = create_sales_boxplot(data)
                st.plotly_chart(fig, width="stretch")
            
            with st.expander("📋 View Data Sample"):
                st.dataframe(data.head(20), width="stretch")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Save button
            col_save1, col_save2, col_save3 = st.columns([1, 1, 1])
            with col_save2:
                if st.button("💾 Save Data to CSV", width="stretch"):
                    output_mgr = OutputManager()
                    filepath = output_mgr.output_dir / f'generated_data_{output_mgr.timestamp}.csv'
                    data.to_csv(filepath, index=False)
                    st.success(f"✅ Data saved to {filepath.name}")
            
            col_next1, col_next2, col_next3 = st.columns([1, 1, 1])
            with col_next2:
                if st.button("Next: Forecasting →", width="stretch", type="primary", key="demo_next"):
                    navigate_to('forecast')
        else:
            st.info("👆 Click 'Generate Demo Data' to start")
    
    # ========================================================================
    # TAB 3: CONFIGURATION
    # ========================================================================
    with tab3:
        st.markdown("### System Configuration")
        st.markdown("Adjust parameters for all modules.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Data Generation")
            st.session_state.n_stores = st.slider("Number of Stores", 1, 10, st.session_state.n_stores)
            st.session_state.n_products = st.slider("Number of Products", 1, 10, st.session_state.n_products)
            st.session_state.n_weeks = st.slider("Historical Weeks", 20, 208, st.session_state.n_weeks)
            
            st.markdown("#### 📈 Forecasting")
            st.session_state.forecast_horizon = st.slider("Forecast Horizon (weeks)", 1, 52, st.session_state.forecast_horizon)
        
        with col2:
            st.markdown("#### 📦 Inventory")
            st.session_state.planning_horizon = st.slider("Planning Horizon (weeks)", 1, 52, st.session_state.planning_horizon)
            st.session_state.service_level = st.slider("Service Level", 0.80, 0.99, st.session_state.service_level, 0.01)
            st.session_state.lead_time = st.slider("Lead Time (weeks)", 1, 8, st.session_state.lead_time)
            st.session_state.review_period = st.slider("Review Period (weeks)", 1, 4, st.session_state.get('review_period', 1))
            
            st.markdown("#### 🚚 Routing")
            st.session_state.max_payload = st.slider("Max Payload (units)", 50, 500, st.session_state.max_payload, 10)
            st.session_state.n_customers = st.slider("Number of Customers", 10, 100, st.session_state.n_customers)
        
        st.success("✅ Configuration updated")
    
    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================================
# FORECASTING PAGE
# ============================================================================
def show_forecast_page():
    show_navigation_bar('forecast')
    
    st.markdown('<div class="content-container">', unsafe_allow_html=True)
    st.markdown("## 📈 Demand Forecasting")
    st.markdown("---")
    
    if not st.session_state.data_generated:
        st.warning("⚠️ Please generate data first")
        if st.button("← Go to Data Generation", width="stretch"):
            navigate_to('data')
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.metric("Forecast Horizon", f"{st.session_state.forecast_horizon} weeks")
    with col_info2:
        st.metric("Model", "LightGBM")
    with col_info3:
        st.metric("Time Series", st.session_state.n_stores * st.session_state.n_products)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🤖 Train & Forecast", key="forecast", width="stretch", type="primary"):
        with st.spinner("Training forecasting model... This may take a minute..."):
            data = st.session_state.data
            
            split_date = data['date'].max() - pd.Timedelta(weeks=st.session_state.forecast_horizon)
            train_data = data[data['date'] <= split_date]
            
            forecaster = Forecaster(
                primary_keys=['store', 'product'],
                date_col='date',
                target_col='sales',
                frequency='W',
                forecast_horizon=st.session_state.forecast_horizon
            )
            forecaster.fit(train_data)
            
            forecasts = forecaster.predict(data)
            
            st.session_state.forecasts = forecasts
            st.session_state.forecasts_generated = True
            st.success("✅ Forecasts generated successfully!")
    
    if st.session_state.forecasts_generated:
        forecasts = st.session_state.forecasts
        
        st.markdown("#### Forecasts vs Actuals")
        
        stores = sorted(forecasts['store'].unique())
        products = sorted(forecasts['product'].unique())
        
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            selected_store = st.selectbox("Select Store", stores)
        with col_s2:
            selected_product = st.selectbox("Select Product", products)
        
        subset = forecasts[
            (forecasts['store'] == selected_store) & 
            (forecasts['product'] == selected_product)
        ]
        
        fig = create_forecast_chart(subset)
        fig.update_layout(title=f'Store {selected_store} - Product {selected_product}')
        st.plotly_chart(fig, width="stretch")
        
        with st.expander("📋 View Forecast Data"):
            st.dataframe(subset, width="stretch")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Save button
        col_save1, col_save2, col_save3 = st.columns([1, 1, 1])
        with col_save2:
            if st.button("💾 Save Forecasts to CSV", width="stretch"):
                output_mgr = OutputManager()
                filepath = output_mgr.save_forecasts(forecasts)
                st.success(f"✅ Forecasts saved to {filepath.name}")
        
        col_next1, col_next2, col_next3 = st.columns([1, 1, 1])
        with col_next2:
            if st.button("Next: Inventory Optimization →", width="stretch", type="primary"):
                navigate_to('inventory')
    else:
        st.info("👆 Click 'Train & Forecast' to generate predictions")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# INVENTORY PAGE
# ============================================================================
def show_inventory_page():
    show_navigation_bar('inventory')
    
    st.markdown('<div class="content-container">', unsafe_allow_html=True)
    st.markdown("## 📦 Inventory Optimization")
    st.markdown("---")
    
    if not st.session_state.forecasts_generated:
        st.warning("⚠️ Please generate forecasts first")
        if st.button("← Go to Forecasting", width="stretch"):
            navigate_to('forecast')
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.metric("Service Level", f"{st.session_state.service_level*100:.0f}%")
    with col_info2:
        st.metric("Lead Time", f"{st.session_state.lead_time} weeks")
    with col_info3:
        st.metric("Planning Horizon", f"{st.session_state.planning_horizon} weeks")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("⚙️ Optimize Inventory", key="optimize", width="stretch", type="primary"):
        with st.spinner("Running inventory simulation..."):
            forecasts = st.session_state.forecasts
            
            optimizer = Optimizer(
                primary_keys=['store', 'product'],
                date_col='date',
                target_col='sales',
                inv_col='inventory',
                planning_horizon=st.session_state.planning_horizon,
                service_level=st.session_state.service_level,
                review_period=st.session_state.review_period,
                lead_time=st.session_state.lead_time
            )
            
            inventory_plan = optimizer.simulate(forecasts)
            
            st.session_state.inventory_plan = inventory_plan
            st.session_state.inventory_optimized = True
            st.success("✅ Inventory optimized successfully!")
    
    if st.session_state.inventory_optimized:
        inventory_plan = st.session_state.inventory_plan
        test_inventory = inventory_plan[inventory_plan['sample'] == 'test']
        
        avg_inventory, total_orders, stockouts, fill_rate = calculate_inventory_metrics(test_inventory)
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Avg Inventory", f"{avg_inventory:.1f}")
        with col_m2:
            st.metric("Total Orders", f"{total_orders:.0f}")
        with col_m3:
            st.metric("Stockouts", stockouts)
        with col_m4:
            st.metric("Fill Rate", f"{fill_rate:.1f}%")
        
        st.markdown("#### Inventory Levels Over Time")
        
        stores = sorted(inventory_plan['store'].unique())
        products = sorted(inventory_plan['product'].unique())
        
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            selected_store = st.selectbox("Select Store", stores, key="inv_store")
        with col_s2:
            selected_product = st.selectbox("Select Product", products, key="inv_product")
        
        subset = inventory_plan[
            (inventory_plan['store'] == selected_store) & 
            (inventory_plan['product'] == selected_product)
        ]
        
        # Filter to show only test (future) data
        subset_test = subset[subset['sample'] == 'test']
        
        fig = create_inventory_chart(subset_test)
        fig.update_layout(title=f'Store {selected_store} - Product {selected_product}')
        st.plotly_chart(fig, width="stretch")
        
        col_o1, col_o2 = st.columns(2)
        
        with col_o1:
            fig = create_orders_chart(subset_test)
            st.plotly_chart(fig, width="stretch")
        
        with col_o2:
            fig = create_shipments_chart(subset_test)
            st.plotly_chart(fig, width="stretch")
        
        with st.expander("📋 View Inventory Data (Future)"):
            st.dataframe(subset_test, width="stretch")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Save button
        col_save1, col_save2, col_save3 = st.columns([1, 1, 1])
        with col_save2:
            if st.button("💾 Save Inventory Plan to CSV", width="stretch"):
                output_mgr = OutputManager()
                filepath = output_mgr.save_inventory(inventory_plan)
                st.success(f"✅ Inventory plan saved to {filepath.name}")
        
        col_next1, col_next2, col_next3 = st.columns([1, 1, 1])
        with col_next2:
            if st.button("Next: Delivery Routing →", width="stretch", type="primary"):
                navigate_to('routing')
    else:
        st.info("👆 Click 'Optimize Inventory' to run simulation")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# ROUTING PAGE
# ============================================================================
def show_routing_page():
    show_navigation_bar('routing')
    
    st.markdown('<div class="content-container">', unsafe_allow_html=True)
    st.markdown("## 🚚 Delivery Routing")
    st.markdown("---")
    
    if not st.session_state.forecasts_generated:
        st.warning("⚠️ Please generate forecasts first")
        if st.button("← Go to Forecasting", width="stretch"):
            navigate_to('forecast')
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.metric("Max Payload", f"{st.session_state.max_payload} units")
    with col_info2:
        st.metric("Customers", st.session_state.n_customers)
    with col_info3:
        st.metric("Origin", "08020")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🗺️ Optimize Routes", key="route", width="stretch", type="primary"):
        with st.spinner("Optimizing delivery routes..."):
            forecasts = st.session_state.forecasts
            
            router = Router(
                primary_keys=['store', 'product'],
                date_col='date',
                target_col='sales',
                max_payload=st.session_state.max_payload,
                origin='08020'
            )
            
            router.generate_customers(n_customers=st.session_state.n_customers)
            deliveries = router.distribute_demand(forecasts)
            
            if len(deliveries) > 0:
                deliveries = router.assign_trucks(deliveries)
                deliveries, routes = router.optimize_routes(deliveries)
                
                st.session_state.deliveries = deliveries
                st.session_state.routes = routes
                st.session_state.routes_optimized = True
                st.success("✅ Routes optimized successfully!")
            else:
                st.error("No deliveries to route. Try adjusting parameters.")
    
    if st.session_state.routes_optimized:
        deliveries = st.session_state.deliveries
        routes = st.session_state.routes
        
        n_trucks, total_distance, avg_utilization, total_deliveries = calculate_routing_metrics(routes, deliveries, st.session_state.max_payload)
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Total Trucks", n_trucks)
        with col_m2:
            st.metric("Total Distance", f"{total_distance:.0f}")
        with col_m3:
            st.metric("Avg Utilization", f"{avg_utilization:.1f}%")
        with col_m4:
            st.metric("Total Deliveries", total_deliveries)
        
        st.markdown("#### Truck Utilization")
        fig = create_truck_utilization_chart(deliveries, st.session_state.max_payload)
        st.plotly_chart(fig, width="stretch", height=500)
        
        st.markdown("#### Route Analysis")
        col_r1, col_r2 = st.columns(2)
        
        with col_r1:
            fig = create_route_distance_chart(routes)
            st.plotly_chart(fig, width="stretch", height=450)
        
        with col_r2:
            fig = create_distance_vs_stops_chart(routes, deliveries)
            st.plotly_chart(fig, width="stretch", height=450)
        
        with st.expander("📋 View Delivery Data"):
            st.dataframe(deliveries, width="stretch")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Save all button
        col_save1, col_save2, col_save3 = st.columns([1, 1, 1])
        with col_save2:
            if st.button("💾 Save All Results to CSV", width="stretch", type="primary"):
                output_mgr = OutputManager()
                saved_files = output_mgr.save_all(
                    st.session_state.forecasts,
                    st.session_state.inventory_plan,
                    deliveries,
                    routes
                )
                st.success(f"✅ All results saved to output/ directory:")
                for output_type, filepath in saved_files.items():
                    st.write(f"   - {filepath.name}")
        
        st.markdown("<br>", unsafe_allow_html=True)
    else:
        st.info("👆 Click 'Optimize Routes' to plan deliveries")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# MAIN APP ROUTER
# ============================================================================
if st.session_state.current_page == 'landing':
    show_landing_page()
elif st.session_state.current_page == 'data':
    show_data_page()
elif st.session_state.current_page == 'forecast':
    show_forecast_page()
elif st.session_state.current_page == 'inventory':
    show_inventory_page()
elif st.session_state.current_page == 'routing':
    show_routing_page()
else:
    show_landing_page()

