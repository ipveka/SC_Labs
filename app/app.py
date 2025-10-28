"""
SC Labs - Supply Chain Optimization Dashboard
Beautiful landing page with module navigation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np

from auxiliar.auxiliar import generate_data
from forecaster.forecaster import Forecaster
from optimizer.optimizer import Optimizer
from router.router import Router
from app_utils import *

# Page configuration
st.set_page_config(
    page_title="Supply Chain Labs",
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

# Configuration defaults
if 'n_stores' not in st.session_state:
    st.session_state.n_stores = 3
if 'n_products' not in st.session_state:
    st.session_state.n_products = 2
if 'n_weeks' not in st.session_state:
    st.session_state.n_weeks = 52
if 'forecast_horizon' not in st.session_state:
    st.session_state.forecast_horizon = 4
if 'model_type' not in st.session_state:
    st.session_state.model_type = 'simple_feedforward'
if 'service_level' not in st.session_state:
    st.session_state.service_level = 0.95
if 'lead_time' not in st.session_state:
    st.session_state.lead_time = 2
if 'max_payload' not in st.session_state:
    st.session_state.max_payload = 100
if 'n_customers' not in st.session_state:
    st.session_state.n_customers = 30

# Navigation functions
def navigate_to(page):
    st.session_state.current_page = page
    st.rerun()

def show_navigation_bar(current_module=None):
    """Display navigation bar with module links"""
    col1, col2, col3, col4, col5, col6 = st.columns([2, 1, 1, 1, 1, 1])
    
    with col1:
        if st.button("🏠 Home", use_container_width=True):
            navigate_to('landing')
    
    with col2:
        if st.button("📊 Data", use_container_width=True, disabled=(current_module == 'data')):
            navigate_to('data')
    
    with col3:
        if st.button("📈 Forecast", use_container_width=True, disabled=(current_module == 'forecast')):
            navigate_to('forecast')
    
    with col4:
        if st.button("📦 Inventory", use_container_width=True, disabled=(current_module == 'inventory')):
            navigate_to('inventory')
    
    with col5:
        if st.button("🚚 Routing", use_container_width=True, disabled=(current_module == 'routing')):
            navigate_to('routing')
    
    with col6:
        if st.button("⚙️ Settings", use_container_width=True):
            navigate_to('settings')
    
    st.markdown("---")


# ============================================================================
# LANDING PAGE
# ============================================================================
def show_landing_page():
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">📦 SC Labs</div>
        <div class="hero-subtitle">Supply Chain Optimization Platform</div>
        <div class="hero-description">
            A comprehensive end-to-end solution for modern supply chain management.
            Leverage machine learning for demand forecasting, optimize inventory levels,
            and streamline delivery operations—all in one powerful platform.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<h2 style="text-align: center; color: white; margin: 3rem 0 2rem;">Explore Our Modules</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("""
        <div class="module-card">
            <span class="module-icon">📊</span>
            <div class="module-title">Data Generation</div>
            <div class="module-description">
                Create realistic synthetic supply chain data with customizable parameters.
            </div>
            <ul class="module-features">
                <li>✓ Configurable stores and products</li>
                <li>✓ Realistic demand patterns</li>
                <li>✓ Historical data simulation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Launch Data Generation →", key="btn_data", use_container_width=True):
            navigate_to('data')
    
    with col2:
        st.markdown("""
        <div class="module-card">
            <span class="module-icon">📈</span>
            <div class="module-title">Demand Forecasting</div>
            <div class="module-description">
                Predict future demand using state-of-the-art machine learning models.
            </div>
            <ul class="module-features">
                <li>✓ Multiple ML models</li>
                <li>✓ Time series analysis</li>
                <li>✓ Accuracy metrics</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Launch Forecasting →", key="btn_forecast", use_container_width=True):
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
        if st.button("Launch Inventory →", key="btn_inventory", use_container_width=True):
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
        if st.button("Launch Routing →", key="btn_routing", use_container_width=True):
            navigate_to('routing')
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; color: white; padding: 2rem;'>
        <p style='font-size: 1.1rem;'>Built with ❤️ using Streamlit, GluonTS, and PyTorch</p>
        <p style='font-size: 0.9rem; opacity: 0.8;'>© 2024 SC Labs</p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# SETTINGS PAGE
# ============================================================================
def show_settings_page():
    show_navigation_bar()
    
    st.markdown('<div class="content-container">', unsafe_allow_html=True)
    st.markdown("## ⚙️ Configuration Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Data Generation")
        st.session_state.n_stores = st.slider("Number of Stores", 1, 5, st.session_state.n_stores)
        st.session_state.n_products = st.slider("Number of Products", 1, 5, st.session_state.n_products)
        st.session_state.n_weeks = st.slider("Historical Weeks", 20, 104, st.session_state.n_weeks)
        
        st.markdown("### 📈 Forecasting")
        st.session_state.forecast_horizon = st.slider("Forecast Horizon (weeks)", 2, 12, st.session_state.forecast_horizon)
        st.session_state.model_type = st.selectbox(
            "Forecasting Model",
            options=['simple_feedforward', 'deepar', 'transformer'],
            format_func=lambda x: {
                'simple_feedforward': 'Simple Feed Forward (Fast)',
                'deepar': 'DeepAR (Probabilistic)',
                'transformer': 'Temporal Fusion Transformer (Advanced)'
            }[x],
            index=['simple_feedforward', 'deepar', 'transformer'].index(st.session_state.model_type)
        )
    
    with col2:
        st.markdown("### 📦 Inventory")
        st.session_state.service_level = st.slider("Service Level", 0.80, 0.99, st.session_state.service_level, 0.01)
        st.session_state.lead_time = st.slider("Lead Time (weeks)", 1, 4, st.session_state.lead_time)
        
        st.markdown("### 🚚 Routing")
        st.session_state.max_payload = st.slider("Max Payload (units)", 50, 200, st.session_state.max_payload, 10)
        st.session_state.n_customers = st.slider("Number of Customers", 10, 50, st.session_state.n_customers)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# DATA GENERATION PAGE
# ============================================================================
def show_data_page():
    show_navigation_bar('data')
    
    st.markdown('<div class="content-container">', unsafe_allow_html=True)
    st.markdown("## 📊 Synthetic Data Generation")
    st.markdown("---")
    
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.metric("Stores", st.session_state.n_stores)
    with col_info2:
        st.metric("Products", st.session_state.n_products)
    with col_info3:
        st.metric("Weeks", st.session_state.n_weeks)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🎲 Generate Data", key="gen_data", use_container_width=True, type="primary"):
        with st.spinner("Generating synthetic data..."):
            st.session_state.data = generate_data(
                n_stores=st.session_state.n_stores,
                n_products=st.session_state.n_products,
                n_weeks=st.session_state.n_weeks,
                start_date='2024-01-01',
                seed=42
            )
            st.session_state.data_generated = True
            st.success("✅ Data generated successfully!")
    
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
        st.plotly_chart(fig, use_container_width=True)
        
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            fig = create_sales_histogram(data)
            st.plotly_chart(fig, use_container_width=True)
        
        with col_d2:
            fig = create_sales_boxplot(data)
            st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("📋 View Data Sample"):
            st.dataframe(data.head(20), use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        col_next1, col_next2, col_next3 = st.columns([1, 1, 1])
        with col_next2:
            if st.button("Next: Forecasting →", use_container_width=True, type="primary"):
                navigate_to('forecast')
    else:
        st.info("👆 Click 'Generate Data' to start")
    
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
        if st.button("← Go to Data Generation", use_container_width=True):
            navigate_to('data')
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.metric("Forecast Horizon", f"{st.session_state.forecast_horizon} weeks")
    with col_info2:
        model_names = {
            'simple_feedforward': 'SimpleFeedForward',
            'deepar': 'DeepAR',
            'transformer': 'Transformer'
        }
        st.metric("Model", model_names[st.session_state.model_type])
    with col_info3:
        st.metric("Time Series", st.session_state.n_stores * st.session_state.n_products)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🤖 Train & Forecast", key="forecast", use_container_width=True, type="primary"):
        with st.spinner("Training forecasting model... This may take a minute..."):
            data = st.session_state.data
            
            split_date = data['date'].max() - pd.Timedelta(weeks=st.session_state.forecast_horizon)
            train_data = data[data['date'] <= split_date]
            
            forecaster = Forecaster(
                primary_keys=['store', 'product'],
                date_col='date',
                target_col='sales',
                frequency='W',
                forecast_horizon=st.session_state.forecast_horizon,
                model_type=st.session_state.model_type
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
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("📋 View Forecast Data"):
            st.dataframe(subset, use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        col_next1, col_next2, col_next3 = st.columns([1, 1, 1])
        with col_next2:
            if st.button("Next: Inventory Optimization →", use_container_width=True, type="primary"):
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
        if st.button("← Go to Forecasting", use_container_width=True):
            navigate_to('forecast')
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.metric("Service Level", f"{st.session_state.service_level*100:.0f}%")
    with col_info2:
        st.metric("Lead Time", f"{st.session_state.lead_time} weeks")
    with col_info3:
        st.metric("Planning Horizon", "8 weeks")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("⚙️ Optimize Inventory", key="optimize", use_container_width=True, type="primary"):
        with st.spinner("Running inventory simulation..."):
            forecasts = st.session_state.forecasts
            
            optimizer = Optimizer(
                primary_keys=['store', 'product'],
                date_col='date',
                target_col='sales',
                inv_col='inventory',
                planning_horizon=8,
                service_level=st.session_state.service_level,
                review_period=1,
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
        
        fig = create_inventory_chart(subset)
        fig.update_layout(title=f'Store {selected_store} - Product {selected_product}')
        st.plotly_chart(fig, use_container_width=True)
        
        col_o1, col_o2 = st.columns(2)
        
        with col_o1:
            fig = create_orders_chart(subset)
            st.plotly_chart(fig, use_container_width=True)
        
        with col_o2:
            fig = create_shipments_chart(subset)
            st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("📋 View Inventory Data"):
            st.dataframe(subset, use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        col_next1, col_next2, col_next3 = st.columns([1, 1, 1])
        with col_next2:
            if st.button("Next: Delivery Routing →", use_container_width=True, type="primary"):
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
        if st.button("← Go to Forecasting", use_container_width=True):
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
    
    if st.button("🗺️ Optimize Routes", key="route", use_container_width=True, type="primary"):
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
        st.plotly_chart(fig, use_container_width=True)
        
        col_r1, col_r2 = st.columns(2)
        
        with col_r1:
            fig = create_route_distance_chart(routes)
            st.plotly_chart(fig, use_container_width=True)
        
        with col_r2:
            fig = create_distance_vs_stops_chart(routes, deliveries)
            st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("📋 View Delivery Data"):
            st.dataframe(deliveries, use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.success("🎉 All modules completed! You can navigate between modules using the buttons above.")
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
elif st.session_state.current_page == 'settings':
    show_settings_page()
else:
    show_landing_page()
