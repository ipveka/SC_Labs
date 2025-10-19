"""
SC Labs - Supply Chain Optimization Dashboard
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
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.2rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_generated' not in st.session_state:
    st.session_state.data_generated = False
if 'forecasts_generated' not in st.session_state:
    st.session_state.forecasts_generated = False
if 'inventory_optimized' not in st.session_state:
    st.session_state.inventory_optimized = False
if 'routes_optimized' not in st.session_state:
    st.session_state.routes_optimized = False

# Header
st.markdown('<div class="main-header">📦 SC Labs - Supply Chain Optimization</div>', unsafe_allow_html=True)
st.markdown("### Interactive demonstration of demand forecasting, inventory optimization, and delivery routing")

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/300x100/1f77b4/ffffff?text=SC+Labs", width="stretch")
    st.markdown("## 🎛️ Configuration")
    
    st.markdown("### Data Generation")
    n_stores = st.slider("Number of Stores", 1, 5, 3)
    n_products = st.slider("Number of Products", 1, 5, 2)
    n_weeks = st.slider("Historical Weeks", 20, 104, 52)
    
    st.markdown("### Forecasting")
    forecast_horizon = st.slider("Forecast Horizon (weeks)", 2, 12, 4)
    
    st.markdown("### Inventory")
    service_level = st.slider("Service Level", 0.80, 0.99, 0.95, 0.01)
    lead_time = st.slider("Lead Time (weeks)", 1, 4, 2)
    
    st.markdown("### Routing")
    max_payload = st.slider("Max Payload (units)", 50, 200, 100, 10)
    n_customers = st.slider("Number of Customers", 10, 50, 30)
    
    st.markdown("---")
    if st.button("🔄 Reset All", width="stretch"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Generation", "📈 Forecasting", "📦 Inventory", "🚚 Routing"])

# ============================================================================
# TAB 1: DATA GENERATION
# ============================================================================
with tab1:
    st.markdown('<div class="section-header">📊 Synthetic Data Generation</div>', unsafe_allow_html=True)
    st.markdown("Generate realistic supply chain data with trend, seasonality, and noise.")
    
    if st.button("🎲 Generate Data", key="gen_data", width="stretch", type="primary"):
        with st.spinner("Generating synthetic data..."):
            st.session_state.data = generate_data(
                n_stores=n_stores,
                n_products=n_products,
                n_weeks=n_weeks,
                start_date='2024-01-01',
                seed=42
            )
            st.session_state.data_generated = True
            st.success("✅ Data generated successfully!")
    
    if st.session_state.data_generated:
        data = st.session_state.data
        
        # Metrics
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Total Records", f"{len(data):,}")
        with col_m2:
            st.metric("Avg Sales", f"{data['sales'].mean():.1f}")
        with col_m3:
            st.metric("Date Range", f"{n_weeks} weeks")
        with col_m4:
            st.metric("Time Series", n_stores * n_products)
        
        # Sales over time plot
        st.markdown("#### Sales Over Time")
        fig = create_sales_trend_chart(data)
        st.plotly_chart(fig, width="stretch")
        
        # Distribution
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            fig = create_sales_histogram(data)
            st.plotly_chart(fig, width="stretch")
        
        with col_d2:
            fig = create_sales_boxplot(data)
            st.plotly_chart(fig, width="stretch")
        
        # Data preview
        with st.expander("📋 View Data Sample"):
            st.dataframe(data.head(20), width="stretch")
    else:
        st.info("👆 Click 'Generate Data' to start")

# ============================================================================
# TAB 2: FORECASTING
# ============================================================================
with tab2:
    st.markdown('<div class="section-header">📈 Demand Forecasting</div>', unsafe_allow_html=True)
    st.markdown("Train a neural network model to predict future demand using GluonTS.")
    
    if not st.session_state.data_generated:
        st.warning("⚠️ Please generate data first in the 'Data Generation' tab")
    else:
        if st.button("🤖 Train & Forecast", key="forecast", width="stretch", type="primary"):
            with st.spinner("Training forecasting model... This may take a minute..."):
                data = st.session_state.data
                
                # Split data
                split_date = data['date'].max() - pd.Timedelta(weeks=forecast_horizon)
                train_data = data[data['date'] <= split_date]
                
                # Train forecaster
                forecaster = Forecaster(
                    primary_keys=['store', 'product'],
                    date_col='date',
                    target_col='sales',
                    frequency='W',
                    forecast_horizon=forecast_horizon
                )
                forecaster.fit(train_data)
                
                # Generate forecasts
                forecasts = forecaster.predict(data)
                
                st.session_state.forecasts = forecasts
                st.session_state.forecasts_generated = True
                st.success("✅ Forecasts generated successfully!")
        
        if st.session_state.forecasts_generated:
            forecasts = st.session_state.forecasts
            
            # Forecast visualization
            st.markdown("#### Forecasts vs Actuals")
            
            # Select store-product combination
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
            
            # Data preview
            with st.expander("📋 View Forecast Data"):
                st.dataframe(subset, width="stretch")
        else:
            st.info("👆 Click 'Train & Forecast' to generate predictions")

# ============================================================================
# TAB 3: INVENTORY OPTIMIZATION
# ============================================================================
with tab3:
    st.markdown('<div class="section-header">📦 Inventory Optimization</div>', unsafe_allow_html=True)
    st.markdown("Optimize inventory levels using reorder point policy with safety stock.")
    
    if not st.session_state.forecasts_generated:
        st.warning("⚠️ Please generate forecasts first in the 'Forecasting' tab")
    else:
        if st.button("⚙️ Optimize Inventory", key="optimize", width="stretch", type="primary"):
            with st.spinner("Running inventory simulation..."):
                forecasts = st.session_state.forecasts
                
                optimizer = Optimizer(
                    primary_keys=['store', 'product'],
                    date_col='date',
                    target_col='sales',
                    inv_col='inventory',
                    planning_horizon=8,
                    service_level=service_level,
                    review_period=1,
                    lead_time=lead_time
                )
                
                inventory_plan = optimizer.simulate(forecasts)
                
                st.session_state.inventory_plan = inventory_plan
                st.session_state.inventory_optimized = True
                st.success("✅ Inventory optimized successfully!")
        
        if st.session_state.inventory_optimized:
            inventory_plan = st.session_state.inventory_plan
            test_inventory = inventory_plan[inventory_plan['sample'] == 'test']
            
            # Metrics
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
            
            # Inventory visualization
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
            st.plotly_chart(fig, width="stretch")
            
            # Orders and shipments
            col_o1, col_o2 = st.columns(2)
            
            with col_o1:
                fig = create_orders_chart(subset)
                st.plotly_chart(fig, width="stretch")
            
            with col_o2:
                fig = create_shipments_chart(subset)
                st.plotly_chart(fig, width="stretch")
            
            # Data preview
            with st.expander("📋 View Inventory Data"):
                st.dataframe(subset, width="stretch")
        else:
            st.info("👆 Click 'Optimize Inventory' to run simulation")

# ============================================================================
# TAB 4: DELIVERY ROUTING
# ============================================================================
with tab4:
    st.markdown('<div class="section-header">🚚 Delivery Routing</div>', unsafe_allow_html=True)
    st.markdown("Optimize delivery routes with truck assignment and payload constraints.")
    
    if not st.session_state.forecasts_generated:
        st.warning("⚠️ Please generate forecasts first in the 'Forecasting' tab")
    else:
        if st.button("🗺️ Optimize Routes", key="route", width="stretch", type="primary"):
            with st.spinner("Optimizing delivery routes..."):
                forecasts = st.session_state.forecasts
                
                router = Router(
                    primary_keys=['store', 'product'],
                    date_col='date',
                    target_col='sales',
                    max_payload=max_payload,
                    origin='08020'
                )
                
                # Generate customers
                router.generate_customers(n_customers=n_customers)
                
                # Distribute demand
                deliveries = router.distribute_demand(forecasts)
                
                if len(deliveries) > 0:
                    # Assign trucks
                    deliveries = router.assign_trucks(deliveries)
                    
                    # Optimize routes
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
            
            # Metrics
            n_trucks, total_distance, avg_utilization, total_deliveries = calculate_routing_metrics(routes, deliveries, max_payload)
            
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("Total Trucks", n_trucks)
            with col_m2:
                st.metric("Total Distance", f"{total_distance:.0f}")
            with col_m3:
                st.metric("Avg Utilization", f"{avg_utilization:.1f}%")
            with col_m4:
                st.metric("Total Deliveries", total_deliveries)
            
            # Truck utilization
            st.markdown("#### Truck Utilization")
            fig = create_truck_utilization_chart(deliveries, max_payload)
            st.plotly_chart(fig, width="stretch")
            
            # Route distances
            col_r1, col_r2 = st.columns(2)
            
            with col_r1:
                fig = create_route_distance_chart(routes)
                st.plotly_chart(fig, width="stretch")
            
            with col_r2:
                fig = create_distance_vs_stops_chart(routes, deliveries)
                st.plotly_chart(fig, width="stretch")
            
            # Data preview
            with st.expander("📋 View Delivery Data"):
                st.dataframe(deliveries, width="stretch")
        else:
            st.info("👆 Click 'Optimize Routes' to plan deliveries")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>SC Labs - Supply Chain Optimization Dashboard</p>
    <p>Built with Streamlit | Powered by GluonTS, Pandas, and Plotly</p>
</div>
""", unsafe_allow_html=True)
