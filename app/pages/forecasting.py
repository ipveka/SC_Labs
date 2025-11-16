"""
Forecasting page module
"""
import streamlit as st
import pandas as pd

from forecaster.forecaster import Forecaster
from utils.output_manager import OutputManager
from app_utils import create_forecast_chart
from components import page_header, section_divider, progress_steps


def show_forecast_page(show_navigation_bar, navigate_to):
    """Display forecasting page"""
    show_navigation_bar('forecast')
    
    st.markdown('<div class="content-container">', unsafe_allow_html=True)
    page_header("📈", "Demand Forecasting", "Predict future demand using LightGBM with automated feature engineering")
    
    # Show progress
    progress_steps(["Data", "Forecast", "Inventory", "Routing"], 1)
    
    if not st.session_state.data_generated:
        st.warning("⚠️ Please generate or upload data first")
        if st.button("← Go to Data & Settings", width="stretch"):
            navigate_to('data')
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    section_divider("Configuration")
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
        
        section_divider("Results")
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
