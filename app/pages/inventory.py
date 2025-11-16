"""
Inventory optimization page module
"""
import streamlit as st

from optimizer.optimizer import Optimizer
from utils.output_manager import OutputManager
from app_utils import calculate_inventory_metrics, create_inventory_chart, create_orders_chart, create_shipments_chart
from components import page_header, section_divider, progress_steps


def show_inventory_page(show_navigation_bar, navigate_to):
    """Display inventory optimization page"""
    show_navigation_bar('inventory')
    
    st.markdown('<div class="content-container">', unsafe_allow_html=True)
    page_header("📦", "Inventory Optimization", "Optimize stock levels using reorder point policies with safety stock")
    
    # Show progress
    progress_steps(["Data", "Forecast", "Inventory", "Routing"], 2)
    
    if not st.session_state.forecasts_generated:
        st.warning("⚠️ Please generate forecasts first")
        if st.button("← Go to Forecasting", width="stretch"):
            navigate_to('forecast')
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    section_divider("Configuration")
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
        
        section_divider("Performance Metrics")
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
