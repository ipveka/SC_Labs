"""
Routing optimization page module
"""
import streamlit as st

from router.router import Router
from utils.output_manager import OutputManager
from app_utils import calculate_routing_metrics, create_truck_utilization_chart, create_route_distance_chart, create_distance_vs_stops_chart
from components import page_header, section_divider, progress_steps


def show_routing_page(show_navigation_bar, navigate_to):
    """Display routing optimization page"""
    show_navigation_bar('routing')
    
    st.markdown('<div class="content-container">', unsafe_allow_html=True)
    page_header("🚚", "Delivery Routing", "Optimize delivery routes with intelligent truck assignment and payload management")
    
    # Show progress
    progress_steps(["Data", "Forecast", "Inventory", "Routing"], 3)
    
    if not st.session_state.forecasts_generated:
        st.warning("⚠️ Please generate forecasts first")
        if st.button("← Go to Forecasting", width="stretch"):
            navigate_to('forecast')
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    section_divider("Configuration")
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
        
        section_divider("Performance Metrics")
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
