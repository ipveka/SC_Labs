"""
Navigation components for SC Labs dashboard
"""
import streamlit as st


def navigate_to(page):
    """Navigate to a specific page"""
    st.session_state.current_page = page
    st.rerun()


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
