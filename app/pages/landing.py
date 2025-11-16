"""
Landing page module
"""
import streamlit as st


def show_footer():
    """Display footer on all pages"""
    st.markdown("""
    <div style='text-align: center; background: white; color: #718096; padding: 2.5rem 2rem; margin-top: 3rem; border-top: 2px solid #e2e8f0; border-radius: 0 0 20px 20px;'>
        <p style='font-size: 0.95rem; font-weight: 600; margin: 0; letter-spacing: 0.05em;'>SC LABS • BARCELONA • 2025</p>
        <p style='font-size: 0.85rem; margin-top: 0.5rem; color: #a0aec0;'>Supply Chain Optimization Platform</p>
    </div>
    """, unsafe_allow_html=True)


def show_landing_page(navigate_to):
    """Display landing page with module cards"""
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
