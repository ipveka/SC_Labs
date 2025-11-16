"""
Data & Settings page module
"""
import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime

from auxiliar.auxiliar import generate_data
from utils.output_manager import OutputManager
from utils.data_validator import validate_data_quality, get_recommended_forecast_horizon, format_validation_report
from app_utils import create_sales_trend_chart, create_sales_histogram, create_sales_boxplot
from components import page_header, section_divider


def show_data_page(show_navigation_bar, navigate_to):
    """Display data & settings page with tabs"""
    show_navigation_bar('data')
    
    st.markdown('<div class="content-container">', unsafe_allow_html=True)
    page_header("⚙️", "Data & Settings", "Configure your data source and system parameters")
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload Data", "🎲 Generate Demo Data", "📂 Load Previous Results", "⚙️ Configuration"])
    
    # TAB 1: UPLOAD DATA
    with tab1:
        _show_upload_tab(navigate_to)
    
    # TAB 2: GENERATE DEMO DATA
    with tab2:
        _show_demo_data_tab(navigate_to)
    
    # TAB 3: LOAD PREVIOUS RESULTS
    with tab3:
        _show_load_results_tab()
    
    # TAB 4: CONFIGURATION
    with tab4:
        _show_configuration_tab()
    
    st.markdown('</div>', unsafe_allow_html=True)


def _show_upload_tab(navigate_to):
    """Upload data tab"""
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
    sample_path = Path(__file__).parent.parent.parent / 'data' / 'sample_sales_data.csv'
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
            data = pd.read_csv(uploaded_file)
            
            # Validate required columns
            required_cols = ['store', 'product', 'date', 'sales']
            missing_cols = [col for col in required_cols if col not in data.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                st.info("Please ensure your CSV has: store, product, date, sales")
            else:
                data['date'] = pd.to_datetime(data['date'])
                
                if not pd.api.types.is_numeric_dtype(data['sales']):
                    st.error("❌ 'sales' column must contain numeric values")
                else:
                    st.session_state.data = data
                    st.session_state.data_generated = True
                    st.success("✅ Data uploaded successfully!")
                    
                    # Run data quality validation
                    validation_results = validate_data_quality(data)
                    
                    if validation_results['warnings'] or validation_results['recommendations']:
                        with st.expander("⚠️ Data Quality Assessment", expanded=True):
                            st.markdown(format_validation_report(validation_results))
                            recommended_horizon = get_recommended_forecast_horizon(data)
                            st.info(f"💡 Recommended forecast horizon: {recommended_horizon} periods based on your data")
                    
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


def _show_demo_data_tab(navigate_to):
    """Generate demo data tab"""
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


def _show_load_results_tab():
    """Load previous results tab"""
    st.markdown("### Load Previous Results")
    st.markdown("Resume work by loading previously saved forecasts, inventory plans, or routing results.")
    
    output_dir = Path(__file__).parent.parent.parent / 'output'
    
    if not output_dir.exists() or not any(output_dir.glob('*.csv')):
        st.info("📁 No previous results found in the output directory.")
        st.markdown("Run the pipeline and save results to see them here.")
    else:
        files = list(output_dir.glob('*.csv'))
        files = [f for f in files if f.name != '.gitkeep']
        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        if not files:
            st.info("📁 No previous results found.")
        else:
            st.markdown("#### Available Result Sets")
            
            # Group files by timestamp pattern
            file_groups = {}
            for file in files:
                parts = file.stem.split('_')
                if len(parts) >= 3:
                    timestamp = '_'.join(parts[-2:])
                    file_type = '_'.join(parts[:-2])
                    
                    if timestamp not in file_groups:
                        file_groups[timestamp] = {
                            'files': [],
                            'mtime': file.stat().st_mtime
                        }
                    file_groups[timestamp]['files'].append((file_type, file))
            
            sorted_groups = sorted(file_groups.items(), key=lambda x: x[1]['mtime'], reverse=True)
            
            for timestamp, group_info in sorted_groups[:10]:
                files_in_group = group_info['files']
                mtime = datetime.fromtimestamp(group_info['mtime'])
                
                with st.expander(f"📊 Results from {mtime.strftime('%Y-%m-%d %H:%M:%S')}", expanded=False):
                    st.markdown(f"**Files in this set:** {len(files_in_group)}")
                    
                    cols = st.columns(len(files_in_group))
                    
                    for idx, (file_type, file_path) in enumerate(files_in_group):
                        with cols[idx]:
                            if 'forecast' in file_type:
                                st.markdown("📈 **Forecasts**")
                            elif 'inventory' in file_type:
                                st.markdown("📦 **Inventory**")
                            elif 'deliveries' in file_type:
                                st.markdown("🚚 **Deliveries**")
                            elif 'routes' in file_type:
                                st.markdown("🗺️ **Routes**")
                            elif 'generated_data' in file_type:
                                st.markdown("📊 **Data**")
                            else:
                                st.markdown(f"📄 **{file_type}**")
                    
                    col_load1, col_load2, col_load3 = st.columns([1, 1, 1])
                    
                    with col_load2:
                        if st.button("📂 Load This Result Set", key=f"load_{timestamp}", width="stretch"):
                            try:
                                for file_type, file_path in files_in_group:
                                    df = pd.read_csv(file_path)
                                    
                                    if 'forecast' in file_type:
                                        if 'date' in df.columns:
                                            df['date'] = pd.to_datetime(df['date'])
                                        st.session_state.forecasts = df
                                        st.session_state.forecasts_generated = True
                                    
                                    elif 'inventory' in file_type:
                                        if 'date' in df.columns:
                                            df['date'] = pd.to_datetime(df['date'])
                                        st.session_state.inventory_plan = df
                                        st.session_state.inventory_optimized = True
                                    
                                    elif 'deliveries' in file_type:
                                        if 'date' in df.columns:
                                            df['date'] = pd.to_datetime(df['date'])
                                        st.session_state.deliveries = df
                                    
                                    elif 'routes' in file_type:
                                        st.session_state.routes = df
                                        st.session_state.routes_optimized = True
                                    
                                    elif 'generated_data' in file_type or 'data' in file_type:
                                        if 'date' in df.columns:
                                            df['date'] = pd.to_datetime(df['date'])
                                        st.session_state.data = df
                                        st.session_state.data_generated = True
                                
                                st.success("✅ Results loaded successfully!")
                                st.info("Navigate to the relevant module to view the loaded results.")
                                
                            except Exception as e:
                                st.error(f"❌ Error loading files: {str(e)}")
            
            st.markdown("---")
            st.markdown("**💡 Tip:** Results are automatically saved with timestamps when you use the 'Save' buttons in each module.")


def _show_configuration_tab():
    """Configuration tab"""
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
        
        # Advanced forecasting settings
        with st.expander("🔧 Advanced Forecasting Settings"):
            st.markdown("**LightGBM Model Parameters**")
            st.info("These settings control the machine learning model. Default values work well for most cases.")
            
            if 'lgb_num_boost_round' not in st.session_state:
                st.session_state.lgb_num_boost_round = 200
            if 'lgb_learning_rate' not in st.session_state:
                st.session_state.lgb_learning_rate = 0.05
            if 'lgb_num_leaves' not in st.session_state:
                st.session_state.lgb_num_leaves = 31
            
            st.session_state.lgb_num_boost_round = st.slider(
                "Training Iterations", 50, 500, st.session_state.lgb_num_boost_round, 50,
                help="More iterations = better fit but slower training"
            )
            st.session_state.lgb_learning_rate = st.slider(
                "Learning Rate", 0.01, 0.20, st.session_state.lgb_learning_rate, 0.01,
                help="Lower = more conservative learning, higher = faster but less stable"
            )
            st.session_state.lgb_num_leaves = st.slider(
                "Tree Complexity", 15, 63, st.session_state.lgb_num_leaves, 2,
                help="Higher = more complex model (risk of overfitting)"
            )
    
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
