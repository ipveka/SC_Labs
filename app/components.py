"""
Shared UI components for SC Labs dashboard
"""
import streamlit as st


def page_header(icon, title, description):
    """Display consistent page header across all modules"""
    st.markdown(f"""
    <div style='margin-bottom: 2rem;'>
        <div style='display: flex; align-items: center; gap: 1rem; margin-bottom: 0.5rem;'>
            <span style='font-size: 2.5rem;'>{icon}</span>
            <h1 style='margin: 0; font-size: 2.25rem; font-weight: 700; color: #2d3748; letter-spacing: -0.02em;'>{title}</h1>
        </div>
        <p style='font-size: 1.05rem; color: #4a5568; margin: 0; padding-left: 4rem;'>{description}</p>
    </div>
    <hr style='border: none; border-top: 2px solid #e2e8f0; margin: 1.5rem 0 2rem;'>
    """, unsafe_allow_html=True)


def info_card(title, value, icon="📊", color="#667eea"):
    """Display an info card with icon"""
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, {color}15 0%, {color}05 100%); 
                border-left: 4px solid {color}; 
                border-radius: 12px; 
                padding: 1.25rem; 
                margin: 0.75rem 0;
                transition: all 0.3s ease;'>
        <div style='display: flex; align-items: center; gap: 1rem;'>
            <span style='font-size: 2rem;'>{icon}</span>
            <div>
                <div style='font-size: 0.85rem; font-weight: 600; color: #718096; text-transform: uppercase; letter-spacing: 0.05em;'>{title}</div>
                <div style='font-size: 1.5rem; font-weight: 700; color: {color}; margin-top: 0.25rem;'>{value}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def section_divider(text=""):
    """Display a section divider with optional text"""
    if text:
        st.markdown(f"""
        <div style='display: flex; align-items: center; margin: 2rem 0 1.5rem;'>
            <div style='flex: 1; height: 2px; background: linear-gradient(to right, #e2e8f0, transparent);'></div>
            <span style='padding: 0 1.5rem; font-size: 0.9rem; font-weight: 600; color: #718096; text-transform: uppercase; letter-spacing: 0.1em;'>{text}</span>
            <div style='flex: 1; height: 2px; background: linear-gradient(to left, #e2e8f0, transparent);'></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='height: 2px; background: linear-gradient(to right, #e2e8f0, transparent, #e2e8f0); margin: 2rem 0;'></div>
        """, unsafe_allow_html=True)


def action_button_row(buttons):
    """Display a row of action buttons with consistent spacing"""
    cols = st.columns(len(buttons))
    for idx, (label, key, callback) in enumerate(buttons):
        with cols[idx]:
            if st.button(label, key=key, width="stretch", type="primary"):
                callback()


def metric_card(label, value, delta=None, icon="📊"):
    """Enhanced metric card with icon and optional delta"""
    delta_html = ""
    if delta:
        delta_color = "#48bb78" if delta > 0 else "#f56565"
        delta_symbol = "▲" if delta > 0 else "▼"
        delta_html = f"<div style='font-size: 0.9rem; color: {delta_color}; font-weight: 600; margin-top: 0.25rem;'>{delta_symbol} {abs(delta):.1f}%</div>"
    
    st.markdown(f"""
    <div style='background: white; 
                border: 1px solid #e2e8f0; 
                border-radius: 12px; 
                padding: 1.5rem; 
                text-align: center;
                transition: all 0.3s ease;
                box-shadow: 0 2px 8px rgba(0,0,0,0.04);'>
        <div style='font-size: 2rem; margin-bottom: 0.75rem;'>{icon}</div>
        <div style='font-size: 0.8rem; font-weight: 600; color: #718096; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;'>{label}</div>
        <div style='font-size: 2rem; font-weight: 700; color: #667eea;'>{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def status_badge(text, status="info"):
    """Display a status badge"""
    colors = {
        "success": ("#48bb78", "#f0fff4"),
        "warning": ("#ed8936", "#fffaf0"),
        "error": ("#f56565", "#fff5f5"),
        "info": ("#667eea", "#f7fafc")
    }
    color, bg = colors.get(status, colors["info"])
    
    st.markdown(f"""
    <span style='display: inline-block; 
                 background: {bg}; 
                 color: {color}; 
                 padding: 0.35rem 0.85rem; 
                 border-radius: 20px; 
                 font-size: 0.85rem; 
                 font-weight: 600;
                 border: 1px solid {color}40;'>{text}</span>
    """, unsafe_allow_html=True)


def progress_steps(steps, current_step):
    """Display progress steps indicator"""
    import streamlit as st
    
    # Build the complete HTML in one string
    steps_parts = []
    
    for idx, step in enumerate(steps):
        is_current = idx == current_step
        is_completed = idx < current_step
        
        if is_completed:
            color = "#48bb78"
            icon = "✓"
        elif is_current:
            color = "#667eea"
            icon = str(idx + 1)
        else:
            color = "#cbd5e0"
            icon = str(idx + 1)
        
        # Build step HTML
        step_html = f"""
            <div style='display: flex; flex-direction: column; align-items: center;'>
                <div style='width: 40px; height: 40px; border-radius: 50%; background: {color}; color: white; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 1.1rem; box-shadow: 0 2px 8px {color}40;'>{icon}</div>
                <div style='font-size: 0.75rem; font-weight: 600; color: {color}; margin-top: 0.5rem; text-align: center;'>{step}</div>
            </div>
        """
        
        # Add connector if not last step
        if idx < len(steps) - 1:
            connector_color = color if is_completed else '#e2e8f0'
            step_html += f"<div style='flex: 1; height: 2px; background: {connector_color}; margin: 0 0.5rem;'></div>"
        
        steps_parts.append(step_html)
    
    # Combine all parts
    full_html = f"""
    <div style='display: flex; align-items: center; margin: 2rem 0; padding: 1.5rem; background: white; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.04);'>
        {''.join(steps_parts)}
    </div>
    """
    
    st.markdown(full_html, unsafe_allow_html=True)
