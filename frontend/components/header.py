"""
Header Component - NO CUSTOM CSS VERSION
Main application header and title
"""
import streamlit as st
from datetime import datetime

def render_header():
    """Render main header - USING STANDARD STREAMLIT ONLY"""
    # REMOVED: Custom CSS markup
    # Now using standard Streamlit
    st.title("🏠 GeoInsight AI")
    st.caption("Complete Real Estate Intelligence Platform")
    st.divider()

def render_page_header(title: str, subtitle: str = None, icon: str = "📊"):
    """Render page-specific header"""
    st.header(f"{icon} {title}")
    if subtitle:
        st.caption(subtitle)

def render_section_header(title: str, icon: str = ""):
    """Render section header"""
    if icon:
        st.subheader(f"{icon} {title}")
    else:
        st.subheader(title)

def render_footer():
    """Render application footer"""
    st.divider()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.caption("🚀 GeoInsight AI v4.2.0")
    
    with col2:
        from api_client import api
        st.caption(f"🔗 {api.base_url}")
    
    with col3:
        st.caption(f"📅 {datetime.now().strftime('%Y-%m-%d')}")
    
    with col4:
        st.caption(f"🕐 {datetime.now().strftime('%H:%M:%S')}")