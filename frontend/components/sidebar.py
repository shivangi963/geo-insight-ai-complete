"""
Sidebar Component
Application sidebar with stats and controls
"""
import streamlit as st
from api_client import api
from utils import format_currency, format_number

def render_sidebar():
    """Render complete sidebar"""
    with st.sidebar:
        # Logo/Header
        st.image("https://img.icons8.com/fluency/96/city.png", width=100)
        st.title("⚙️ Control Panel")
        
        # Connection Status Section
        render_connection_status()
        
        st.divider()
        
        # Quick Stats Section
        render_quick_stats()
        
        st.divider()
        
        # Features Status Section
        render_features_status()
        
        st.divider()
        
        # Quick Actions
        render_quick_actions()

def render_connection_status():
    """Render backend connection status"""
    st.subheader("🔗 Connection")
    
    if st.button("🔍 Test Backend", use_container_width=True):
        with st.spinner("Testing..."):
            health = api.health_check()
            if health:
                st.success("✅ Connected!")
                
                # Show backend info in expander
                with st.expander("📊 Backend Info"):
                    st.json({
                        "Version": health.get('version'),
                        "Status": health.get('status'),
                        "Database": health.get('database'),
                        "Features": health.get('features', {})
                    })
            else:
                st.error("❌ Offline")
                st.caption(f"Backend URL: {api.base_url}")

def render_quick_stats():
    """Render live statistics"""
    st.subheader("📊 Live Stats")
    
    stats = api.get_stats()
    if stats:
        # Properties count
        total_props = stats.get('total_properties', 0)
        st.metric(
            "🏠 Properties", 
            format_number(total_props),
            help="Total properties in database"
        )
        
        # Analyses count
        total_analyses = stats.get('total_analyses', 0)
        st.metric(
            "🗺️ Analyses", 
            format_number(total_analyses),
            help="Total neighborhood analyses"
        )
        
        # Cities count
        unique_cities = stats.get('unique_cities', 0)
        st.metric(
            "🏙️ Cities", 
            format_number(unique_cities),
            help="Unique cities covered"
        )
        
        # Average price
        avg_price = stats.get('average_price', 0)
        if avg_price > 0:
            st.metric(
                "💰 Avg Price", 
                format_currency(avg_price),
                help="Average property price"
            )
        
        # System status
        system_status = stats.get('system_status', 'unknown')
        status_emoji = "✅" if system_status == 'healthy' else "⚠️"
        st.caption(f"{status_emoji} System: {system_status.title()}")
        
        # Uptime
        uptime = stats.get('uptime', 'N/A')
        st.caption(f"⏱️ Uptime: {uptime}")
    else:
        st.info("Stats unavailable")

def render_features_status():
    """Render features availability status"""
    st.subheader("🔧 Features")
    
    health = api.health_check()
    if health:
        features = health.get('features', {})
        
        # Create feature status list
        feature_list = [
            ("Async Tasks", features.get('celery', False)),
            ("Vector Search", features.get('vector_db', False)),
            ("AI Agent", features.get('ai_agent', False)),
            ("Geospatial", features.get('geospatial', True)),
            ("Rate Limiting", features.get('rate_limiting', False))
        ]
        
        for feature_name, is_enabled in feature_list:
            status_emoji = "✅" if is_enabled else "⚠️"
            st.write(f"{status_emoji} {feature_name}")
    else:
        st.warning("⚠️ Cannot check features")

def render_quick_actions():
    """Render quick action buttons"""
    st.subheader("⚡ Quick Actions")
    
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    if st.button("🗑️ Clear Cache", use_container_width=True):
        st.cache_data.clear()
        st.success("✅ Cache cleared!")
    
    # Settings expander
    with st.expander("⚙️ Settings"):
        st.checkbox("Show debug info", value=False, key="show_debug")
        st.checkbox("Auto-refresh stats", value=False, key="auto_refresh")
        st.number_input("Refresh interval (s)", min_value=5, max_value=60, value=30, key="refresh_interval")