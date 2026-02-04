"""
GeoInsight AI - Main Streamlit Application
Fixed version without custom CSS dependency
"""
import streamlit as st
import sys
import os

# Add frontend directory to path if needed
if os.path.exists('frontend'):
    sys.path.insert(0, 'frontend')

# Import configuration
try:
    from config import ui_config
except ImportError:
    # Fallback configuration
    class UIConfig:
        page_title = "GeoInsight AI - Real Estate Intelligence"
        page_icon = "🏠"
        layout = "wide"
        initial_sidebar_state = "expanded"
    ui_config = UIConfig()

# Configure page
st.set_page_config(
    page_title=ui_config.page_title,
    page_icon=ui_config.page_icon,
    layout=ui_config.layout,
    initial_sidebar_state=ui_config.initial_sidebar_state
)

# Initialize session state
def init_session_state(key: str, default_value):
    """Initialize session state variable if not exists"""
    if key not in st.session_state:
        st.session_state[key] = default_value

init_session_state('analysis_history', [])
init_session_state('agent_history', [])
init_session_state('nav_to_analysis', '')
init_session_state('ai_query', '')
init_session_state('show_ai_history', False)

# Import components
try:
    from components.sidebar import render_sidebar
    from components.header import render_header, render_footer
    HAS_COMPONENTS = True
except ImportError:
    HAS_COMPONENTS = False
    st.warning("⚠️ Component imports failed. Running in basic mode.")

# Render sidebar
if HAS_COMPONENTS:
    render_sidebar()
else:
    st.sidebar.title("GeoInsight AI")
    st.sidebar.info("Control Panel")

# Render header
if HAS_COMPONENTS:
    render_header()
else:
    st.title("🏠 GeoInsight AI")
    st.caption("Complete Real Estate Intelligence Platform")
    st.divider()

# Main content tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏘️ Properties",
    "🗺️ Neighborhood",
    "🤖 AI Assistant",
    "📸 Image Analysis",
    "🔍 Vector Search",
    "📊 Dashboard"
])

# Tab 1: Properties
with tab1:
    try:
        from pages.properties import render_properties_page
        render_properties_page()
    except Exception as e:
        st.error(f"❌ Error: {e}")

with tab2:
    try:
        from pages.neighborhood import render_neighborhood_page
        render_neighborhood_page()
    except Exception as e:
        st.error(f"❌ Error: {e}")

with tab3:
    try:
        from pages.ai_assistant import render_ai_assistant_page
        render_ai_assistant_page()
    except Exception as e:
        st.error(f"❌ Error: {e}")

with tab4:
    try:
        from pages.image_analysis import render_image_analysis_page
        render_image_analysis_page()
    except Exception as e:
        st.error(f"❌ Error: {e}")

with tab5:
    try:
        from pages.vector_search import render_vector_search_page
        render_vector_search_page()
    except Exception as e:
        st.error(f"❌ Error: {e}")

with tab6:
    try:
        from pages.dashboard import render_dashboard_page
        render_dashboard_page()
    except Exception as e:
        st.error(f"❌ Error: {e}")

# Render footer
if HAS_COMPONENTS:
    render_footer()
else:
    st.divider()
    st.caption("GeoInsight AI")