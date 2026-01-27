"""
Main Streamlit Application - NO CUSTOM CSS VERSION
Entry point for GeoInsight AI frontend
"""
import streamlit as st
from config import ui_config
# REMOVED: from styles import get_custom_css  ← DELETE THIS LINE
from components.sidebar import render_sidebar
from components.header import render_header, render_footer
from utils import init_session_state

# Configure page
st.set_page_config(
    page_title=ui_config.page_title,
    page_icon=ui_config.page_icon,
    layout=ui_config.layout,
    initial_sidebar_state=ui_config.initial_sidebar_state
)

# REMOVED: Apply custom CSS  ← DELETE THESE 2 LINES
# st.markdown(get_custom_css(), unsafe_allow_html=True)

# Initialize session state
init_session_state('analysis_history', [])
init_session_state('agent_history', [])
init_session_state('nav_to_analysis', '')
init_session_state('ai_query', '')

# Render sidebar
render_sidebar()

# Render header
render_header()

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
    from pages.properties import render_properties_page
    render_properties_page()

# Tab 2: Neighborhood Analysis
with tab2:
    from pages.neighborhood import render_neighborhood_page
    render_neighborhood_page()

# Tab 3: AI Assistant
with tab3:
    from pages.ai_assistant import render_ai_assistant_page
    render_ai_assistant_page()

# Tab 4: Image Analysis
with tab4:
    from pages.image_analysis import render_image_analysis_page
    render_image_analysis_page()

# Tab 5: Vector Search
with tab5:
    from pages.vector_search import render_vector_search_page
    render_vector_search_page()

# Tab 6: Dashboard
with tab6:
    from pages.dashboard import render_dashboard_page
    render_dashboard_page()

# Render footer
render_footer()