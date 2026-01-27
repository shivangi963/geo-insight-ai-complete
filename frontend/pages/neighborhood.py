"""
Neighborhood Analysis Page
Analyze locations for amenities and walkability
"""
import streamlit as st
from api_client import api
from utils import (
    poll_task_status, format_number, get_walkability_label,
    show_success_message, show_error_message, get_amenity_display_name,
    init_session_state, get_session_state
)
from components.header import render_section_header
from config import map_config
import plotly.express as px

def render_neighborhood_page():
    """Main neighborhood analysis page"""
    render_section_header("Neighborhood Intelligence", "🗺️")
    
    st.markdown("Analyze any location for amenities, walkability, and local insights using OpenStreetMap data.")
    
    # Check if navigated from properties
    default_address = get_session_state('nav_to_analysis', '')
    
    # Analysis form
    render_analysis_form(default_address)
    
    st.divider()
    
    # Recent analyses
    render_recent_analyses()

def render_analysis_form(default_address: str = ''):
    """Render the analysis request form"""
    with st.form("neighborhood_form"):
        address = st.text_input(
            "📍 Enter Full Address",
            value=default_address,
            placeholder="e.g., MIT Campus, Manipal, Karnataka, India",
            help="More specific = better results"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            radius = st.slider(
                "🔍 Search Radius (meters)", 
                map_config.min_radius, 
                map_config.max_radius, 
                map_config.default_radius, 
                100
            )
        
        with col2:
            generate_map = st.checkbox("🗺️ Generate Interactive Map", value=True)
        
        # Amenity selection
        st.markdown("### 🎯 Select Amenities")
        
        amenities_selected = render_amenity_selector()
        
        submitted = st.form_submit_button(
            "🚀 Start Analysis", 
            type="primary", 
            use_container_width=True
        )
        
        if submitted:
            handle_analysis_submission(address, radius, amenities_selected, generate_map)

def render_amenity_selector() -> list:
    """Render amenity type selector and return selected types"""
    amenity_options = {
        "🍽️ Restaurants": "restaurant",
        "☕ Cafes": "cafe",
        "🏫 Schools": "school",
        "🏥 Hospitals": "hospital",
        "🌳 Parks": "park",
        "🛒 Supermarkets": "supermarket",
        "🏦 Banks": "bank",
        "💊 Pharmacies": "pharmacy",
        "💪 Gyms": "gym",
        "📚 Libraries": "library",
        "🚇 Transit": "transit_station"
    }
    
    cols = st.columns(4)
    amenities_selected = []
    
    default_types = ['restaurant', 'cafe', 'school', 'hospital', 'park', 'supermarket']
    
    for idx, (label, value) in enumerate(amenity_options.items()):
        with cols[idx % 4]:
            default = value in default_types
            if st.checkbox(label, value=default, key=f"amenity_{value}"):
                amenities_selected.append(value)
    
    return amenities_selected

def handle_analysis_submission(address: str, radius: int, amenities: list, generate_map: bool):
    """Handle form submission and start analysis"""
    if not address:
        show_error_message("Please enter an address")
        return
    
    if not amenities:
        show_error_message("Select at least one amenity type")
        return
    
    st.divider()
    st.subheader("⚙️ Running Analysis")
    
    # Start analysis
    with st.spinner("Creating analysis task..."):
        response = api.start_neighborhood_analysis({
            "address": address,
            "radius_m": radius,
            "amenity_types": amenities,
            "include_buildings": False,
            "generate_map": generate_map
        })
    
    if not response:
        return
    
    analysis_id = response.get('analysis_id')
    task_id = response.get('task_id')
    
    show_success_message("Analysis started!")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Analysis ID:** `{analysis_id}`")
    with col2:
        st.info(f"**Task ID:** `{task_id}`")
    
    # Poll for results
    result = poll_task_status(task_id, max_wait=120)
    
    if result:
        # Save to history
        history = get_session_state('analysis_history', [])
        history.append({
            'address': address,
            'analysis_id': analysis_id,
            'walk_score': result.get('walk_score'),
            'total_amenities': result.get('total_amenities')
        })
        st.session_state.analysis_history = history[-10:]  # Keep last 10
        
        # Display results
        display_analysis_results(result, analysis_id, generate_map)

def display_analysis_results(result: dict, analysis_id: str, generate_map: bool):
    """Display the analysis results"""
    st.divider()
    st.subheader("📊 Analysis Results")
    
    # Key Metrics
    render_key_metrics(result)
    
    # Walkability interpretation
    render_walkability_interpretation(result.get('walk_score', 0))
    
    # Amenities breakdown
    amenities = result.get('amenities', {})
    if amenities:
        render_amenities_breakdown(amenities)
    
    # Interactive Map
    if generate_map:
        render_interactive_map(analysis_id)

def render_key_metrics(result: dict):
    """Render key metrics row"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        walk_score = result.get('walk_score', 0)
        st.metric("🚶 Walk Score", f"{walk_score:.1f}/100")
    
    with col2:
        total = result.get('total_amenities', 0)
        st.metric("📍 Amenities", format_number(total))
    
    with col3:
        coords = result.get('coordinates')
        if coords:
            st.metric("🌍 Latitude", f"{coords[0]:.4f}")
    
    with col4:
        if coords:
            st.metric("🌍 Longitude", f"{coords[1]:.4f}")

def render_walkability_interpretation(walk_score: float):
    """Display walkability interpretation"""
    st.divider()
    
    label, css_class = get_walkability_label(walk_score)
    
    if walk_score >= 90:
        st.success("🌟 **Walker's Paradise!** Daily errands do not require a car.")
    elif walk_score >= 70:
        st.success("✅ **Very Walkable!** Most errands can be accomplished on foot.")
    elif walk_score >= 50:
        st.info("ℹ️ **Somewhat Walkable.** Some amenities within walking distance.")
    elif walk_score >= 25:
        st.warning("⚠️ **Car-Dependent.** Most errands require a car.")
    else:
        st.error("❌ **Very Car-Dependent.** Almost all errands require a car.")

def render_amenities_breakdown(amenities: dict):
    """Render amenities breakdown with charts and lists"""
    st.divider()
    st.subheader("🎯 Amenities Found")
    
    # Create chart data
    amenity_counts = {k.replace('_', ' ').title(): len(v) for k, v in amenities.items() if v}
    
    if amenity_counts:
        # Bar chart
        fig = px.bar(
            x=list(amenity_counts.keys()),
            y=list(amenity_counts.values()),
            labels={'x': 'Amenity Type', 'y': 'Count'},
            title="Amenity Distribution",
            color=list(amenity_counts.values()),
            color_continuous_scale='viridis'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed lists
        st.markdown("### 📋 Nearby Places")
        
        cols = st.columns(3)
        
        for idx, (atype, items) in enumerate(amenities.items()):
            if items:
                with cols[idx % 3]:
                    display_name = get_amenity_display_name(atype)
                    with st.expander(f"{display_name} ({len(items)})"):
                        for i, item in enumerate(items[:10], 1):
                            name = item.get('name', 'Unknown')
                            dist = item.get('distance_km', 0)
                            st.write(f"**{i}. {name}**")
                            st.caption(f"   📏 {dist:.2f} km away")
    else:
        st.info("No amenities found in the search radius")

def render_interactive_map(analysis_id: str):
    """Render interactive map iframe"""
    st.divider()
    st.subheader("🗺️ Interactive Map")
    
    map_url = f"{api.base_url}/api/neighborhood/{analysis_id}/map"
    
    try:
        st.components.v1.iframe(map_url, height=600, scrolling=True)
    except Exception as e:
        st.error(f"Map display error: {e}")
        st.info(f"[📍 Open map in new tab]({map_url})")

def render_recent_analyses():
    """Render recent analyses list"""
    st.subheader("📜 Recent Analyses")
    
    recent = api.get("/api/neighborhood/recent", params={"limit": 10})
    
    if not recent:
        st.info("No recent analyses")
        return
    
    analyses = recent.get('analyses', [])
    
    if not analyses:
        st.info("No analyses yet. Start your first analysis above!")
        return
    
    for analysis in analyses:
        render_analysis_card(analysis)

def render_analysis_card(analysis: dict):
    """Render individual analysis card"""
    status = analysis.get('status', 'unknown')
    address = analysis.get('address', 'Unknown')
    
    status_emoji = {
        'completed': '✅',
        'processing': '⏳',
        'pending': '🔄',
        'failed': '❌'
    }.get(status, '❓')
    
    with st.expander(f"{status_emoji} {address}"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write(f"**Status:** {status.title()}")
            walk_score = analysis.get('walk_score')
            if walk_score:
                st.write(f"**Walk Score:** {walk_score:.1f}/100")
        
        with col2:
            total = analysis.get('total_amenities', 0)
            st.write(f"**Amenities:** {total}")
            has_map = analysis.get('map_available', False)
            st.write(f"**Map:** {'✅' if has_map else '❌'}")
        
        with col3:
            created = analysis.get('created_at', 'N/A')
            st.write(f"**Created:** {created}")
            
            analysis_id = analysis.get('analysis_id')
            if analysis_id and st.button("📊 View Full", key=f"view_{analysis_id}"):
                # Load full analysis
                full_analysis = api.get_analysis(analysis_id)
                if full_analysis:
                    display_analysis_results(full_analysis, analysis_id, True)