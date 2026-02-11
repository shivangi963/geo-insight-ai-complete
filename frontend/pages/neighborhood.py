"""
Neighborhood Analysis Page - Simplified Version
Map display features removed
"""
import streamlit as st
from api_client import api
from utils import (
    poll_task_status, format_number, get_walkability_label,
    show_success_message, show_error_message, get_amenity_display_name,
    init_session_state, get_session_state
)
from components.header import render_section_header
from config import map_config, TASK_MAX_WAIT, TASK_POLL_INTERVAL
import plotly.express as px
import requests

def render_neighborhood_page():
    render_section_header("Neighborhood Intelligence")
    
    st.markdown("Analyze any location for amenities, walkability, and local insights using OpenStreetMap data.")
    default_address = get_session_state('nav_to_analysis', '')
    
    # Analysis form
    analysis_triggered = render_analysis_form(default_address)
    
    st.divider()
    
    # Recent analyses
    render_recent_analyses()

def render_analysis_form(default_address: str = '') -> bool:
    """Render form and return True if analysis was triggered"""
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
        
        st.markdown("### 📍 Select Amenities")
        
        amenities_selected = render_amenity_selector()
        
        # Submit button (inside form)
        submitted = st.form_submit_button(
            "🚀 Start Analysis", 
            type="primary", 
            use_container_width=True
        )
    
    # Handle submission OUTSIDE the form
    if submitted:
        if not address:
            show_error_message("Please enter an address")
            return False
        
        if not amenities_selected:
            show_error_message("Select at least one amenity type")
            return False
        
        handle_analysis_submission(address, radius, amenities_selected)
        return True
    
    return False

def render_amenity_selector() -> list:
    """Render amenity checkboxes"""
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

def handle_analysis_submission(address: str, radius: int, amenities: list):
    """Handle analysis submission - OUTSIDE form context"""
    st.divider()
    st.subheader("🔄 Running Analysis")
    
    with st.spinner("Creating analysis task..."):
        response = api.start_neighborhood_analysis({
            "address": address,
            "radius_m": radius,
            "amenity_types": amenities,
            "include_buildings": False,
            "generate_map": True
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
    result = poll_task_status(task_id, max_wait=TASK_MAX_WAIT)
    
    if result:
        # Save to history
        history = get_session_state('analysis_history', [])
        history.append({
            'address': address,
            'analysis_id': analysis_id,
            'walk_score': result.get('walk_score'),
            'total_amenities': result.get('total_amenities')
        })
        st.session_state.analysis_history = history[-10:]
        
        # Display results
        display_analysis_results(result, analysis_id)

def display_analysis_results(result: dict, analysis_id: str, generate_map: bool = True):
    """Display analysis results"""
    st.divider()
    st.subheader("✅ Analysis Results")
    
    render_key_metrics(result)
    render_walkability_interpretation(result.get('walk_score', 0))
    
    amenities = result.get('amenities', {})
    if amenities:
        render_amenities_breakdown(amenities)
    
    if generate_map:
        render_interactive_map(analysis_id)

def render_key_metrics(result: dict):
    """Render key metrics cards"""
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
            st.metric("🌐 Latitude", f"{coords[0]:.4f}")
    
    with col4:
        if coords:
            st.metric("🌐 Longitude", f"{coords[1]:.4f}")

def render_walkability_interpretation(walk_score: float):
    """Display walkability interpretation"""
    st.divider()
    
    label, css_class = get_walkability_label(walk_score)
    
    if walk_score >= 90:
        st.success("🏆 Walker's Paradise! Daily errands do not require a car.")
    elif walk_score >= 70:
        st.success("✅ Very Walkable! Most errands can be accomplished on foot.")
    elif walk_score >= 50:
        st.info("🚶 Somewhat Walkable. Some amenities within walking distance.")
    elif walk_score >= 25:
        st.warning("🚗 Car-Dependent. Most errands require a car.")
    else:
        st.error("🚙 Very Car-Dependent. Almost all errands require a car.")

def render_amenities_breakdown(amenities: dict):
    """Render amenities chart and list"""
    st.divider()
    st.subheader("📊 Amenities Found")
    
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
        
        # Detailed list
        st.markdown("### 📍 Nearby Places")
        
        cols = st.columns(3)
        
        for idx, (atype, items) in enumerate(amenities.items()):
            if items:
                with cols[idx % 3]:
                    display_name = get_amenity_display_name(atype)
                    with st.expander(f"{display_name} ({len(items)})"):
                        for i, item in enumerate(items, 1):
                            name = item.get('name', 'Unknown')
                            dist = item.get('distance_km', 0)
                            st.write(f"**{i}. {name}**")
                            st.caption(f"📏 {dist:.2f} km away")
    else:
        st.info("ℹ️ No amenities found in the search radius")

def render_interactive_map(analysis_id: str):
    """Display interactive map"""
    st.divider()
    st.subheader("🗺️ Interactive Map")
    
    map_url = f"{api.base_url}/api/neighborhood/{analysis_id}/map"
    
    try:
        # Verify analysis
        response = api.get(f"/api/neighborhood/{analysis_id}")
        
        if not response:
            st.error("❌ Could not load analysis data")
            return
        
        status = response.get('status')
        map_path = response.get('map_path')
        
        if status != "completed":
            st.warning(f"⏳ Analysis not completed yet. Status: {status}")
            return
        
        if not map_path:
            st.warning("⚠️ Map was not generated for this analysis")
            return
        
        # Fetch and display map
        with st.spinner("Loading map..."):
            try:
                # Fetch HTML content from backend
                html_response = requests.get(map_url, timeout=10)
                
                if html_response.status_code == 200:
                    html_content = html_response.text
                    
                    # Display using Streamlit's HTML component
                    st.components.v1.html(
                        html_content,
                        height=700,
                        scrolling=True
                    )
                    
                else:
                    st.error(f"❌ Failed to load map: HTTP {html_response.status_code}")
            
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Network error: {e}")
        
    except Exception as e:
        st.error(f"❌ Error: {e}")

def render_recent_analyses():
    """Render recent analyses list"""
    st.subheader("📜 Recent Analyses")
    
    recent = api.get("/api/neighborhood/recent", params={"limit": 10})
    
    if not recent:
        st.info("ℹ️ No recent analyses available")
        return
    
    analyses = recent.get('analyses', [])
    
    if not analyses:
        st.info("📭 No analyses yet. Start your first analysis above!")
        return
    
    for analysis in analyses:
        render_analysis_card(analysis)

def render_analysis_card(analysis: dict):
    """Render analysis card"""
    status = analysis.get('status', 'unknown')
    address = analysis.get('address', 'Unknown')
    
    # Status emoji map
    status_emoji_map = {
        'completed': '✅',
        'processing': '⏳',
        'pending': '🕐',
        'failed': '❌'
    }
    status_emoji = status_emoji_map.get(status, '❓')
    
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
        
        with col3:
            created = analysis.get('created_at', 'N/A')
            st.write(f"**Created:** {created}")
            
            analysis_id = analysis.get('analysis_id')
            if analysis_id and st.button("👁️ View Details", key=f"view_{analysis_id}"):
                full_analysis = api.get_analysis(analysis_id)
                if full_analysis:
                    display_analysis_results(full_analysis, analysis_id)


def display_analysis_results(result: dict, analysis_id: str, generate_map: bool = True):
    """Display analysis results with comparison feature"""
    st.divider()
    st.subheader("✅ Analysis Results")
    
    render_key_metrics(result)
    render_walkability_interpretation(result.get('walk_score', 0))
    
    amenities = result.get('amenities', {})
    if amenities:
        render_amenities_breakdown(amenities)
    
    if generate_map:
        render_interactive_map(analysis_id)
    
    # ✨ NEW: Add "Find Similar" button
    st.divider()
    render_similarity_search_section(analysis_id, result)


def render_similarity_search_section(analysis_id: str, query_analysis: dict):
    """Render similarity search section"""
    st.subheader("🔍 Find Similar Neighborhoods")
    
    st.markdown("""
    Discover neighborhoods with similar characteristics across the entire database.
    Comparison based on amenities, walkability, and urban features.
    """)
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if st.button("🔍 Find Similar Neighborhoods", type="primary", use_container_width=True, key=f"find_similar_{analysis_id}"):
            find_and_display_similar(analysis_id, query_analysis)
    
    with col2:
        limit = st.number_input("Results", min_value=1, max_value=20, value=5, key=f"limit_{analysis_id}")
    
    with col3:
        threshold = st.number_input("Min Match %", min_value=0, max_value=100, value=60, key=f"threshold_{analysis_id}")


def find_and_display_similar(analysis_id: str, query_analysis: dict):
    """Find and display similar neighborhoods"""
    from api_client import api
    import requests
    
    with st.spinner("🔍 Searching for similar neighborhoods..."):
        try:
            # Get similarity threshold from session state
            threshold = st.session_state.get(f"threshold_{analysis_id}", 60) / 100
            limit = st.session_state.get(f"limit_{analysis_id}", 5)
            
            response = requests.get(
                f"{api.base_url}/api/neighborhood/{analysis_id}/similar",
                params={
                    "limit": limit,
                    "threshold": threshold
                },
                timeout=30
            )
            
            if response.status_code != 200:
                st.error(f"❌ Search failed: {response.text}")
                return
            
            result = response.json()
        
        except Exception as e:
            st.error(f"❌ Error: {e}")
            return
    
    # Display results
    report = result.get('report', {})
    similar_neighborhoods = report.get('similar_neighborhoods', [])
    
    if not similar_neighborhoods:
        st.info("ℹ️ No similar neighborhoods found")
        st.caption(f"Try lowering the minimum match percentage (currently {threshold*100:.0f}%)")
        return
    
    st.success(f"✅ Found {len(similar_neighborhoods)} similar neighborhoods!")
    
    # Display comparison
    render_comparison_results(query_analysis, similar_neighborhoods, report)


def render_comparison_results(query_analysis: dict, similar_neighborhoods: list, report: dict):
    """Render detailed comparison results"""
    st.divider()
    st.subheader("📊 Comparison Results")
    
    # Query summary
    query_info = report.get('query', {})
    
    with st.expander("📍 Reference Location (Your Search)", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📍 Address", query_info.get('address', 'N/A'))
        with col2:
            st.metric("🚶 Walk Score", f"{query_info.get('walk_score', 0):.1f}")
        with col3:
            st.metric("📌 Amenities", query_info.get('total_amenities', 0))
    
    # Similar neighborhoods
    st.markdown("### 🏘️ Similar Neighborhoods")
    
    for idx, neighborhood in enumerate(similar_neighborhoods, 1):
        render_similarity_card(neighborhood, idx, query_info)


def render_similarity_card(neighborhood: dict, idx: int, query_info: dict):
    """Render individual similarity result card"""
    similarity = neighborhood.get('similarity_score', 0) * 100
    address = neighborhood.get('address', 'Unknown')
    
    with st.expander(f"#{idx} - {address} | {similarity:.1f}% Match", expanded=(idx == 1)):
        # Similarity progress bar
        st.progress(neighborhood.get('similarity_score', 0))
        
        # Metrics comparison
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            walk_score = neighborhood.get('walk_score', 0)
            walk_diff = neighborhood.get('walk_score_diff', 0)
            st.metric(
                "🚶 Walk Score",
                f"{walk_score:.1f}",
                delta=f"{walk_diff:+.1f}" if walk_diff != 0 else None
            )
        
        with col2:
            total_amenities = neighborhood.get('total_amenities', 0)
            query_total = query_info.get('total_amenities', 0)
            amenity_diff = total_amenities - query_total
            st.metric(
                "📍 Amenities",
                total_amenities,
                delta=f"{amenity_diff:+d}" if amenity_diff != 0 else None
            )
        
        with col3:
            st.metric("🎯 Match", f"{similarity:.1f}%")
        
        with col4:
            analysis_id = neighborhood.get('analysis_id')
            if st.button("👁️ View Details", key=f"view_{idx}_{analysis_id}"):
                # Open analysis in new expander
                from api_client import api
                full_analysis = api.get_analysis(analysis_id)
                if full_analysis:
                    st.session_state[f'show_analysis_{analysis_id}'] = True
        
        # Key differences
        differences = neighborhood.get('key_differences', [])
        if differences:
            st.markdown("**🔍 Key Differences:**")
            for diff in differences:
                st.write(f"• {diff}")
        
        # Amenity breakdown comparison
        with st.expander("📊 Detailed Amenity Comparison"):
            render_amenity_comparison(
                query_info.get('amenity_breakdown', {}),
                neighborhood.get('amenity_breakdown', {})
            )
        
        # Map preview
        map_path = neighborhood.get('map_path')
        coordinates = neighborhood.get('coordinates')
        
        if map_path or coordinates:
            if st.button("🗺️ Show Map", key=f"map_{idx}_{analysis_id}"):
                if map_path:
                    # Show saved map
                    analysis_id = neighborhood.get('analysis_id')
                    render_interactive_map(analysis_id)
                elif coordinates:
                    # Generate quick map
                    st.info("📍 Map generation coming soon")


def render_amenity_comparison(query_amenities: dict, candidate_amenities: dict):
    """Render side-by-side amenity comparison"""
    import pandas as pd
    
    # Combine amenity types
    all_types = set(list(query_amenities.keys()) + list(candidate_amenities.keys()))
    
    comparison_data = []
    for amenity_type in sorted(all_types):
        query_count = query_amenities.get(amenity_type, 0)
        candidate_count = candidate_amenities.get(amenity_type, 0)
        diff = candidate_count - query_count
        
        comparison_data.append({
            'Amenity': amenity_type.replace('_', ' ').title(),
            'Reference': query_count,
            'Similar': candidate_count,
            'Difference': f"{diff:+d}" if diff != 0 else "Same"
        })
    
    df = pd.DataFrame(comparison_data)
    st.dataframe(df, use_container_width=True, hide_index=True)