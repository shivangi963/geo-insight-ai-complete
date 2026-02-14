"""
Neighborhood Intelligence Page
Three independent tools:
  1. Neighborhood Analysis  – run a full OSM + walk-score analysis
  2. Green Space Analysis    – standalone address → green-space calculator
  3. Find Similar             – pick any past analysis and find matches
"""
import streamlit as st
import requests
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Dict

from api_client import api
from utils import (
    poll_task_status, format_number, get_walkability_label,
    show_success_message, show_error_message, get_amenity_display_name,
    init_session_state, get_session_state, format_percentage,
    validate_file_size
)
from components.header import render_section_header
from config import map_config, TASK_MAX_WAIT, TASK_POLL_INTERVAL


# ──────────────────────────────────────────────
# PAGE ENTRY POINT
# ──────────────────────────────────────────────

def render_neighborhood_page():
    render_section_header("Neighborhood Intelligence")
    st.markdown(
        "Three independent tools — run any of them without needing the others first."
    )

    tab1, tab2, tab3 = st.tabs([
        "🗺️ Neighborhood Analysis",
        "🌳 Green Space Analysis",
        "🔍 Find Similar Neighborhoods",
    ])

    with tab1:
        _render_analysis_tab()

    with tab2:
        _render_green_space_tab()

    with tab3:
        _render_find_similar_tab()


# ══════════════════════════════════════════════
# TAB 1 – NEIGHBORHOOD ANALYSIS
# ══════════════════════════════════════════════

def _render_analysis_tab():
    st.subheader("🗺️ Full Neighborhood Analysis")
    st.markdown(
        "Fetch amenities from OpenStreetMap, compute a walk score, "
        "and generate an interactive map for any address."
    )

    default_address = get_session_state("nav_to_analysis", "")
    triggered = _render_analysis_form(default_address)

    if not triggered and "current_analysis" in st.session_state:
        stored = st.session_state.current_analysis
        _display_analysis_results(stored["result"], stored["analysis_id"])

    st.divider()
    _render_recent_analyses()


def _render_analysis_form(default_address: str = "") -> bool:
    with st.form("neighborhood_form"):
        address = st.text_input(
            "📍 Enter Full Address",
            value=default_address,
            placeholder="e.g., MIT Campus, Manipal, Karnataka, India",
            help="More specific = better results",
        )

        col1, _ = st.columns(2)
        with col1:
            radius = st.slider(
                "🔍 Search Radius (metres)",
                map_config.min_radius,
                map_config.max_radius,
                map_config.default_radius,
                100,
            )

        st.markdown("### 📍 Select Amenities")
        amenities_selected = _render_amenity_selector()

        submitted = st.form_submit_button(
            "🚀 Start Analysis", type="primary", use_container_width=True
        )

    if submitted:
        if not address:
            show_error_message("Please enter an address")
            return False
        if not amenities_selected:
            show_error_message("Select at least one amenity type")
            return False
        if "current_analysis" in st.session_state:
            del st.session_state["current_analysis"]
        _handle_analysis_submission(address, radius, amenities_selected)
        return True
    return False


def _render_amenity_selector() -> list:
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
        "🚇 Transit": "transit_station",
    }
    cols = st.columns(4)
    default_types = ["restaurant", "cafe", "school", "hospital", "park", "supermarket"]
    selected = []
    for idx, (label, value) in enumerate(amenity_options.items()):
        with cols[idx % 4]:
            if st.checkbox(label, value=(value in default_types), key=f"amenity_{value}"):
                selected.append(value)
    return selected


def _handle_analysis_submission(address: str, radius: int, amenities: list):
    st.divider()
    st.subheader("🔄 Running Analysis")

    with st.spinner("Creating analysis task…"):
        response = api.start_neighborhood_analysis({
            "address": address,
            "radius_m": radius,
            "amenity_types": amenities,
            "include_buildings": False,
            "generate_map": True,
        })

    if not response:
        return

    analysis_id = response.get("analysis_id")
    task_id = response.get("task_id")
    show_success_message("Analysis started!")

    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Analysis ID:** `{analysis_id}`")
    with col2:
        st.info(f"**Task ID:** `{task_id}`")

    result = poll_task_status(task_id, max_wait=TASK_MAX_WAIT)

    if result:
        history = get_session_state("analysis_history", [])
        history.append({
            "address": address,
            "analysis_id": analysis_id,
            "walk_score": result.get("walk_score"),
            "total_amenities": result.get("total_amenities"),
        })
        st.session_state.analysis_history = history[-10:]
        st.session_state.current_analysis = {"result": result, "analysis_id": analysis_id}
        _display_analysis_results(result, analysis_id)


def _display_analysis_results(result: dict, analysis_id: str, generate_map: bool = True):
    st.divider()
    col1, col2 = st.columns([4, 1])
    with col1:
        st.subheader("✅ Analysis Results")
    with col2:
        if st.button("🗑️ Clear Results", key="clear_analysis"):
            if "current_analysis" in st.session_state:
                del st.session_state["current_analysis"]
            st.rerun()

    _render_key_metrics(result)
    _render_walkability_interpretation(result.get("walk_score", 0))

    amenities = result.get("amenities", {})
    if amenities:
        _render_amenities_breakdown(amenities)

    if generate_map:
        _render_interactive_map(analysis_id)


def _render_key_metrics(result: dict):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🚶 Walk Score", f"{result.get('walk_score', 0):.1f}/100")
    with col2:
        st.metric("📍 Amenities", format_number(result.get("total_amenities", 0)))
    coords = result.get("coordinates")
    with col3:
        st.metric("🌐 Latitude", f"{coords[0]:.4f}" if coords else "—")
    with col4:
        st.metric("🌐 Longitude", f"{coords[1]:.4f}" if coords else "—")


def _render_walkability_interpretation(walk_score: float):
    st.divider()
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


def _render_amenities_breakdown(amenities: dict):
    st.divider()
    st.subheader("📊 Amenities Found")
    amenity_counts = {k.replace("_", " ").title(): len(v) for k, v in amenities.items() if v}
    if amenity_counts:
        fig = px.bar(
            x=list(amenity_counts.keys()),
            y=list(amenity_counts.values()),
            labels={"x": "Amenity Type", "y": "Count"},
            title="Amenity Distribution",
            color=list(amenity_counts.values()),
            color_continuous_scale="viridis",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📍 Nearby Places")
        cols = st.columns(3)
        for idx, (atype, items) in enumerate(amenities.items()):
            if items:
                with cols[idx % 3]:
                    display_name = get_amenity_display_name(atype)
                    with st.expander(f"{display_name} ({len(items)})"):
                        for i, item in enumerate(items, 1):
                            st.write(f"**{i}. {item.get('name', 'Unknown')}**")
                            st.caption(f"📏 {item.get('distance_km', 0):.2f} km away")
    else:
        st.info("ℹ️ No amenities found in the search radius")


def _render_interactive_map(analysis_id: str):
    st.divider()
    st.subheader("🗺️ Interactive Map")
    map_url = f"{api.base_url}/api/neighborhood/{analysis_id}/map"
    response = api.get(f"/api/neighborhood/{analysis_id}")
    if not response:
        st.error("❌ Could not load analysis data")
        return
    if response.get("status") != "completed":
        st.warning(f"⏳ Analysis not completed yet. Status: {response.get('status')}")
        return
    if not response.get("map_path"):
        st.warning("⚠️ Map was not generated for this analysis")
        return
    with st.spinner("Loading map…"):
        try:
            html_response = requests.get(map_url, timeout=10)
            if html_response.status_code == 200:
                st.components.v1.html(html_response.text, height=700, scrolling=True)
            else:
                st.error(f"❌ Failed to load map: HTTP {html_response.status_code}")
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Network error: {e}")


def _render_recent_analyses():
    st.subheader("📜 Recent Analyses")
    recent = api.get("/api/neighborhood/recent", params={"limit": 10})
    if not recent:
        st.info("ℹ️ No recent analyses available")
        return
    analyses = recent.get("analyses", [])
    if not analyses:
        st.info("📭 No analyses yet. Start your first analysis above!")
        return
    for analysis in analyses:
        _render_analysis_card(analysis)


def _render_analysis_card(analysis: dict):
    status = analysis.get("status", "unknown")
    address = analysis.get("address", "Unknown")
    status_emoji = {"completed": "✅", "processing": "⏳", "pending": "🕐", "failed": "❌"}.get(status, "❓")
    with st.expander(f"{status_emoji} {address}"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Status:** {status.title()}")
            if ws := analysis.get("walk_score"):
                st.write(f"**Walk Score:** {ws:.1f}/100")
        with col2:
            st.write(f"**Amenities:** {analysis.get('total_amenities', 0)}")
        with col3:
            st.write(f"**Created:** {analysis.get('created_at', 'N/A')}")
            if aid := analysis.get("analysis_id"):
                if st.button("👁️ View Details", key=f"view_{aid}"):
                    full = api.get_analysis(aid)
                    if full:
                        _display_analysis_results(full, aid)


# ══════════════════════════════════════════════
# TAB 2 – GREEN SPACE ANALYSIS  (standalone)
# ══════════════════════════════════════════════

def _render_green_space_tab():
    """Standalone green space analysis — no prior neighborhood analysis needed."""
    st.subheader("🌳 Green Space Coverage Analysis")
    st.markdown(
        "Enter **any address** to analyse green space coverage using "
        "OpenStreetMap tile data. No prior analysis required."
    )

    st.info(
        "**Detects:** 🌱 Parks & Grass · 🌲 Forests & Woods · "
        "⚽ Recreation Areas · 🌿 Natural Spaces"
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        address = st.text_input(
            "📍 Address",
            placeholder="e.g., Cubbon Park, Bengaluru, Karnataka, India",
            help="Enter any address to analyse green coverage",
            key="gs_standalone_address",
        )
    with col2:
        radius = st.slider(
            "Search Radius (m)", 100, 4000, 500, 100, key="gs_standalone_radius"
        )

    if st.button("🚀 Analyse Green Space", type="primary", use_container_width=True, key="gs_run"):
        if not address:
            show_error_message("Please enter an address")
        else:
            result = _run_green_space_analysis(address, radius)
            if result:
                _display_green_space_results(result)

    st.divider()
    st.subheader("📋 Recent Green Space Analyses")
    _display_recent_green_analyses()


def _run_green_space_analysis(address: str, radius: int) -> Optional[Dict]:
    st.divider()
    st.subheader("🔄 Running Analysis")
    with st.spinner("🌍 Starting green space analysis…"):
        try:
            response = requests.post(
                f"{api.base_url}/api/analysis/green-space",
                params={"address": address, "radius_m": radius},
                headers={"Content-Type": "application/json"},
                json={},
                timeout=10,
            )
        except requests.exceptions.RequestException as e:
            show_error_message(f"Network error: {e}")
            return None

    if response.status_code != 202:
        show_error_message(f"Failed to start analysis: {response.text}")
        return None

    data = response.json()
    analysis_id = data.get("analysis_id")
    task_id = data.get("task_id")
    if not analysis_id or not task_id:
        show_error_message("Invalid response from server")
        return None

    show_success_message("Analysis started!")
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Analysis ID:** `{analysis_id}`")
    with col2:
        st.info(f"**Task ID:** `{task_id}`")

    st.info("⏳ This may take 30–60 seconds…")
    result = poll_task_status(task_id, max_wait=TASK_MAX_WAIT)
    if result:
        full = requests.get(f"{api.base_url}/api/analysis/green-space/{analysis_id}", timeout=10)
        if full.status_code == 200:
            return full.json()
    return None


def _display_green_space_results(result: Dict):
    import io
    from PIL import Image as PILImage

    st.divider()
    st.subheader("✅ Analysis Complete")

    green_pct        = result.get("green_space_percentage", 0)
    green_pixels     = result.get("green_pixels", 0)
    total_pixels     = result.get("total_pixels", 0)
    breakdown        = result.get("breakdown", {})
    viz_path         = result.get("visualization_path")
    address          = result.get("address", "Unknown")
    coordinates      = result.get("coordinates", {})

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🌳 Green Coverage", f"{green_pct:.1f}%")
    with col2:
        st.metric("🟢 Green Pixels", f"{green_pixels:,}")
    with col3:
        st.metric("📐 Total Pixels", f"{total_pixels:,}")

    st.markdown("### 🎯 Coverage Gauge")
    st.plotly_chart(_create_green_gauge(green_pct), use_container_width=True)

    interp = _green_interpretation(green_pct)
    st.info(f"**Interpretation:** {interp}")

    if breakdown and sum(breakdown.values()) > 0:
        st.markdown("### 🌲 Breakdown by Type")
        labels = {
            "parks_grass":   ("Parks/Grass",   "🌱"),
            "forests_woods": ("Forests/Woods", "🌲"),
            "recreation":    ("Recreation",    "⚽"),
            "natural_areas": ("Natural Areas", "🌿"),
        }
        bcols = st.columns(4)
        for idx, (key, pct) in enumerate(breakdown.items()):
            label, icon = labels.get(key, (key.replace("_", " ").title(), "🟢"))
            with bcols[idx % 4]:
                st.metric(f"{icon} {label}", f"{pct:.1f}%")
        st.plotly_chart(_create_breakdown_chart(breakdown), use_container_width=True)

    if viz_path:
        st.markdown("### 🗺️ Visual Analysis")
        try:
            viz_response = requests.get(f"{api.base_url}/{viz_path}", timeout=10)
            if viz_response.status_code == 200:
                img = PILImage.open(io.BytesIO(viz_response.content))
                st.image(img, caption="Green spaces highlighted", width=400)
            else:
                st.warning("Visualization image not available")
        except Exception as e:
            st.warning(f"Could not load visualization: {e}")

    with st.expander("📍 Location Details"):
        st.write(f"**Address:** {address}")
        if coordinates:
            st.write(f"**Latitude:** {coordinates.get('latitude', 'N/A')}")
            st.write(f"**Longitude:** {coordinates.get('longitude', 'N/A')}")
        st.write(f"**Search Radius:** {result.get('search_radius_m', 'N/A')} m")
        st.write("**Data Source:** OpenStreetMap")


def _create_green_gauge(percentage: float) -> go.Figure:
    color = "#28a745" if percentage >= 50 else "#ffc107" if percentage >= 30 else "#dc3545"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=percentage,
        title={"text": "Green Coverage %", "font": {"size": 22}},
        number={"suffix": "%", "font": {"size": 36}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 20],  "color": "#ffe6e6"},
                {"range": [20, 40], "color": "#fff4e6"},
                {"range": [40, 60], "color": "#ffffcc"},
                {"range": [60, 80], "color": "#e6ffe6"},
                {"range": [80, 100],"color": "#ccffcc"},
            ],
            "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 50},
        },
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def _create_breakdown_chart(breakdown: Dict[str, float]) -> go.Figure:
    labels_map  = {"parks_grass": "Parks/Grass", "forests_woods": "Forests/Woods",
                   "recreation": "Recreation", "natural_areas": "Natural Areas"}
    colors_map  = {"parks_grass": "#90EE90", "forests_woods": "#228B22",
                   "recreation": "#3CB371", "natural_areas": "#6B8E23"}
    labels  = [labels_map.get(k, k) for k in breakdown]
    values  = list(breakdown.values())
    colors  = [colors_map.get(k, "#00FF00") for k in breakdown]
    fig = go.Figure(data=[go.Bar(
        x=labels, y=values, marker_color=colors,
        text=[f"{v:.1f}%" for v in values], textposition="auto",
    )])
    fig.update_layout(title="Green Space Breakdown", xaxis_title="Green Type",
                      yaxis_title="Coverage (%)", height=380, showlegend=False)
    return fig


def _green_interpretation(pct: float) -> str:
    if pct >= 60:
        return "🌲 Excellent! Abundant green coverage — parks, forests, and natural spaces."
    elif pct >= 40:
        return "🌳 Good green coverage. Healthy amount of vegetation and parks."
    elif pct >= 20:
        return "🌱 Moderate green coverage. Some parks and green areas present."
    elif pct >= 10:
        return "🏙️ Limited green space. Mostly urban area with minimal vegetation."
    return "🏢 Very low green coverage. Highly urbanised with minimal natural spaces."


def _display_recent_green_analyses():
    try:
        resp = requests.get(
            f"{api.base_url}/api/analysis/green-space/recent?limit=5", timeout=5
        )
        if resp.status_code == 200:
            analyses = resp.json().get("analyses", [])
            if not analyses:
                st.info("No recent analyses found")
                return
            for a in analyses:
                _render_green_card(a)
        else:
            st.warning("Could not fetch recent analyses")
    except Exception as e:
        st.error(f"Error loading recent analyses: {e}")


def _render_green_card(analysis: dict):
    status    = analysis.get("status", "unknown")
    address   = analysis.get("address", "Unknown")
    green_pct = analysis.get("green_space_percentage")
    created   = analysis.get("created_at", "")
    icon      = {"completed": "✅", "failed": "❌", "processing": "⏳"}.get(status, "❓")
    with st.expander(f"{icon} {address} — {created[:10] if created else ''}"):
        if status == "completed" and green_pct is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Green Coverage", f"{green_pct:.1f}%")
            with col2:
                st.metric("Radius", f"{analysis.get('search_radius_m', 0)} m")
            if bd := analysis.get("breakdown", {}):
                for k, v in bd.items():
                    st.write(f"- {k.replace('_',' ').title()}: {v}%")
        else:
            st.write(f"Status: {status}")


# ══════════════════════════════════════════════
# TAB 3 – FIND SIMILAR NEIGHBORHOODS (standalone)
# ══════════════════════════════════════════════

def _render_find_similar_tab():
    """Standalone similarity search — no new analysis required."""
    st.subheader("🔍 Find Similar Neighborhoods")
    st.markdown(
        "Select any **previously completed** neighborhood analysis as your "
        "reference location, then discover areas with similar characteristics "
        "across the entire database — **no new analysis needed**."
    )

    # ── Load completed analyses ──────────────────
    recent_resp = api.get("/api/neighborhood/recent", params={"limit": 50})
    analyses    = (recent_resp or {}).get("analyses", [])
    completed   = [a for a in analyses if a.get("status") == "completed"]

    if not completed:
        st.warning(
            "⚠️ No completed analyses found yet. "
            "Run at least one Neighborhood Analysis first, then come back here."
        )
        st.info(
            "💡 Tip: Go to the **🗺️ Neighborhood Analysis** tab, enter an address "
            "and click *Start Analysis*. Once it finishes you can use it here."
        )
        return

    # ── Reference selector ───────────────────────
    st.markdown("### 1️⃣ Choose Your Reference Location")

    options = {
        f"{a.get('address', 'Unknown')} "
        f"(WS: {a.get('walk_score', 0):.0f} | "
        f"Amenities: {a.get('total_amenities', 0)})": a.get("analysis_id")
        for a in completed
    }

    selected_label = st.selectbox(
        "Reference analysis", list(options.keys()), key="fs_reference_select"
    )
    selected_analysis_id = options[selected_label]

    # Show summary card for selected reference
    ref_analysis = next((a for a in completed if a.get("analysis_id") == selected_analysis_id), None)
    if ref_analysis:
        with st.container(border=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🚶 Walk Score", f"{ref_analysis.get('walk_score', 0):.1f}")
            with col2:
                st.metric("📍 Amenities", ref_analysis.get("total_amenities", 0))
            with col3:
                st.metric("🗓️ Date", (ref_analysis.get("created_at") or "")[:10])

    st.markdown("### 2️⃣ Search Parameters")
    col1, col2 = st.columns(2)
    with col1:
        limit = st.slider("Max results", 1, 20, 5, key="fs_limit")
    with col2:
        threshold_pct = st.slider("Min similarity %", 0, 100, 60, key="fs_threshold")

    # ── Search button ────────────────────────────
    if st.button("🔍 Find Similar Neighborhoods", type="primary",
                 use_container_width=True, key="fs_run"):
        _run_find_similar(selected_analysis_id, limit, threshold_pct / 100)

    # ── Display stored results ───────────────────
    if "fs_results" in st.session_state and st.session_state.fs_results:
        st.divider()
        _render_similarity_results(st.session_state.fs_results)


def _run_find_similar(analysis_id: str, limit: int, threshold: float):
    """Call API and store results in session state."""
    with st.spinner("🔍 Searching for similar neighborhoods…"):
        try:
            resp = requests.get(
                f"{api.base_url}/api/neighborhood/{analysis_id}/similar",
                params={"limit": limit, "threshold": threshold},
                timeout=30,
            )
        except requests.exceptions.RequestException as e:
            show_error_message(f"Network error: {e}")
            st.session_state.fs_results = None
            return

    if resp.status_code != 200:
        show_error_message(f"Search failed: {resp.text}")
        st.session_state.fs_results = None
        return

    data   = resp.json()
    report = data.get("report", {})
    found  = report.get("similar_neighborhoods", [])

    if not found:
        st.info(
            f"ℹ️ No similar neighborhoods found at {threshold*100:.0f}% threshold. "
            "Try lowering the minimum similarity."
        )
        st.session_state.fs_results = None
        return

    show_success_message(f"Found {len(found)} similar neighborhood(s)!")
    st.session_state.fs_results = {"neighborhoods": found, "report": report,
                                    "query_analysis_id": analysis_id}


def _render_similarity_results(data: dict):
    neighborhoods = data["neighborhoods"]
    report        = data["report"]
    query_info    = report.get("query", {})

    st.subheader(f"📊 Results — {len(neighborhoods)} match(es)")

    # Reference summary
    with st.container(border=True):
        st.markdown("**📍 Reference Location**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Address:** {query_info.get('address', 'N/A')}")
        with col2:
            st.metric("Walk Score", f"{query_info.get('walk_score', 0):.1f}")
        with col3:
            st.metric("Amenities", query_info.get("total_amenities", 0))

    st.markdown("---")

    for idx, nb in enumerate(neighborhoods, 1):
        _render_similarity_card(nb, idx, query_info)


def _render_similarity_card(nb: dict, idx: int, query_info: dict):
    similarity = nb.get("similarity_score", 0) * 100
    address    = nb.get("address", "Unknown")

    with st.expander(
        f"#{idx} — {address}  |  {similarity:.1f}% match",
        expanded=(idx == 1),
    ):
        st.progress(nb.get("similarity_score", 0))

        col1, col2, col3, col4 = st.columns(4)
        walk_diff = nb.get("walk_score_diff", 0)
        with col1:
            st.metric("🚶 Walk Score", f"{nb.get('walk_score', 0):.1f}",
                      delta=f"{walk_diff:+.1f}" if walk_diff else None)
        with col2:
            amenity_diff = nb.get("total_amenities", 0) - query_info.get("total_amenities", 0)
            st.metric("📍 Amenities", nb.get("total_amenities", 0),
                      delta=f"{amenity_diff:+d}" if amenity_diff else None)
        with col3:
            st.metric("🎯 Match", f"{similarity:.1f}%")
        with col4:
            aid = nb.get("analysis_id")
            if aid and st.button("🗺️ View Map", key=f"fs_map_{idx}_{aid}"):
                st.session_state[f"fs_show_map_{idx}"] = True

        # Key differences
        diffs = nb.get("key_differences", [])
        if diffs:
            st.markdown("**🔍 Key Differences:**")
            for d in diffs:
                st.write(f"• {d}")

        # Amenity comparison toggle
        if st.checkbox("📊 Show Amenity Comparison", key=f"fs_cmp_{idx}"):
            _render_amenity_comparison(
                query_info.get("amenity_breakdown", {}),
                nb.get("amenity_breakdown", {}),
            )

        # Map
        if st.session_state.get(f"fs_show_map_{idx}"):
            aid = nb.get("analysis_id")
            if aid:
                map_url = f"{api.base_url}/api/neighborhood/{aid}/map"
                try:
                    html_resp = requests.get(map_url, timeout=10)
                    if html_resp.status_code == 200:
                        st.components.v1.html(html_resp.text, height=500, scrolling=True)
                    else:
                        st.warning("Map not available")
                except Exception as e:
                    st.warning(f"Could not load map: {e}")


def _render_amenity_comparison(query_a: dict, candidate_a: dict):
    import pandas as pd
    all_types = set(list(query_a) + list(candidate_a))
    rows = []
    for t in sorted(all_types):
        qc = query_a.get(t, 0)
        cc = candidate_a.get(t, 0)
        diff = cc - qc
        rows.append({
            "Amenity": t.replace("_", " ").title(),
            "Reference": qc,
            "Similar": cc,
            "Difference": f"{diff:+d}" if diff != 0 else "Same",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)