"""
Image Analysis Page
Two tools:
  1. Green Space Analysis  – identical functionality also available in
                             the Neighborhood tab (no prior analysis needed)
  2. Street Scene Detection – YOLO object detection on uploaded images
"""
import streamlit as st
import requests
from PIL import Image
import io
import time
from typing import Optional, Dict
import plotly.graph_objects as go

from api_client import api
from utils import (
    format_percentage, show_success_message, show_error_message,
    poll_task_status, validate_file_size
)
from components.header import render_section_header
from config import TASK_MAX_WAIT


def render_image_analysis_page():
    render_section_header("Image Analysis", "🖼️")

    st.markdown(
        "Computer-vision tools for location analysis and urban object detection."
    )

    tab1, tab2 = st.tabs(["🌳 Green Space Analysis", "🚗 Street Scene Detection"])

    with tab1:
        render_green_space_tab()

    with tab2:
        render_street_scene_tab()


# ══════════════════════════════════════════════
# TAB 1 – GREEN SPACE ANALYSIS
# ══════════════════════════════════════════════

def render_green_space_tab():
    st.subheader("🌳 Green Space Coverage Analysis")

    st.info(
        "💡 **Tip:** This tool is also available directly in the "
        "**🗺️ Neighborhood** tab under *Green Space Analysis* — "
        "no prior analysis required either way."
    )

    st.markdown(
        "Enter any address to calculate green coverage from OpenStreetMap tile data."
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        default_address = st.session_state.get("selected_address", "")
        address = st.text_input(
            "📍 Enter Address",
            value=default_address,
            placeholder="e.g., Central Park, New York, NY",
            key="green_space_address",
        )
    with col2:
        radius = st.slider(
            "Search Radius (m)", 100, 4000, 500, 100, key="green_space_radius"
        )

    if st.button("🚀 Analyse Green Space", type="primary",
                 use_container_width=True, key="gs_image_tab_run"):
        if not address:
            show_error_message("Please enter an address")
            return
        result = run_green_space_analysis(address, radius)
        if result:
            display_green_space_results(result)

    st.divider()
    st.subheader("📋 Recent Green Space Analyses")
    display_recent_green_analyses()


def run_green_space_analysis(address: str, radius: int) -> Optional[Dict]:
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

    data        = response.json()
    analysis_id = data.get("analysis_id")
    task_id     = data.get("task_id")
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
        full = requests.get(
            f"{api.base_url}/api/analysis/green-space/{analysis_id}", timeout=10
        )
        if full.status_code == 200:
            return full.json()
    return None


def display_green_space_results(result: Dict):
    st.divider()
    st.subheader("✅ Analysis Complete")

    green_pct    = result.get("green_space_percentage", 0)
    green_pixels = result.get("green_pixels", 0)
    total_pixels = result.get("total_pixels", 0)
    breakdown    = result.get("breakdown", {})
    viz_path     = result.get("visualization_path")
    address      = result.get("address", "Unknown")
    coordinates  = result.get("coordinates", {})

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🌳 Green Coverage", f"{green_pct:.1f}%")
    with col2:
        st.metric("🟢 Green Pixels", f"{green_pixels:,}")
    with col3:
        st.metric("📐 Total Pixels", f"{total_pixels:,}")

    st.markdown("### 🎯 Green Coverage Gauge")
    st.plotly_chart(create_gauge_chart(green_pct), use_container_width=True)
    st.info(f"**Interpretation:** {get_green_space_interpretation(green_pct)}")

    if breakdown and sum(breakdown.values()) > 0:
        st.markdown("### 🌲 Breakdown by Type")
        labels_map = {
            "parks_grass":   ("Parks/Grass",   "🌱"),
            "forests_woods": ("Forests/Woods", "🌲"),
            "recreation":    ("Recreation",    "⚽"),
            "natural_areas": ("Natural Areas", "🌿"),
        }
        bcols = st.columns(4)
        for idx, (key, pct) in enumerate(breakdown.items()):
            label, icon = labels_map.get(key, (key.replace("_", " ").title(), "🟢"))
            with bcols[idx % 4]:
                st.metric(f"{icon} {label}", f"{pct:.1f}%")
        st.plotly_chart(create_breakdown_chart(breakdown), use_container_width=True)

    if viz_path:
        st.markdown("### 🗺️ Visual Analysis")
        try:
            viz_resp = requests.get(f"{api.base_url}/{viz_path}", timeout=10)
            if viz_resp.status_code == 200:
                img = Image.open(io.BytesIO(viz_resp.content))
                st.image(img, caption="Green spaces highlighted by type", width=400)
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


def create_gauge_chart(percentage: float) -> go.Figure:
    color = "#28a745" if percentage >= 50 else "#ffc107" if percentage >= 30 else "#dc3545"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=percentage,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Green Coverage %", "font": {"size": 24}},
        number={"suffix": "%", "font": {"size": 40}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "darkblue"},
            "bar": {"color": color},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "gray",
            "steps": [
                {"range": [0, 20],  "color": "#ffe6e6"},
                {"range": [20, 40], "color": "#fff4e6"},
                {"range": [40, 60], "color": "#ffffcc"},
                {"range": [60, 80], "color": "#e6ffe6"},
                {"range": [80, 100],"color": "#ccffcc"},
            ],
            "threshold": {"line": {"color": "red", "width": 4},
                          "thickness": 0.75, "value": 50},
        },
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def create_breakdown_chart(breakdown: Dict[str, float]) -> go.Figure:
    labels_map = {"parks_grass": "Parks/Grass", "forests_woods": "Forests/Woods",
                  "recreation": "Recreation", "natural_areas": "Natural Areas"}
    colors_map = {"parks_grass": "#90EE90", "forests_woods": "#228B22",
                  "recreation": "#3CB371", "natural_areas": "#6B8E23"}
    labels = [labels_map.get(k, k) for k in breakdown]
    values = list(breakdown.values())
    colors = [colors_map.get(k, "#00FF00") for k in breakdown]
    fig = go.Figure(data=[go.Bar(
        x=labels, y=values, marker_color=colors,
        text=[f"{v:.1f}%" for v in values], textposition="auto",
    )])
    fig.update_layout(title="Green Space Breakdown", xaxis_title="Green Type",
                      yaxis_title="Coverage (%)", height=400, showlegend=False)
    return fig


def get_green_space_interpretation(percentage: float) -> str:
    if percentage >= 60:
        return "🌲 Excellent! This area has abundant green coverage with parks, forests, and natural spaces."
    elif percentage >= 40:
        return "🌳 Good green coverage. The area has a healthy amount of vegetation and parks."
    elif percentage >= 20:
        return "🌱 Moderate green coverage. Some parks and green areas present."
    elif percentage >= 10:
        return "🏙️ Limited green space. Mostly urban area with minimal vegetation."
    return "🏢 Very low green coverage. Highly urbanised area with minimal natural spaces."


def display_recent_green_analyses():
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
                render_analysis_card(a)
        else:
            st.warning("Could not fetch recent analyses")
    except Exception as e:
        st.error(f"Error loading recent analyses: {e}")


def render_analysis_card(analysis: dict):
    status    = analysis.get("status", "unknown")
    address   = analysis.get("address", "Unknown")
    green_pct = analysis.get("green_space_percentage")
    created   = analysis.get("created_at", "")
    icon      = {"completed": "✅", "failed": "❌", "processing": "⏳",
                 "pending": "🔄"}.get(status, "❓")
    with st.expander(f"{icon} {address} — {created[:10] if created else ''}"):
        if status == "completed" and green_pct is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Green Coverage", f"{green_pct:.1f}%")
            with col2:
                st.metric("Radius", f"{analysis.get('search_radius_m', 0)} m")
            if bd := analysis.get("breakdown", {}):
                st.write("**Breakdown:**")
                for key, value in bd.items():
                    st.write(f"- {key.replace('_', ' ').title()}: {value}%")
        else:
            st.write(f"Status: {status}")


# ══════════════════════════════════════════════
# TAB 2 – STREET SCENE DETECTION
# ══════════════════════════════════════════════

def render_street_scene_tab():
    st.subheader("🚗 Street Scene Detection")
    st.info(
        "**Upload a street scene image to detect vehicles and pedestrians.**\n\n"
        "Detects: 🚗 Cars · 🚛 Trucks & Buses · 🏍️ Motorcycles · "
        "🚴 Bicycles · 🚶 Pedestrians"
    )

    uploaded_file = st.file_uploader(
        "📤 Upload Street Scene Image",
        type=["jpg", "jpeg", "png"],
        help="Upload a photo of a street, road, or public space",
        key="street_scene_upload",
    )

    if uploaded_file:
        if not validate_file_size(uploaded_file, 10):
            return

        col1, col2 = st.columns([2, 1])
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
        with col2:
            st.markdown("### 📋 File Info")
            st.info(f"**Name:** {uploaded_file.name}")
            st.info(f"**Size:** {uploaded_file.size / 1024:.1f} KB")

        if st.button("🔍 Detect Objects", type="primary", use_container_width=True):
            result = run_street_detection(uploaded_file)
            if result:
                display_street_detection_results(result)


def run_street_detection(uploaded_file) -> Optional[Dict]:
    with st.spinner("🔍 Analysing street scene…"):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        try:
            response = requests.post(
                f"{api.base_url}/api/analysis/street-scene",
                files=files,
                timeout=30,
            )
        except requests.exceptions.RequestException as e:
            show_error_message(f"Network error: {e}")
            return None

    if response.status_code == 200:
        show_success_message("Detection complete!")
        return response.json()
    else:
        show_error_message(f"Detection failed: {response.text}")
        return None


def display_street_detection_results(result: Dict):
    st.divider()
    st.subheader("✅ Detection Complete")

    detections   = result.get("detections", [])
    class_counts = result.get("class_counts", {})
    total_objects = result.get("total_objects", 0)

    st.success(f"🎯 Detected **{total_objects}** objects")

    if class_counts:
        st.markdown("### 📊 Detection Summary")
        n   = len(class_counts)
        cols = st.columns(min(n, 4))
        icons = {"car": "🚗", "truck": "🚛", "bus": "🚌",
                 "motorcycle": "🏍️", "bicycle": "🚴", "person": "🚶"}
        for idx, (cls, cnt) in enumerate(class_counts.items()):
            icon = icons.get(cls, "📦")
            with cols[idx % min(n, 4)]:
                st.metric(f"{icon} {cls.title()}", cnt)

    if detections:
        with st.expander("📋 Detailed Detections", expanded=False):
            for idx, det in enumerate(detections, 1):
                st.write(f"**{idx}. {det.get('class', 'Unknown').title()}**")
                st.write(f"   Confidence: {det.get('confidence', 0):.2%}")
                bbox = det.get("bbox", [])
                if bbox:
                    st.write(
                        f"   Location: ({bbox[0]:.0f}, {bbox[1]:.0f}) "
                        f"– ({bbox[2]:.0f}, {bbox[3]:.0f})"
                    )
                st.divider()


if __name__ == "__main__":
    render_image_analysis_page()