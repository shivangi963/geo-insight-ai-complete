
import streamlit as st
import requests
from PIL import Image
import io
import time
from typing import Optional, Dict
import plotly.graph_objects as go
from streamlit_javascript import st_javascript

# API Configuration
API_URL = st.secrets.get("API_URL", "http://localhost:8000")


def render_image_analysis_page():
    """Main page renderer"""
    st.title("🖼️ Image Analysis")
    
    # Create tabs
    tab1, tab2 = st.tabs(["🌳 Green Space Analysis", "🚗 Street Scene Analysis"])
    
    with tab1:
        render_green_space_tab()
    
    with tab2:
        render_street_scene_tab()


def render_green_space_tab():
    """Green Space Analysis using OpenStreetMap"""
    st.header("🌳 Green Space Analysis")
    st.markdown("""
    Analyze green coverage in any area using OpenStreetMap data.
    Detects parks, forests, recreation areas, and natural spaces.
    """)
    
    # Input Section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Get address from session state or use default
        default_address = st.session_state.get('selected_address', '')
        address = st.text_input(
            "📍 Enter Address",
            value=default_address,
            placeholder="e.g., Central Park, New York, NY",
            help="Enter any address to analyze green space coverage"
        )
    
    with col2:
        radius = st.slider(
            "Search Radius (meters)",
            min_value=100,
            max_value=2000,
            value=500,
            step=100,
            help="Area radius to analyze around the address"
        )
    
    # Analysis Button
    if st.button("🚀 Analyze Green Space", type="primary", use_container_width=True):
        if not address:
            st.error("Please enter an address")
            return
        
        # Run analysis
        with st.spinner("Analyzing green space... This may take 30-60 seconds"):
            result = run_green_space_analysis(address, radius)
        
        if result:
            st.success("✅ Analysis completed!")
            display_green_space_results(result)
        else:
            st.error("❌ Analysis failed. Please try again.")
    
    # Display Recent Analyses
    st.markdown("---")
    st.subheader("📋 Recent Analyses")
    display_recent_analyses()


def run_green_space_analysis(address: str, radius: int) -> Optional[Dict]:
    """
    Submit green space analysis and poll for results
    
    Args:
        address: Address to analyze
        radius: Search radius in meters
    
    Returns:
        Analysis results or None if failed
    """
    try:
        # Submit analysis request
        response = requests.post(
            f"{API_URL}/api/satellite/analyze",
            json={
                "address": address,
                "radius_m": radius,
                "calculate_green_space": True
            },
            timeout=10
        )
        
        if response.status_code != 202:
            st.error(f"API Error: {response.text}")
            return None
        
        data = response.json()
        analysis_id = data.get('analysis_id')
        
        if not analysis_id:
            st.error("No analysis ID returned")
            return None
        
        # Poll for results
        max_attempts = 60  # 60 seconds max
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for attempt in range(max_attempts):
            time.sleep(1)
            
            # Get analysis status
            result_response = requests.get(
                f"{API_URL}/api/satellite/{analysis_id}",
                timeout=5
            )
            
            if result_response.status_code == 200:
                result = result_response.json()
                status = result.get('status', 'pending')
                progress = result.get('progress', 0)
                message = result.get('message', 'Processing...')
                
                # Update progress
                progress_bar.progress(progress / 100)
                status_text.text(f"Status: {message}")
                
                if status == 'completed':
                    progress_bar.empty()
                    status_text.empty()
                    return result
                elif status == 'failed':
                    error = result.get('error', 'Unknown error')
                    st.error(f"Analysis failed: {error}")
                    return None
        
        st.error("Analysis timed out")
        return None
        
    except requests.exceptions.RequestException as e:
        st.error(f"Network error: {e}")
        return None
    except Exception as e:
        st.error(f"Error: {e}")
        return None


def display_green_space_results(result: Dict):
    """Display green space analysis results"""
    
    # Extract data
    green_pct = result.get('green_space_percentage', 0)
    green_pixels = result.get('green_pixels', 0)
    total_pixels = result.get('total_pixels', 0)
    breakdown = result.get('breakdown', {})
    visualization_path = result.get('visualization_path')
    address = result.get('address', 'Unknown')
    coordinates = result.get('coordinates', {})
    
    # Metrics Section
    st.markdown("### 📊 Analysis Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Green Coverage",
            value=f"{green_pct:.1f}%",
            delta=None
        )
    
    with col2:
        st.metric(
            label="Green Pixels",
            value=f"{green_pixels:,}"
        )
    
    with col3:
        st.metric(
            label="Total Pixels",
            value=f"{total_pixels:,}"
        )
    
    # Gauge Chart
    st.markdown("### 🎯 Green Coverage Gauge")
    fig = create_gauge_chart(green_pct)
    st.plotly_chart(fig, use_container_width=True)
    
    # Interpretation
    interpretation = get_green_space_interpretation(green_pct)
    st.info(f"**Interpretation:** {interpretation}")
    
    # Breakdown by Type
    if breakdown:
        st.markdown("### 🌲 Breakdown by Green Type")
        
        breakdown_cols = st.columns(4)
        labels = {
            'parks_grass': ('Parks/Grass', '🌱'),
            'forests_woods': ('Forests/Woods', '🌲'),
            'recreation': ('Recreation', '⚽'),
            'natural_areas': ('Natural Areas', '🌿')
        }
        
        for idx, (key, pct) in enumerate(breakdown.items()):
            label, icon = labels.get(key, (key.replace('_', ' ').title(), '🟢'))
            with breakdown_cols[idx % 4]:
                st.metric(
                    label=f"{icon} {label}",
                    value=f"{pct:.1f}%"
                )
        
        # Breakdown Chart
        if sum(breakdown.values()) > 0:
            fig_breakdown = create_breakdown_chart(breakdown)
            st.plotly_chart(fig_breakdown, use_container_width=True)
    
    # Visualization Image
    if visualization_path:
        st.markdown("### 🗺️ Visual Analysis")
        try:
            # Fetch visualization image
            viz_url = f"{API_URL}/{visualization_path}"
            viz_response = requests.get(viz_url, timeout=10)
            
            if viz_response.status_code == 200:
                image = Image.open(io.BytesIO(viz_response.content))
                st.image(
                    image,
                    caption="Green spaces highlighted: Light Green = Parks, Dark Green = Forests, Medium = Recreation, Olive = Natural",
                    use_container_width=True
                )
            else:
                st.warning("Visualization image not available")
        except Exception as e:
            st.warning(f"Could not load visualization: {e}")
    
    # Location Details
    with st.expander("📍 Location Details"):
        st.write(f"**Address:** {address}")
        if coordinates:
            st.write(f"**Latitude:** {coordinates.get('latitude', 'N/A')}")
            st.write(f"**Longitude:** {coordinates.get('longitude', 'N/A')}")
        st.write(f"**Search Radius:** {result.get('search_radius_m', 'N/A')} meters")
        st.write(f"**Map Source:** OpenStreetMap")


def create_gauge_chart(percentage: float) -> go.Figure:
    """Create a gauge chart for green coverage percentage"""
    
    # Determine color based on percentage
    if percentage >= 50:
        color = "#28a745"  # Green
    elif percentage >= 30:
        color = "#ffc107"  # Yellow
    else:
        color = "#dc3545"  # Red
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=percentage,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Green Coverage %", 'font': {'size': 24}},
        number={'suffix': "%", 'font': {'size': 40}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 20], 'color': '#ffe6e6'},
                {'range': [20, 40], 'color': '#fff4e6'},
                {'range': [40, 60], 'color': '#ffffcc'},
                {'range': [60, 80], 'color': '#e6ffe6'},
                {'range': [80, 100], 'color': '#ccffcc'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig


def create_breakdown_chart(breakdown: Dict[str, float]) -> go.Figure:
    """Create bar chart for green type breakdown"""
    
    labels_map = {
        'parks_grass': 'Parks/Grass',
        'forests_woods': 'Forests/Woods',
        'recreation': 'Recreation',
        'natural_areas': 'Natural Areas'
    }
    
    colors_map = {
        'parks_grass': '#90EE90',
        'forests_woods': '#228B22',
        'recreation': '#3CB371',
        'natural_areas': '#6B8E23'
    }
    
    labels = [labels_map.get(k, k) for k in breakdown.keys()]
    values = list(breakdown.values())
    colors = [colors_map.get(k, '#00FF00') for k in breakdown.keys()]
    
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=[f'{v:.1f}%' for v in values],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Green Space Breakdown",
        xaxis_title="Green Type",
        yaxis_title="Coverage (%)",
        height=400,
        showlegend=False
    )
    
    return fig


def get_green_space_interpretation(percentage: float) -> str:
    """Get interpretation text based on green space percentage"""
    if percentage >= 60:
        return "🌲 Excellent! This area has abundant green coverage with parks, forests, and natural spaces."
    elif percentage >= 40:
        return "🌳 Good green coverage. The area has a healthy amount of vegetation and parks."
    elif percentage >= 20:
        return "🌱 Moderate green coverage. Some parks and green areas present."
    elif percentage >= 10:
        return "🏙️ Limited green space. Mostly urban area with minimal vegetation."
    else:
        return "🏢 Very low green coverage. Highly urbanized area with minimal natural spaces."


def display_recent_analyses():
    """Display list of recent analyses"""
    try:
        response = requests.get(
            f"{API_URL}/api/satellite/recent?limit=5",
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            analyses = data.get('analyses', [])
            
            if not analyses:
                st.info("No recent analyses found")
                return
            
            for analysis in analyses:
                status = analysis.get('status', 'unknown')
                address = analysis.get('address', 'Unknown')
                green_pct = analysis.get('green_space_percentage')
                created = analysis.get('created_at', '')
                
                # Status icon
                status_icon = {
                    'completed': '✅',
                    'failed': '❌',
                    'processing': '⏳',
                    'pending': '🔄'
                }.get(status, '❓')
                
                with st.expander(f"{status_icon} {address} - {created[:10] if created else ''}"):
                    if status == 'completed' and green_pct is not None:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Green Coverage", f"{green_pct:.1f}%")
                        with col2:
                            st.metric("Radius", f"{analysis.get('search_radius_m', 0)}m")
                        
                        # Show breakdown if available
                        breakdown = analysis.get('breakdown', {})
                        if breakdown:
                            st.write("**Breakdown:**")
                            for key, value in breakdown.items():
                                st.write(f"- {key.replace('_', ' ').title()}: {value}%")
                    else:
                        st.write(f"Status: {status}")
        else:
            st.warning("Could not fetch recent analyses")
            
    except Exception as e:
        st.error(f"Error loading recent analyses: {e}")


def render_street_scene_tab():
    """Street Scene Object Detection"""
    st.header("🚗 Street Scene Analysis")
    st.markdown("""
    Upload a street scene image to detect vehicles and pedestrians.
    Useful for traffic analysis and urban planning.
    """)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload Street Scene Image",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a photo of a street, road, or public space"
    )
    
    if uploaded_file:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Analyze button
        if st.button("🔍 Detect Objects", type="primary", use_container_width=True):
            with st.spinner("Analyzing image..."):
                result = run_object_detection(uploaded_file)
            
            if result:
                display_object_detection_results(result)


def run_object_detection(uploaded_file) -> Optional[Dict]:
    """
    Run object detection on uploaded image
    
    Args:
        uploaded_file: Streamlit uploaded file object
    
    Returns:
        Detection results or None
    """
    try:
        # Prepare file for upload
        files = {
            'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
        }
        
        # Submit to API
        response = requests.post(
            f"{API_URL}/api/image-analysis/detect",
            files=files,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        st.error(f"Network error: {e}")
        return None
    except Exception as e:
        st.error(f"Error: {e}")
        return None


def display_object_detection_results(result: Dict):
    """Display object detection results"""
    
    detections = result.get('detections', [])
    annotated_image_url = result.get('annotated_image_url')
    
    st.success(f"✅ Detected {len(detections)} objects")
    
    # Count by class
    class_counts = {}
    for det in detections:
        class_name = det.get('class', 'Unknown')
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    # Display counts
    st.markdown("### 📊 Detection Summary")
    cols = st.columns(len(class_counts))
    
    icons = {
        'car': '🚗',
        'truck': '🚛',
        'bus': '🚌',
        'motorcycle': '🏍️',
        'bicycle': '🚴',
        'person': '🚶'
    }
    
    for idx, (class_name, count) in enumerate(class_counts.items()):
        icon = icons.get(class_name, '📦')
        with cols[idx]:
            st.metric(
                label=f"{icon} {class_name.title()}",
                value=count
            )
    
    # Display annotated image
    if annotated_image_url:
        st.markdown("### 🖼️ Annotated Image")
        try:
            img_response = requests.get(f"{API_URL}/{annotated_image_url}", timeout=10)
            if img_response.status_code == 200:
                annotated_image = Image.open(io.BytesIO(img_response.content))
                st.image(annotated_image, caption="Detected Objects", use_container_width=True)
        except Exception as e:
            st.warning(f"Could not load annotated image: {e}")
    
    # Detailed detections
    with st.expander("📋 Detailed Detections"):
        for idx, det in enumerate(detections, 1):
            st.write(f"**{idx}. {det.get('class', 'Unknown').title()}**")
            st.write(f"   Confidence: {det.get('confidence', 0):.2%}")
            bbox = det.get('bbox', [])
            if bbox:
                st.write(f"   Location: ({bbox[0]:.0f}, {bbox[1]:.0f}) - ({bbox[2]:.0f}, {bbox[3]:.0f})")


# Main entry point
if __name__ == "__main__":
    render_image_analysis_page()