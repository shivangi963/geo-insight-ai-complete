"""
Image Analysis Page
Computer vision analysis for property images
"""
import requests
import streamlit as st
from api_client import api
from utils import (
    poll_task_status, validate_file_size, 
    show_success_message, show_error_message,
    format_number
)
from components.header import render_section_header
from config import feature_config, API_URL
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def upload_and_analyze_image(api_url, uploaded_file, analysis_type="object_detection"):
    """
    Properly upload image file to backend API
    
    Args:
        api_url: Base API URL
        uploaded_file: Streamlit UploadedFile object
        analysis_type: Type of analysis to perform
    
    Returns:
        API response dictionary
    """
    import requests
    
    if uploaded_file is None:
        st.error("Please upload an image first")
        return None
    
    try:
        # Reset file pointer to beginning
        uploaded_file.seek(0)
        
        # Create proper multipart/form-data payload
        files = {
            'file': (
                uploaded_file.name,           # filename
                uploaded_file,                # file object
                uploaded_file.type            # content type
            )
        }
        
        # Add parameters
        params = {
            'analysis_type': analysis_type
        }
        
        # Make request
        response = requests.post(
            f"{api_url}/api/analysis/image",
            files=files,
            params=params,
            timeout=30
        )
        
        # Handle response
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 400:
            error_detail = response.json().get('detail', 'Bad request')
            st.error(f"❌ Upload error: {error_detail}")
            return None
        elif response.status_code == 500:
            error_detail = response.json().get('detail', 'Server error')
            st.error(f"❌ Server error: {error_detail}")
            return None
        else:
            st.error(f"❌ Unexpected error: HTTP {response.status_code}")
            return None
    
    except requests.exceptions.Timeout:
        st.error("⏰ Request timed out. Please try again.")
        return None
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None
    

def render_image_analysis_page():
    """Main image analysis page"""
    render_section_header("Computer Vision Analysis", "📸")
    
    st.markdown("""
    Upload images for AI-powered analysis:
    - **🚗 Street Scene:** Detect vehicles, pedestrians, traffic
    - **🌳 Green Space:** Calculate vegetation coverage
    """)
    
    # Analysis type selector
    analysis_type = st.radio(
        "🔬 Analysis Type",
        ["object_detection", "green_space"],
        format_func=lambda x: "🚗 Street Scene (Object Detection)" if x == "object_detection" else "🌳 Green Space Calculator",
        horizontal=True,
        key="img_analysis_type"
    )
    
    # File uploader
    render_file_uploader(analysis_type)

def render_file_uploader(analysis_type: str):
    """Render file upload section"""
    # In frontend/app.py - Image Analysis section
    uploaded_file = st.file_uploader("Upload street image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None and st.button("Analyze Image"):
        # Make sure to send it correctly
        files = {
            'file': (uploaded_file.name, uploaded_file, uploaded_file.type)
        } 
        response = requests.post(
            f"{API_URL}/api/analysis/image",
            files=files,
            params={"analysis_type": "object_detection"}
        )
    
    # Display uploaded image
    render_image_preview(uploaded_file)
    
    # Analyze button
    if st.button("🚀 Analyze Image", type="primary", use_container_width=True, key="analyze_img"):
        handle_image_analysis(uploaded_file, analysis_type)

def render_upload_help():
    """Show help when no image uploaded"""
    st.info("👆 Upload an image to get started")
    
    with st.expander("💡 Tips for Best Results"):
        st.markdown("""
        **🚗 Street Scene Analysis:**
        - Use clear, well-lit street photos
        - Include roads, sidewalks, vehicles
        - Avoid overly zoomed photos
        
        **🌳 Green Space Analysis:**
        - Use satellite/aerial imagery
        - Ensure good contrast between green areas and buildings
        - Higher resolution = better accuracy
        
        **📸 General Tips:**
        - JPEG or PNG format
        - Max 10MB file size
        - Higher resolution preferred
        - Good lighting
        """)

def render_image_preview(uploaded_file):
    """Display preview of uploaded image"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        st.markdown("### 📋 File Info")
        st.info(f"**Name:** {uploaded_file.name}")
        st.info(f"**Size:** {uploaded_file.size / 1024:.1f} KB")
        st.info(f"**Type:** {uploaded_file.type}")
        
        # Validate size
        if not validate_file_size(uploaded_file, feature_config.max_file_size_mb):
            st.error(f"File too large! Max: {feature_config.max_file_size_mb}MB")

def handle_image_analysis(uploaded_file, analysis_type: str):
    """Handle image upload and analysis - FIXED VERSION"""
    from api_client import api
    
    # Validate file size
    if not validate_file_size(uploaded_file, 10):
        return
    
    st.divider()
    
    # Use the new upload function
    with st.spinner("📤 Uploading and analyzing image..."):
        result = upload_and_analyze_image(
            api_url=api.base_url,
            uploaded_file=uploaded_file,
            analysis_type=analysis_type
        )
    
    if not result:
        return
    
    show_success_message("Analysis complete!")
    
    # Display results based on type
    if analysis_type == "object_detection":
        render_object_detection_results(result)
    else:
        render_green_space_results(result)
 

def render_object_detection_results(result: dict):
    """Display object detection results"""
    st.subheader("📊 Street Scene Analysis")
    
    detections = result.get('detections', [])
    class_counts = result.get('class_counts', {})
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🎯 Objects Detected", len(detections))
    
    with col2:
        st.metric("🏷️ Object Types", len(class_counts))
    
    with col3:
        cars = class_counts.get('car', 0)
        st.metric("🚗 Vehicles", cars)
    
    # Detailed breakdown
    if class_counts:
        render_object_detection_chart(class_counts)
        render_object_detection_table(detections)
    
    # Annotated image
    annotated_path = result.get('annotated_image_path')
    if annotated_path:
        render_annotated_image(annotated_path)

def render_object_detection_chart(class_counts: dict):
    """Render object detection bar chart"""
    st.divider()
    st.markdown("### 📊 Object Distribution")
    
    df = pd.DataFrame(
        list(class_counts.items()),
        columns=['Object', 'Count']
    ).sort_values('Count', ascending=False)
    
    fig = px.bar(
        df, 
        x='Object', 
        y='Count',
        title="Detected Objects by Type",
        color='Count',
        color_continuous_scale='blues'
    )
    
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

def render_object_detection_table(detections: list):
    """Render detailed detection table"""
    if not detections:
        return
    
    with st.expander(f"📋 View All {len(detections)} Detections"):
        df = pd.DataFrame(detections)
        
        # Format confidence as percentage
        if 'confidence' in df.columns:
            df['confidence'] = df['confidence'].apply(lambda x: f"{x*100:.1f}%")
        
        st.dataframe(df, use_container_width=True)

def render_annotated_image(image_path: str):
    """Display annotated image"""
    st.divider()
    st.subheader("🖼️ Annotated Image")
    
    try:
        st.image(image_path, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not display image: {e}")
        st.info(f"Image saved at: {image_path}")

def render_green_space_results(result: dict):
    """Display green space analysis results"""
    st.subheader("🌳 Green Space Analysis")
    
    green_pct = result.get('green_space_percentage', 0)
    total_px = result.get('total_pixels', 0)
    green_px = result.get('green_pixels', 0)
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🌳 Green Coverage", f"{green_pct:.2f}%")
    
    with col2:
        st.metric("Green Pixels", format_number(green_px))
    
    with col3:
        st.metric("Total Pixels", format_number(total_px))
    
    # Gauge chart
    render_green_space_gauge(green_pct)
    
    # Interpretation
    render_green_space_interpretation(green_pct)
    
    # Visualization
    viz_path = result.get('visualization_path')
    if viz_path:
        st.divider()
        st.subheader("🗺️ Visualization")
        try:
            st.image(viz_path, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not display: {e}")
            st.info(f"Saved at: {viz_path}")

def render_green_space_gauge(green_pct: float):
    """Render gauge chart for green coverage"""
    st.divider()
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=green_pct,
        title={'text': "Green Coverage"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkgreen"},
            'steps': [
                {'range': [0, 20], 'color': "lightgray"},
                {'range': [20, 50], 'color': "yellow"},
                {'range': [50, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

def render_green_space_interpretation(green_pct: float):
    """Display interpretation of green coverage"""
    st.divider()
    st.markdown("### 📊 Interpretation")
    
    if green_pct > 50:
        st.success("🌟 **Excellent green coverage!** This area has abundant vegetation.")
        recommendation = "Great for outdoor activities, property values likely higher."
    elif green_pct > 25:
        st.info("✅ **Good green coverage.** Reasonable amount of vegetation.")
        recommendation = "Decent balance of green space and development."
    elif green_pct > 10:
        st.warning("⚠️ **Moderate green coverage.** Limited vegetation.")
        recommendation = "Consider proximity to parks and green areas."
    else:
        st.error("❌ **Low green coverage.** Very limited vegetation.")
        recommendation = "Highly urbanized area with minimal green space."
    
    st.caption(f"💡 {recommendation}")
    
    # Additional insights
    with st.expander("📚 Understanding Green Coverage"):
        st.markdown("""
        **What is Green Coverage?**
        - Percentage of area covered by vegetation
        - Includes parks, trees, gardens, grass
        
        **Why it Matters:**
        - 🌡️ Temperature regulation
        - 💨 Air quality improvement
        - 🧘 Mental health benefits
        - 🏠 Property value impact
        
        **Ideal Ranges:**
        - **50%+** Excellent (parks, suburbs)
        - **25-50%** Good (residential areas)
        - **10-25%** Moderate (mixed development)
        - **<10%** Low (urban cores)
        """)