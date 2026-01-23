"""
FIXED GeoInsight AI Frontend
- Proper task polling
- Better error handling  
- Real-time status updates
- Simplified but powerful
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from datetime import datetime
import time

# ==================== CONFIG ====================

st.set_page_config(
    page_title="GeoInsight AI",
    page_icon="🏠",
    layout="wide"
)

API_URL = "http://localhost:8000"

# ==================== STYLES ====================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ==================== API CLIENT ====================

class APIClient:
    """Centralized API client with proper error handling"""
    
    @staticmethod
    def get(endpoint, params=None):
        try:
            response = requests.get(f"{API_URL}{endpoint}", params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            st.error("🔌 Cannot connect to backend. Is it running on port 8000?")
            return None
        except requests.exceptions.Timeout:
            st.error("⏱️ Request timeout")
            return None
        except Exception as e:
            st.error(f"❌ Error: {e}")
            return None
    
    @staticmethod
    def post(endpoint, data):
        try:
            response = requests.post(f"{API_URL}{endpoint}", json=data, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            st.error("🔌 Cannot connect to backend")
            return None
        except requests.exceptions.HTTPError as e:
            st.error(f"❌ Server error: {e.response.status_code}")
            if e.response.status_code == 400:
                st.error(f"Details: {e.response.json().get('detail', 'Bad request')}")
            return None
        except Exception as e:
            st.error(f"❌ Error: {e}")
            return None

api = APIClient()

# ==================== TASK POLLING ====================

def poll_task(task_id, max_wait=120):
    """Poll task status until completion"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        # Get task status
        data = api.get(f"/api/tasks/{task_id}")
        
        if not data:
            progress_bar.empty()
            status_text.empty()
            return None
        
        status = data.get('status')
        progress = data.get('progress', 0)
        message = data.get('message', '')
        
        # Update UI
        progress_bar.progress(progress / 100)
        
        if status == 'pending':
            status_text.info(f"⏳ Queued... {message}")
        elif status == 'processing':
            status_text.info(f"⚙️ {message}")
        elif status == 'completed':
            progress_bar.progress(1.0)
            status_text.success("✅ Complete!")
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()
            return data.get('result', {})
        elif status == 'failed':
            progress_bar.empty()
            status_text.error(f"❌ Failed: {data.get('error', 'Unknown')}")
            return None
        
        time.sleep(2)  # Poll every 2 seconds
    
    # Timeout
    progress_bar.empty()
    status_text.warning(f"⏱️ Timeout. Task ID: {task_id}")
    return None

# ==================== SIDEBAR ====================

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/city.png", width=80)
    st.title("⚙️ Settings")
    
    # Connection test
    if st.button("🔍 Test Connection", use_container_width=True):
        health = api.get("/health")
        if health:
            st.success(f"✅ Connected v{health.get('version')}")
        else:
            st.error("❌ Backend offline")
    
    st.divider()
    
    # Stats
    st.subheader("📊 Stats")
    stats = api.get("/api/stats")
    if stats:
        st.metric("Properties", stats.get('total_properties', 0))
        st.metric("Analyses", stats.get('total_analyses', 0))
        
        if stats.get('celery_enabled'):
            st.success("✅ Async enabled")
        else:
            st.warning("⚠️ Sync mode")

# ==================== HEADER ====================

st.markdown('<h1 class="main-header">🏠 GeoInsight AI</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666;">AI-Powered Real Estate Intelligence</p>', unsafe_allow_html=True)

# ==================== TABS ====================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏘️ Properties",
    "🗺️ Analysis",
    "🤖 AI Assistant",
    "📊 Dashboard",
    "📸 Image Analysis"
])

# ==================== TAB 1: PROPERTIES ====================

with tab1:
    st.header("🏘️ Property Management")
    
    # Sub-tabs
    prop_tab1, prop_tab2 = st.tabs(["Browse", "Add New"])
    
    with prop_tab1:
        # Fetch properties
        properties = api.get("/api/properties")
        
        if properties:
            df = pd.DataFrame(properties)
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total", len(df))
            with col2:
                if 'price' in df.columns:
                    st.metric("Avg Price", f"${df['price'].mean():,.0f}")
            with col3:
                if 'square_feet' in df.columns:
                    st.metric("Avg Size", f"{df['square_feet'].mean():,.0f} sqft")
            with col4:
                if 'city' in df.columns:
                    st.metric("Cities", df['city'].nunique())
            
            st.divider()
            
            # Filters
            col1, col2 = st.columns(2)
            with col1:
                if 'city' in df.columns:
                    cities = ['All'] + sorted(df['city'].dropna().unique().tolist())
                    city_filter = st.selectbox("Filter by City", cities)
            
            with col2:
                if 'price' in df.columns and len(df) > 0:
                    min_p = int(df['price'].min())
                    max_p = int(df['price'].max())
                    if min_p < max_p:
                        price_range = st.slider("Price Range", min_p, max_p, (min_p, max_p))
                        df = df[(df['price'] >= price_range[0]) & (df['price'] <= price_range[1])]
            
            if city_filter != 'All':
                df = df[df['city'] == city_filter]
            
            st.success(f"Showing {len(df)} properties")
            
            # Display
            for idx, row in df.iterrows():
                with st.expander(f"🏠 {row.get('address', 'N/A')} - ${row.get('price', 0):,.0f}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**City:** {row.get('city', 'N/A')}, {row.get('state', 'N/A')}")
                        st.write(f"**Type:** {row.get('property_type', 'N/A')}")
                    with col2:
                        st.write(f"**Beds:** {row.get('bedrooms', 'N/A')} | **Baths:** {row.get('bathrooms', 'N/A')}")
                        st.write(f"**Size:** {row.get('square_feet', 'N/A'):,} sqft")
        else:
            st.warning("No properties loaded")
    
    with prop_tab2:
        st.subheader("Add New Property")
        
        with st.form("add_property"):
            col1, col2 = st.columns(2)
            
            with col1:
                address = st.text_input("Address *")
                city = st.text_input("City *")
                state = st.text_input("State *")
                zip_code = st.text_input("ZIP *")
            
            with col2:
                price = st.number_input("Price ($) *", min_value=0, value=300000, step=10000)
                bedrooms = st.number_input("Bedrooms *", min_value=0, value=3, step=1)
                bathrooms = st.number_input("Bathrooms *", min_value=0.0, value=2.0, step=0.5)
                square_feet = st.number_input("Square Feet *", min_value=0, value=1500, step=100)
            
            property_type = st.selectbox("Type *", ["Single Family", "Condo", "Apartment", "Townhouse"])
            
            submitted = st.form_submit_button("➕ Add Property", type="primary", use_container_width=True)
            
            if submitted:
                if all([address, city, state, zip_code]):
                    data = {
                        "address": address,
                        "city": city,
                        "state": state,
                        "zip_code": zip_code,
                        "price": price,
                        "bedrooms": int(bedrooms),
                        "bathrooms": float(bathrooms),
                        "square_feet": int(square_feet),
                        "property_type": property_type,
                        "latitude": 0.0,
                        "longitude": 0.0
                    }
                    
                    result = api.post("/api/properties", data)
                    if result:
                        st.success(f"✅ Property added! ID: {result.get('id')}")
                        time.sleep(2)
                        st.rerun()
                else:
                    st.error("❌ Fill all required fields")

# ==================== TAB 2: ANALYSIS ====================

with tab2:
    st.header("🗺️ Neighborhood Analysis")
    
    st.markdown("Analyze any location for amenities, walkability, and neighborhood insights.")
    
    # Input
    address = st.text_input(
        "📍 Enter Address",
        placeholder="e.g., Manipal, Karnataka, India",
        help="Full address gives best results"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        radius = st.slider("🔍 Search Radius (m)", 500, 3000, 1000, 100)
    
    with col2:
        generate_map = st.checkbox("🗺️ Generate Map", value=True)
    
    # Amenities
    st.subheader("🎯 Select Amenities")
    
    amenity_cols = st.columns(4)
    amenities_selected = []
    
    amenity_options = {
        "🍽️ Restaurants": "restaurant",
        "☕ Cafes": "cafe",
        "🏫 Schools": "school",
        "🏥 Hospitals": "hospital",
        "🌳 Parks": "park",
        "🛒 Markets": "supermarket",
        "🏦 Banks": "bank",
        "💊 Pharmacy": "pharmacy"
    }
    
    for idx, (label, value) in enumerate(amenity_options.items()):
        with amenity_cols[idx % 4]:
            if st.checkbox(label, value=(value in ['restaurant', 'cafe', 'school'])):
                amenities_selected.append(value)
    
    # Submit
    if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
        if not address:
            st.error("❌ Enter an address")
        elif not amenities_selected:
            st.error("❌ Select at least one amenity")
        else:
            st.divider()
            
            with st.spinner("🔄 Creating analysis..."):
                response = api.post("/api/neighborhood/analyze", {
                    "address": address,
                    "radius_m": radius,
                    "amenity_types": amenities_selected,
                    "include_buildings": False,
                    "generate_map": generate_map
                })
                
                if response:
                    analysis_id = response.get('analysis_id')
                    task_id = response.get('task_id')
                    
                    st.success(f"✅ Analysis queued! ID: {analysis_id}")
                    st.info(f"Task: {task_id}")
                    
                    # Poll for completion
                    st.subheader("⚙️ Processing")
                    
                    result = poll_task(task_id)
                    
                    if result:
                        # Fetch full results
                        with st.spinner("📥 Loading results..."):
                            full_data = api.get(f"/api/neighborhood/{analysis_id}")
                            
                            if full_data:
                                st.divider()
                                st.subheader("📊 Results")
                                
                                # Metrics
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    walk_score = full_data.get('walk_score', 0)
                                    st.metric("🚶 Walk Score", f"{walk_score:.1f}/100")
                                
                                with col2:
                                    amenities = full_data.get('amenities', {})
                                    total = sum(len(items) for items in amenities.values())
                                    st.metric("📍 Amenities", total)
                                
                                with col3:
                                    st.metric("📊 Status", full_data.get('status', 'N/A'))
                                
                                # Amenities breakdown
                                st.divider()
                                st.subheader("🎯 Amenities Found")
                                
                                if amenities:
                                    amenity_counts = {k: len(v) for k, v in amenities.items() if v}
                                    
                                    if amenity_counts:
                                        fig = px.bar(
                                            x=list(amenity_counts.keys()),
                                            y=list(amenity_counts.values()),
                                            labels={'x': 'Type', 'y': 'Count'},
                                            title="Amenity Distribution"
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Detailed list
                                    cols = st.columns(3)
                                    for idx, (atype, items) in enumerate(amenities.items()):
                                        if items:
                                            with cols[idx % 3]:
                                                with st.expander(f"{atype.title()} ({len(items)})"):
                                                    for item in items[:5]:
                                                        st.write(f"• {item.get('name', 'Unknown')} - {item.get('distance_km', 0):.2f}km")
                                
                                # Map
                                if generate_map and full_data.get('map_path'):
                                    st.divider()
                                    st.subheader("🗺️ Interactive Map")
                                    map_url = f"{API_URL}/api/neighborhood/{analysis_id}/map"
                                    st.components.v1.iframe(map_url, height=600)
                    else:
                        st.error("❌ Analysis failed or timeout")
    
    # Recent analyses
    st.divider()
    st.subheader("📜 Recent Analyses")
    
    recent = api.get("/api/neighborhood/recent", params={"limit": 5})
    if recent:
        for a in recent:
            with st.expander(f"📍 {a.get('address', 'Unknown')}"):
                st.write(f"Status: {a.get('status', 'unknown')}")
                st.write(f"Walk Score: {a.get('walk_score', 'N/A')}")

# ==================== TAB 3: AI ASSISTANT ====================

with tab3:
    st.header("🤖 AI Real Estate Assistant")
    
    st.markdown("Ask questions about investments, prices, rentals, and market analysis.")
    
    # Examples
    with st.expander("💡 Example Questions"):
        st.markdown("""
        - Calculate ROI for $300,000 property with $2,000 monthly rent
        - Is $450,000 a good price for a 3-bedroom?
        - Investment analysis: $500K house, $2,800 rent, 20% down
        - Fair rent for $400K property?
        """)
    
    # Query input
    query = st.text_area(
        "💬 Your Question",
        placeholder="e.g., Calculate ROI for $300k property with $2k rent",
        height=100
    )
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        ask = st.button("🚀 Ask AI", type="primary", use_container_width=True)
    
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.rerun()
    
    if ask and query:
        with st.spinner("🤔 AI thinking..."):
            response = api.post("/api/agent/query", {"query": query})
            
            if response and response.get('success'):
                st.success("✅ Analysis Complete")
                
                st.markdown("### 💡 AI Response")
                st.markdown(response.get('answer', 'No response'))
                
                # Show calculations if available
                if 'calculations' in response:
                    st.divider()
                    st.subheader("📊 Calculations")
                    
                    calc = response['calculations']
                    cols = st.columns(4)
                    
                    key_metrics = [
                        ('price', 'Price'),
                        ('monthly_rent', 'Rent'),
                        ('monthly_cash_flow', 'Cash Flow'),
                        ('cash_on_cash_roi', 'ROI %')
                    ]
                    
                    for idx, (key, label) in enumerate(key_metrics):
                        if key in calc:
                            with cols[idx]:
                                value = calc[key]
                                st.metric(label, f"${value:,.0f}" if 'price' in key or 'flow' in key or 'rent' in key else f"{value:.1f}%")
                
                # Confidence
                if 'confidence' in response:
                    conf = response['confidence']
                    st.progress(conf)
                    st.caption(f"Confidence: {conf*100:.0f}%")

# ==================== TAB 4: DASHBOARD ====================

with tab4:
    st.header("📊 Analytics Dashboard")
    
    properties = api.get("/api/properties")
    
    if properties:
        df = pd.DataFrame(properties)
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Properties", len(df))
        
        with col2:
            if 'price' in df.columns:
                st.metric("Total Value", f"${df['price'].sum():,.0f}")
        
        with col3:
            if 'price' in df.columns:
                st.metric("Avg Price", f"${df['price'].mean():,.0f}")
        
        with col4:
            if 'city' in df.columns:
                st.metric("Cities", df['city'].nunique())
        
        st.divider()
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            if 'price' in df.columns and len(df) > 1:
                st.subheader("💰 Price Distribution")
                fig = px.histogram(df, x='price', nbins=20, title="Property Prices")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'city' in df.columns and len(df['city'].unique()) > 1:
                st.subheader("🌆 By City")
                city_counts = df['city'].value_counts()
                fig = px.pie(values=city_counts.values, names=city_counts.index)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available")


with tab5:
    st.header("📸 Image Analysis")
    
    st.markdown("""
    Upload street scenes or satellite images for AI-powered analysis:
    - **Street Scene**: Detect cars, pedestrians, buildings, traffic
    - **Green Space**: Calculate vegetation coverage from satellite imagery
    """)
    
    # Analysis type selection
    analysis_type = st.radio(
        "🔬 Analysis Type",
        ["object_detection", "green_space"],
        format_func=lambda x: "🚗 Street Scene (Object Detection)" if x == "object_detection" else "🌳 Green Space Calculator",
        horizontal=True
    )
    
    # File uploader
    uploaded_file = st.file_uploader(
        "📤 Upload Image",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a street scene photo or satellite image"
    )
    
    # Preview uploaded image
    if uploaded_file:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            st.info(f"**Filename:** {uploaded_file.name}")
            st.info(f"**Size:** {uploaded_file.size / 1024:.1f} KB")
            st.info(f"**Type:** {uploaded_file.type}")
    
    # Analyze button
    if uploaded_file and st.button("🚀 Analyze Image", type="primary", use_container_width=True):
        
        st.divider()
        
        # Upload to API
        with st.spinner("📤 Uploading image..."):
            try:
                # Prepare file
                files = {
                    'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                }
                
                params = {
                    'analysis_type': analysis_type
                }
                
                # Upload
                response = requests.post(
                    f"{API_URL}/api/analysis/image",
                    files=files,
                    params=params,
                    timeout=30
                )
                
                if response.status_code == 202:
                    result = response.json()
                    task_id = result.get('task_id')
                    
                    st.success(f"✅ Upload successful! Task ID: {task_id}")
                    
                    # Poll for results
                    st.subheader("⚙️ Processing Image")
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    max_wait = 120
                    start_time = time.time()
                    
                    while time.time() - start_time < max_wait:
                        # Get status
                        status_response = requests.get(f"{API_URL}/api/tasks/{task_id}")
                        
                        if status_response.status_code == 200:
                            status_data = status_response.json()
                            status_val = status_data.get('status')
                            progress = status_data.get('progress', 0)
                            
                            # Update UI
                            progress_bar.progress(progress / 100)
                            
                            if status_val == 'pending':
                                status_text.info(f"⏳ Queued...")
                            elif status_val == 'processing':
                                status_text.info(f"⚙️ Processing... {progress}%")
                            elif status_val == 'completed':
                                status_text.success("✅ Analysis complete!")
                                progress_bar.progress(1.0)
                                
                                # Display results
                                st.divider()
                                st.subheader("📊 Analysis Results")
                                
                                analysis_result = status_data.get('result', {})
                                
                                if analysis_type == "object_detection":
                                    # Street scene results
                                    detections = analysis_result.get('detections', [])
                                    class_counts = analysis_result.get('class_counts', {})
                                    
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        st.metric("Total Objects", len(detections))
                                    
                                    with col2:
                                        st.metric("Object Types", len(class_counts))
                                    
                                    with col3:
                                        if 'car' in class_counts:
                                            st.metric("Vehicles", class_counts.get('car', 0))
                                    
                                    # Object breakdown
                                    if class_counts:
                                        st.subheader("🏷️ Objects Detected")
                                        
                                        # Create bar chart
                                        import pandas as pd
                                        df = pd.DataFrame(
                                            list(class_counts.items()),
                                            columns=['Object', 'Count']
                                        )
                                        df = df.sort_values('Count', ascending=False)
                                        
                                        fig = px.bar(
                                            df,
                                            x='Object',
                                            y='Count',
                                            title="Object Distribution",
                                            color='Count'
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                        
                                        # Detailed list
                                        with st.expander("📋 Detailed Detections"):
                                            for obj_class, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
                                                st.write(f"**{obj_class.title()}:** {count}")
                                    
                                    # Annotated image
                                    annotated_path = analysis_result.get('annotated_image_path')
                                    if annotated_path:
                                        st.subheader("🖼️ Annotated Image")
                                        try:
                                            st.image(annotated_path, caption="Detected Objects", use_column_width=True)
                                        except:
                                            st.info(f"Annotated image saved at: {annotated_path}")
                                
                                elif analysis_type == "green_space":
                                    # Green space results
                                    green_pct = analysis_result.get('green_space_percentage', 0)
                                    total_pixels = analysis_result.get('total_pixels', 0)
                                    green_pixels = analysis_result.get('green_pixels', 0)
                                    
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        st.metric("🌳 Green Space", f"{green_pct:.2f}%")
                                    
                                    with col2:
                                        st.metric("Green Pixels", f"{green_pixels:,}")
                                    
                                    with col3:
                                        st.metric("Total Pixels", f"{total_pixels:,}")
                                    
                                    # Gauge chart
                                    import plotly.graph_objects as go
                                    
                                    fig = go.Figure(go.Indicator(
                                        mode="gauge+number",
                                        value=green_pct,
                                        title={'text': "Green Space Coverage"},
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
                                                'value': 50
                                            }
                                        }
                                    ))
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Interpretation
                                    st.subheader("📋 Interpretation")
                                    
                                    if green_pct > 50:
                                        st.success("✅ **Excellent green coverage!** This area has abundant vegetation and parks.")
                                    elif green_pct > 25:
                                        st.info("ℹ️ **Good green coverage.** Decent amount of trees and parks.")
                                    elif green_pct > 10:
                                        st.warning("⚠️ **Moderate green coverage.** Could benefit from more green spaces.")
                                    else:
                                        st.error("❌ **Low green coverage.** Limited vegetation in this area.")
                                    
                                    # Visualization
                                    viz_path = analysis_result.get('visualization_path')
                                    if viz_path:
                                        st.subheader("🗺️ Green Space Visualization")
                                        try:
                                            st.image(viz_path, caption="Green Space Overlay", use_column_width=True)
                                        except:
                                            st.info(f"Visualization saved at: {viz_path}")
                                
                                break
                            
                            elif status_val == 'failed':
                                status_text.error(f"❌ Analysis failed: {status_data.get('error')}")
                                break
                        
                        time.sleep(2)
                    
                    else:
                        st.warning("⏱️ Analysis timeout. Please try again.")
                
                else:
                    st.error(f"❌ Upload failed: {response.status_code}")
                    st.error(response.text)
            
            except Exception as e:
                st.error(f"❌ Error: {e}")
    
    # Sample images section
    st.divider()
    st.subheader("📸 Sample Images")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🚗 Street Scene Examples:**
        - City streets with traffic
        - Intersection views
        - Highway scenes
        - Urban pedestrian areas
        """)
    
    with col2:
        st.markdown("""
        **🌳 Green Space Examples:**
        - Satellite images of parks
        - Neighborhood aerial views
        - Urban planning imagery
        - Campus/campus areas
        """)
    
    # Usage tips
    with st.expander("💡 Tips for Best Results"):
        st.markdown("""
        **For Street Scene Analysis:**
        - Use clear, well-lit images
        - Capture from street level
        - Include visible objects (cars, people, signs)
        - Avoid blurry or obstructed views
        
        **For Green Space Analysis:**
        - Use satellite/aerial imagery
        - Ensure good contrast between vegetation and buildings
        - Capture during daylight with clear visibility
        - Higher resolution images work better
        
        **Technical Requirements:**
        - Supported formats: JPG, JPEG, PNG
        - Max file size: 10 MB
        - Recommended resolution: 1024x768 or higher
        """)

# ==================== FOOTER ====================

st.divider()
st.caption(f"🚀 GeoInsight AI v3.0 | API: {API_URL} | {datetime.now().strftime('%Y-%m-%d %H:%M')}")