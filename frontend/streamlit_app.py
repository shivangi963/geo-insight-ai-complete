"""
FIXED GeoInsight AI Frontend
- Corrected API polling logic
- Better error handling
- Environment variable support
- Real-time status updates
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from datetime import datetime
import time
import os

# ==================== CONFIG ====================

st.set_page_config(
    page_title="GeoInsight AI",
    page_icon="🏠",
    layout="wide"
)

# Support both local and Docker environments
API_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

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
    .status-processing {
        background: #FFA726;
        padding: 0.5rem;
        border-radius: 5px;
        color: white;
    }
    .status-completed {
        background: #66BB6A;
        padding: 0.5rem;
        border-radius: 5px;
        color: white;
    }
    .status-failed {
        background: #EF5350;
        padding: 0.5rem;
        border-radius: 5px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ==================== API CLIENT ====================

class APIClient:
    """Centralized API client with proper error handling"""
    
    @staticmethod
    def get(endpoint, params=None, timeout=10):
        try:
            url = f"{API_URL}{endpoint}"
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            st.error(f"🔌 Cannot connect to backend at {API_URL}")
            st.info("💡 Make sure backend is running on port 8000")
            return None
        except requests.exceptions.Timeout:
            st.error("⏱️ Request timeout")
            return None
        except requests.exceptions.HTTPError as e:
            st.error(f"❌ HTTP Error {e.response.status_code}")
            return None
        except Exception as e:
            st.error(f"❌ Error: {e}")
            return None
    
    @staticmethod
    def post(endpoint, data, timeout=30):
        try:
            url = f"{API_URL}{endpoint}"
            response = requests.post(url, json=data, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            st.error(f"🔌 Cannot connect to backend at {API_URL}")
            return None
        except requests.exceptions.HTTPError as e:
            st.error(f"❌ Server error: {e.response.status_code}")
            if e.response.status_code == 400:
                try:
                    detail = e.response.json().get('detail', 'Bad request')
                    st.error(f"Details: {detail}")
                except:
                    pass
            return None
        except Exception as e:
            st.error(f"❌ Error: {e}")
            return None

api = APIClient()

# ==================== TASK POLLING (FIXED) ====================

def poll_task(task_id, max_wait=120):
    """
    FIXED: Poll task status until completion
    Handles both analysis_ prefix and celery task IDs
    """
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    start_time = time.time()
    poll_interval = 2  # Poll every 2 seconds
    
    while time.time() - start_time < max_wait:
        # Get task status
        data = api.get(f"/api/tasks/{task_id}", timeout=5)
        
        if not data:
            progress_bar.empty()
            status_text.empty()
            st.error("❌ Failed to get task status")
            return None
        
        status = data.get('status', 'unknown').lower()
        progress = data.get('progress', 0)
        message = data.get('message', '')
        
        # Update UI
        progress_pct = min(progress / 100.0, 1.0)
        progress_bar.progress(progress_pct)
        
        if status in ['pending', 'queued']:
            status_text.info(f"⏳ Queued... {message}")
        elif status in ['processing', 'progress']:
            status_text.info(f"⚙️ Processing... {message} ({progress}%)")
        elif status in ['completed', 'success']:
            progress_bar.progress(1.0)
            status_text.success("✅ Complete!")
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()
            return data.get('result', data)
        elif status in ['failed', 'failure', 'error']:
            progress_bar.empty()
            error_msg = data.get('error', 'Unknown error')
            status_text.error(f"❌ Failed: {error_msg}")
            time.sleep(2)
            status_text.empty()
            return None
        elif status == 'not_found':
            status_text.warning(f"⚠️ Task not found: {task_id}")
            time.sleep(1)
            # Don't fail immediately, task might still be queuing
        
        time.sleep(poll_interval)
    
    # Timeout
    progress_bar.empty()
    status_text.warning(f"⏱️ Timeout after {max_wait}s. Task may still be running.")
    st.info(f"Task ID: {task_id} - Check recent analyses tab")
    return None

# ==================== SIDEBAR ====================

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/city.png", width=80)
    st.title("⚙️ Settings")
    
    st.caption(f"API: {API_URL}")
    
    # Connection test
    if st.button("🔍 Test Connection", use_container_width=True):
        with st.spinner("Testing..."):
            health = api.get("/health")
            if health:
                st.success(f"✅ Connected")
                st.json({
                    "version": health.get('version'),
                    "status": health.get('status'),
                    "database": health.get('database', 'unknown')
                })
            else:
                st.error("❌ Backend offline")
    
    st.divider()
    
    # Stats
    st.subheader("📊 Quick Stats")
    stats = api.get("/api/stats")
    if stats:
        st.metric("Properties", stats.get('total_properties', 0))
        st.metric("Analyses", stats.get('total_analyses', 0))
        
        # Show system features
        if stats.get('system_status') == 'healthy':
            st.success("✅ System Healthy")
        else:
            st.warning("⚠️ System Degraded")

# ==================== HEADER ====================

st.markdown('<h1 class="main-header">🏠 GeoInsight AI</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666;">AI-Powered Real Estate Intelligence Platform</p>', unsafe_allow_html=True)

# ==================== TABS ====================

tab1, tab2, tab3, tab4 = st.tabs([
    "🏘️ Properties",
    "🗺️ Neighborhood Analysis",
    "🤖 AI Assistant",
    "📊 Dashboard"
])

# ==================== TAB 1: PROPERTIES ====================

with tab1:
    st.header("🏘️ Property Management")
    
    prop_tab1, prop_tab2 = st.tabs(["Browse", "Add New"])
    
    with prop_tab1:
        properties = api.get("/api/properties")
        
        if properties:
            df = pd.DataFrame(properties)
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total", len(df))
            with col2:
                if 'price' in df.columns and len(df) > 0:
                    st.metric("Avg Price", f"${df['price'].mean():,.0f}")
            with col3:
                if 'square_feet' in df.columns and len(df) > 0:
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
            
            st.success(f"📍 Showing {len(df)} properties")
            
            # Display properties
            for idx, row in df.iterrows():
                with st.expander(f"🏠 {row.get('address', 'N/A')} - ${row.get('price', 0):,.0f}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**City:** {row.get('city', 'N/A')}, {row.get('state', 'N/A')}")
                        st.write(f"**Type:** {row.get('property_type', 'N/A')}")
                        st.write(f"**ZIP:** {row.get('zip_code', 'N/A')}")
                    with col2:
                        st.write(f"**Beds:** {row.get('bedrooms', 'N/A')} | **Baths:** {row.get('bathrooms', 'N/A')}")
                        st.write(f"**Size:** {row.get('square_feet', 'N/A'):,} sqft")
                        if row.get('price') and row.get('square_feet'):
                            price_per_sqft = row['price'] / row['square_feet']
                            st.write(f"**$/sqft:** ${price_per_sqft:.2f}")
        else:
            st.warning("No properties loaded. Add some properties to get started!")
    
    with prop_tab2:
        st.subheader("Add New Property")
        
        with st.form("add_property"):
            col1, col2 = st.columns(2)
            
            with col1:
                address = st.text_input("Address *", placeholder="123 Main St")
                city = st.text_input("City *", placeholder="San Francisco")
                state = st.text_input("State *", placeholder="CA")
                zip_code = st.text_input("ZIP *", placeholder="94105")
            
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
                        "price": float(price),
                        "bedrooms": int(bedrooms),
                        "bathrooms": float(bathrooms),
                        "square_feet": int(square_feet),
                        "property_type": property_type,
                        "latitude": 0.0,
                        "longitude": 0.0
                    }
                    
                    with st.spinner("Creating property..."):
                        result = api.post("/api/properties", data)
                        if result:
                            st.success(f"✅ Property added! ID: {result.get('id')}")
                            time.sleep(2)
                            st.rerun()
                else:
                    st.error("❌ Please fill all required fields")

# ==================== TAB 2: ANALYSIS ====================

with tab2:
    st.header("🗺️ Neighborhood Analysis")
    
    st.markdown("""
    Analyze any location for:
    - 🍽️ Nearby amenities (restaurants, cafes, shops)
    - 🚶 Walkability score
    - 🏫 Schools and hospitals
    - 🗺️ Interactive map visualization
    """)
    
    # Input
    address = st.text_input(
        "📍 Enter Address",
        placeholder="e.g., Manipal, Karnataka, India OR 123 Main St, San Francisco, CA",
        help="Full address gives best results. Works globally!"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        radius = st.slider("🔍 Search Radius (meters)", 500, 3000, 1000, 100)
    
    with col2:
        generate_map = st.checkbox("🗺️ Generate Interactive Map", value=True)
    
    # Amenities selection
    st.subheader("🎯 Select Amenities to Search")
    
    amenity_cols = st.columns(4)
    amenities_selected = []
    
    amenity_options = {
        "🍽️ Restaurants": "restaurant",
        "☕ Cafes": "cafe",
        "🏫 Schools": "school",
        "🏥 Hospitals": "hospital",
        "🌳 Parks": "park",
        "🛒 Supermarkets": "supermarket",
        "🏦 Banks": "bank",
        "💊 Pharmacies": "pharmacy"
    }
    
    for idx, (label, value) in enumerate(amenity_options.items()):
        with amenity_cols[idx % 4]:
            # Default select common amenities
            default = value in ['restaurant', 'cafe', 'school', 'park']
            if st.checkbox(label, value=default):
                amenities_selected.append(value)
    
    # Start Analysis Button
    if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
        if not address:
            st.error("❌ Please enter an address")
        elif not amenities_selected:
            st.error("❌ Please select at least one amenity type")
        else:
            st.divider()
            
            with st.spinner("🔄 Creating analysis request..."):
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
                    
                    st.success(f"✅ Analysis queued!")
                    st.info(f"📋 Analysis ID: {analysis_id}")
                    st.caption(f"Task ID: {task_id}")
                    
                    # Poll for completion
                    st.subheader("⚙️ Processing Your Analysis")
                    
                    result = poll_task(task_id, max_wait=120)
                    
                    if result:
                        # Fetch full results from analysis endpoint
                        with st.spinner("📥 Loading complete results..."):
                            # Extract analysis_id from result if available
                            result_analysis_id = result.get('analysis_id', analysis_id)
                            full_data = api.get(f"/api/neighborhood/{result_analysis_id}")
                            
                            if full_data:
                                st.divider()
                                st.success("✅ Analysis Complete!")
                                
                                # Display Results
                                st.subheader("📊 Analysis Results")
                                
                                # Key Metrics
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    walk_score = full_data.get('walk_score', 0)
                                    st.metric("🚶 Walk Score", f"{walk_score:.1f}/100")
                                
                                with col2:
                                    amenities = full_data.get('amenities', {})
                                    total = sum(len(items) for items in amenities.values())
                                    st.metric("📍 Total Amenities", total)
                                
                                with col3:
                                    st.metric("📂 Categories", len(amenities))
                                
                                with col4:
                                    status = full_data.get('status', 'completed')
                                    st.metric("Status", status.title())
                                
                                # Amenities Breakdown
                                st.divider()
                                st.subheader("🎯 Amenities Found")
                                
                                if amenities:
                                    # Create bar chart
                                    amenity_counts = {k.title(): len(v) for k, v in amenities.items() if v}
                                    
                                    if amenity_counts:
                                        fig = px.bar(
                                            x=list(amenity_counts.keys()),
                                            y=list(amenity_counts.values()),
                                            labels={'x': 'Amenity Type', 'y': 'Count'},
                                            title="Amenity Distribution",
                                            color=list(amenity_counts.values()),
                                            color_continuous_scale='Viridis'
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                        
                                        # Detailed lists
                                        st.subheader("📋 Detailed Amenity Lists")
                                        cols = st.columns(3)
                                        for idx, (atype, items) in enumerate(amenities.items()):
                                            if items:
                                                with cols[idx % 3]:
                                                    with st.expander(f"{atype.title()} ({len(items)})"):
                                                        for item in items[:10]:  # Show top 10
                                                            name = item.get('name', 'Unknown')
                                                            dist = item.get('distance_km', 0)
                                                            st.write(f"• {name}")
                                                            st.caption(f"   {dist:.2f} km away")
                                    else:
                                        st.info("No amenities found in this category")
                                
                                # Interactive Map
                                if generate_map and full_data.get('map_path'):
                                    st.divider()
                                    st.subheader("🗺️ Interactive Neighborhood Map")
                                    
                                    map_url = f"{API_URL}/api/neighborhood/{result_analysis_id}/map"
                                    
                                    try:
                                        st.components.v1.iframe(map_url, height=600, scrolling=True)
                                    except Exception as e:
                                        st.error(f"Could not load map: {e}")
                                        st.info("Map may be available at: " + map_url)
                            else:
                                st.error("❌ Could not retrieve full analysis results")
                    else:
                        st.warning("⏱️ Analysis timed out or failed. Check the Recent Analyses tab.")
    
    # Recent Analyses
    st.divider()
    st.subheader("📜 Recent Analyses")
    
    recent = api.get("/api/neighborhood/recent", params={"limit": 10})
    if recent and 'analyses' in recent:
        analyses_list = recent['analyses']
        if analyses_list:
            for a in analyses_list:
                status = a.get('status', 'unknown')
                status_color = {
                    'completed': '🟢',
                    'processing': '🟡',
                    'failed': '🔴',
                    'pending': '⚪'
                }.get(status, '⚪')
                
                with st.expander(f"{status_color} {a.get('address', 'Unknown')} - {status.title()}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Analysis ID:** {a.get('analysis_id', 'N/A')}")
                        st.write(f"**Walk Score:** {a.get('walk_score', 'N/A')}")
                    with col2:
                        st.write(f"**Amenities:** {a.get('total_amenities', 0)}")
                        st.write(f"**Created:** {a.get('created_at', 'N/A')}")
                    
                    if a.get('map_available'):
                        st.info(f"🗺️ Map available at: {API_URL}/api/neighborhood/{a.get('analysis_id')}/map")
        else:
            st.info("No recent analyses found. Create one above!")
    else:
        st.info("No recent analyses available")

# ==================== TAB 3: AI ASSISTANT ====================

with tab3:
    st.header("🤖 AI Real Estate Assistant")
    
    st.markdown("""
    Ask questions about:
    - 💰 Investment analysis and ROI calculations
    - 🏠 Property price evaluations
    - 📊 Rental market analysis
    - 📈 Cash flow projections
    """)
    
    # Example questions
    with st.expander("💡 Example Questions"):
        st.markdown("""
        **Investment Analysis:**
        - Calculate ROI for $300,000 property with $2,000 monthly rent
        - Analyze investment: $450K price, $2,800 rent, 20% down payment
        
        **Price Evaluation:**
        - Is $500,000 a good price for a 3-bedroom house?
        - What's fair market value for property at $750K?
        
        **Rental Analysis:**
        - Fair rent for $400K property?
        - Rental market analysis for $2,500/month apartment
        """)
    
    # Query input
    query = st.text_area(
        "💬 Your Question",
        placeholder="e.g., Calculate ROI for $300k property with $2k monthly rent",
        height=120
    )
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        ask = st.button("🚀 Ask AI", type="primary", use_container_width=True)
    
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.rerun()
    
    if ask and query:
        with st.spinner("🤔 AI analyzing your question..."):
            response = api.post("/api/agent/query", {"query": query})
            
            if response:
                success = response.get('success', False)
                
                if success:
                    st.success("✅ Analysis Complete")
                    
                    # Display answer
                    st.markdown("### 💡 AI Response")
                    answer = response.get('response', {}).get('answer', 'No response available')
                    st.markdown(answer)
                    
                    # Show calculations if available
                    if 'response' in response and 'calculations' in response['response']:
                        st.divider()
                        st.subheader("📊 Detailed Calculations")
                        
                        calc = response['response']['calculations']
                        
                        # Display key metrics
                        cols = st.columns(4)
                        
                        key_metrics = [
                            ('price', 'Property Price', '$'),
                            ('monthly_rent', 'Monthly Rent', '$'),
                            ('monthly_cash_flow', 'Cash Flow', '$'),
                            ('cash_on_cash_roi', 'ROI', '%')
                        ]
                        
                        for idx, (key, label, symbol) in enumerate(key_metrics):
                            if key in calc:
                                with cols[idx]:
                                    value = calc[key]
                                    if symbol == '$':
                                        st.metric(label, f"${value:,.0f}")
                                    else:
                                        st.metric(label, f"{value:.1f}%")
                        
                        # Show all calculations in expandable section
                        with st.expander("🔢 See All Calculations"):
                            st.json(calc)
                    
                    # Confidence score
                    if 'confidence' in response:
                        conf = response['confidence']
                        st.progress(conf)
                        st.caption(f"Confidence: {conf*100:.0f}%")
                else:
                    st.warning("⚠️ AI could not process this query")
                    if 'error' in response:
                        st.error(f"Error: {response['error']}")

# ==================== TAB 4: DASHBOARD ====================

with tab4:
    st.header("📊 Analytics Dashboard")
    
    properties = api.get("/api/properties")
    
    if properties and len(properties) > 0:
        df = pd.DataFrame(properties)
        
        # Top Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Properties", len(df))
        
        with col2:
            if 'price' in df.columns:
                total_value = df['price'].sum()
                st.metric("Portfolio Value", f"${total_value:,.0f}")
        
        with col3:
            if 'price' in df.columns:
                avg_price = df['price'].mean()
                st.metric("Avg Price", f"${avg_price:,.0f}")
        
        with col4:
            if 'city' in df.columns:
                st.metric("Markets", df['city'].nunique())
        
        st.divider()
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            if 'price' in df.columns and len(df) > 1:
                st.subheader("💰 Price Distribution")
                fig = px.histogram(
                    df, 
                    x='price', 
                    nbins=20, 
                    title="Property Price Distribution",
                    labels={'price': 'Price ($)', 'count': 'Number of Properties'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'city' in df.columns and len(df['city'].unique()) > 1:
                st.subheader("🌆 Properties by City")
                city_counts = df['city'].value_counts()
                fig = px.pie(
                    values=city_counts.values, 
                    names=city_counts.index,
                    title="Geographic Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Additional insights
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            if 'bedrooms' in df.columns:
                st.subheader("🛏️ Bedroom Distribution")
                bed_counts = df['bedrooms'].value_counts().sort_index()
                fig = px.bar(
                    x=bed_counts.index,
                    y=bed_counts.values,
                    labels={'x': 'Bedrooms', 'y': 'Count'},
                    title="Properties by Bedroom Count"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'property_type' in df.columns:
                st.subheader("🏠 Property Types")
                type_counts = df['property_type'].value_counts()
                fig = px.bar(
                    x=type_counts.index,
                    y=type_counts.values,
                    labels={'x': 'Type', 'y': 'Count'},
                    title="Properties by Type"
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📭 No data available. Add some properties to see analytics!")

# ==================== FOOTER ====================

st.divider()
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.caption(f"🚀 GeoInsight AI v4.0 | Backend: {API_URL}")

with col2:
    st.caption(f"📅 {datetime.now().strftime('%Y-%m-%d')}")

with col3:
    st.caption(f"🕐 {datetime.now().strftime('%H:%M:%S')}")