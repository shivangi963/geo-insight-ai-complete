"""
COMPLETE GeoInsight AI Frontend
Uses ALL backend features:
- Properties CRUD
- Neighborhood Analysis with Maps
- AI Agent with Investment Calculations  
- Image Analysis (Street Scene + Green Space)
- Vector DB Search (if enabled)
- Recent Analyses
- Statistics Dashboard
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime
import time
from io import BytesIO

# ==================== CONFIG ====================

st.set_page_config(
    page_title="GeoInsight AI - Complete Platform",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "http://localhost:8000"

# ==================== CUSTOM STYLES ====================

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .warning-box {
        padding: 1rem;
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        border-radius: 4px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE ====================

if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

if 'agent_history' not in st.session_state:
    st.session_state.agent_history = []

# ==================== API CLIENT ====================

class APIClient:
    """Complete API client for all backend endpoints"""
    
    @staticmethod
    def get(endpoint, params=None, timeout=10):
        """GET request"""
        try:
            response = requests.get(f"{API_URL}{endpoint}", params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            st.error("🔌 **Backend Connection Failed**")
            st.error(f"Cannot connect to {API_URL}")
            st.info("Make sure backend is running: `uvicorn app.main:app --reload`")
            return None
        except requests.exceptions.Timeout:
            st.error("⏱️ Request timeout")
            return None
        except requests.exceptions.HTTPError as e:
            st.error(f"❌ HTTP {e.response.status_code}")
            try:
                error_detail = e.response.json().get('detail', e.response.text)
                st.error(f"Details: {error_detail}")
            except:
                st.error(e.response.text)
            return None
        except Exception as e:
            st.error(f"❌ Error: {e}")
            return None
    
    @staticmethod
    def post(endpoint, data=None, files=None, timeout=30):
        """POST request"""
        try:
            if files:
                response = requests.post(f"{API_URL}{endpoint}", files=files, timeout=timeout)
            else:
                response = requests.post(f"{API_URL}{endpoint}", json=data, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            st.error("🔌 Backend connection failed")
            return None
        except requests.exceptions.HTTPError as e:
            st.error(f"❌ Server error {e.response.status_code}")
            try:
                error_detail = e.response.json().get('detail', 'Unknown error')
                st.error(f"Details: {error_detail}")
            except:
                st.error(e.response.text)
            return None
        except Exception as e:
            st.error(f"❌ Error: {e}")
            return None
    
    @staticmethod
    def put(endpoint, data, timeout=30):
        """PUT request"""
        try:
            response = requests.put(f"{API_URL}{endpoint}", json=data, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"❌ Error: {e}")
            return None
    
    @staticmethod
    def delete(endpoint, timeout=30):
        """DELETE request"""
        try:
            response = requests.delete(f"{API_URL}{endpoint}", timeout=timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"❌ Error: {e}")
            return None

api = APIClient()

# ==================== TASK POLLING ====================

def poll_task(task_id, max_wait=120, show_progress=True):
    """Poll task status with progress bar"""
    if show_progress:
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        data = api.get(f"/api/tasks/{task_id}")
        
        if not data:
            if show_progress:
                progress_bar.empty()
                status_text.empty()
            return None
        
        status = data.get('status', 'unknown')
        progress = data.get('progress', 0)
        message = data.get('message', '')
        
        if show_progress:
            progress_bar.progress(min(progress / 100, 1.0))
            
            if status == 'pending':
                status_text.info(f"⏳ Queued: {message}")
            elif status == 'processing':
                status_text.info(f"⚙️ {message} ({progress}%)")
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
        else:
            if status == 'completed':
                return data.get('result', {})
            elif status == 'failed':
                st.error(f"Task failed: {data.get('error')}")
                return None
        
        time.sleep(2)
    
    if show_progress:
        progress_bar.empty()
        status_text.warning(f"⏱️ Timeout. Task: {task_id}")
    
    return None

# ==================== SIDEBAR ====================

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/city.png", width=100)
    st.title("⚙️ Control Panel")
    
    # Connection Status
    st.subheader("🔗 Connection")
    
    if st.button("🔍 Test Backend", use_container_width=True):
        with st.spinner("Testing..."):
            health = api.get("/health")
            if health:
                st.success("✅ Connected!")
                
                # Show backend info
                with st.expander("📊 Backend Info"):
                    st.json({
                        "Version": health.get('version'),
                        "Status": health.get('status'),
                        "Database": health.get('database'),
                        "Features": health.get('features', {})
                    })
            else:
                st.error("❌ Offline")
    
    st.divider()
    
    # Quick Stats
    st.subheader("📊 Live Stats")
    
    stats = api.get("/api/stats")
    if stats:
        st.metric("🏠 Properties", stats.get('total_properties', 0))
        st.metric("🗺️ Analyses", stats.get('total_analyses', 0))
        st.metric("🏙️ Cities", stats.get('unique_cities', 0))
        
        avg_price = stats.get('average_price', 0)
        if avg_price > 0:
            st.metric("💰 Avg Price", f"${avg_price:,.0f}")
    
    st.divider()
    
    # Features Status
    st.subheader("🔧 Features")
    
    health = api.get("/health")
    if health:
        features = health.get('features', {})
        
        celery_status = "✅" if features.get('celery') else "⚠️"
        st.write(f"{celery_status} Async Tasks")
        
        vector_status = "✅" if features.get('vector_db') else "⚠️"
        st.write(f"{vector_status} Vector Search")
        
        ai_status = "✅" if features.get('ai_agent') else "⚠️"
        st.write(f"{ai_status} AI Agent")

# ==================== HEADER ====================

st.markdown('<h1 class="main-header">🏠 GeoInsight AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Complete Real Estate Intelligence Platform</p>', unsafe_allow_html=True)

# ==================== MAIN TABS ====================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏘️ Properties",
    "🗺️ Neighborhood",
    "🤖 AI Assistant",
    "📸 Image Analysis",
    "🔍 Vector Search",
    "📊 Dashboard"
])

# ==================== TAB 1: PROPERTIES ====================

with tab1:
    st.header("🏘️ Property Management")
    
    prop_tab1, prop_tab2 = st.tabs(["📋 Browse Properties", "➕ Add Property"])
    
    # Browse Properties
    with prop_tab1:
        col1, col2 = st.columns([3, 1])
        
        with col2:
            if st.button("🔄 Refresh", use_container_width=True, key="refresh_props"):
                st.rerun()
        
        with st.spinner("Loading properties..."):
            properties = api.get("/api/properties", params={"limit": 100})
        
        if properties and len(properties) > 0:
            df = pd.DataFrame(properties)
            
            # Metrics Row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📊 Total", len(df))
            
            with col2:
                if 'price' in df.columns:
                    st.metric("💰 Avg Price", f"${df['price'].mean():,.0f}")
            
            with col3:
                if 'square_feet' in df.columns:
                    st.metric("📐 Avg Size", f"{df['square_feet'].mean():,.0f} sqft")
            
            with col4:
                if 'city' in df.columns:
                    st.metric("🏙️ Cities", df['city'].nunique())
            
            st.divider()
            
            # Filters
            col1, col2, col3 = st.columns(3)
            
            filtered_df = df.copy()
            
            with col1:
                if 'city' in df.columns:
                    cities = ['All'] + sorted(df['city'].dropna().unique().tolist())
                    city_filter = st.selectbox("🏙️ City", cities, key="city_filter")
                    if city_filter != 'All':
                        filtered_df = filtered_df[filtered_df['city'] == city_filter]
            
            with col2:
                if 'property_type' in df.columns:
                    types = ['All'] + sorted(df['property_type'].dropna().unique().tolist())
                    type_filter = st.selectbox("🏠 Type", types, key="type_filter")
                    if type_filter != 'All':
                        filtered_df = filtered_df[filtered_df['property_type'] == type_filter]
            
            with col3:
                if 'bedrooms' in df.columns:
                    bedrooms = ['All'] + sorted(df['bedrooms'].dropna().unique().tolist())
                    bed_filter = st.selectbox("🛏️ Bedrooms", bedrooms, key="bed_filter")
                    if bed_filter != 'All':
                        filtered_df = filtered_df[filtered_df['bedrooms'] == bed_filter]
            
            # Price Range
            if 'price' in df.columns and len(df) > 0:
                min_p = int(df['price'].min())
                max_p = int(df['price'].max())
                if min_p < max_p:
                    price_range = st.slider(
                        "💵 Price Range",
                        min_p, max_p, (min_p, max_p),
                        key="price_range"
                    )
                    filtered_df = filtered_df[
                        (filtered_df['price'] >= price_range[0]) &
                        (filtered_df['price'] <= price_range[1])
                    ]
            
            st.success(f"📍 Showing {len(filtered_df)} properties")
            
            # Display Properties
            for idx, row in filtered_df.iterrows():
                with st.expander(
                    f"🏠 {row.get('address', 'N/A')} | ${row.get('price', 0):,.0f}",
                    expanded=False
                ):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**📍 Location**")
                        st.write(f"{row.get('city', 'N/A')}, {row.get('state', 'N/A')}")
                        st.write(f"ZIP: {row.get('zip_code', 'N/A')}")
                    
                    with col2:
                        st.markdown("**🏘️ Details**")
                        st.write(f"Type: {row.get('property_type', 'N/A')}")
                        st.write(f"Beds: {row.get('bedrooms', 'N/A')} | Baths: {row.get('bathrooms', 'N/A')}")
                    
                    with col3:
                        st.markdown("**📊 Metrics**")
                        st.write(f"Size: {row.get('square_feet', 0):,} sqft")
                        price = row.get('price', 0)
                        sqft = row.get('square_feet', 1)
                        price_per_sqft = price / sqft if sqft > 0 else 0
                        st.write(f"$/sqft: ${price_per_sqft:,.0f}")
                    
                    # Action Buttons
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button("🗺️ Analyze Area", key=f"analyze_{idx}"):
                            st.session_state.analyze_address = row.get('address')
                            st.switch_page
                    
                    with col2:
                        if st.button("🤖 AI Analysis", key=f"ai_{idx}"):
                            query = f"Analyze investment for ${price:,.0f} property at {row.get('address')}"
                            st.session_state.ai_query = query
        
        else:
            st.info("📭 No properties in database")
            
            with st.expander("💡 How to Add Properties"):
                st.markdown("""
                **Option 1: Use the form in 'Add Property' tab**
                
                **Option 2: Load from Kaggle CSV**
                ```bash
                cd backend
                python load_kaggle_data.py data/your_dataset.csv
                ```
                
                **Option 3: Use the API directly**
                ```bash
                curl -X POST "http://localhost:8000/api/properties" \\
                  -H "Content-Type: application/json" \\
                  -d '{...}'
                ```
                """)
    
    # Add Property
    with prop_tab2:
        st.subheader("➕ Add New Property")
        
        with st.form("add_property_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                address = st.text_input("📍 Address *", placeholder="123 Main Street")
                city = st.text_input("🏙️ City *", placeholder="San Francisco")
                state = st.text_input("🗺️ State *", placeholder="CA")
                zip_code = st.text_input("📮 ZIP Code *", placeholder="94105")
            
            with col2:
                price = st.number_input("💰 Price ($) *", min_value=0, value=500000, step=10000)
                bedrooms = st.number_input("🛏️ Bedrooms *", min_value=0, value=3, step=1)
                bathrooms = st.number_input("🚿 Bathrooms *", min_value=0.0, value=2.0, step=0.5)
                square_feet = st.number_input("📐 Square Feet *", min_value=0, value=1500, step=100)
            
            property_type = st.selectbox(
                "🏠 Property Type *",
                ["Single Family", "Condo", "Apartment", "Townhouse", "Multi-Family"]
            )
            
            description = st.text_area(
                "📝 Description (optional)",
                placeholder="Beautiful property with..."
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                latitude = st.number_input("🌐 Latitude", value=0.0, format="%.6f")
            
            with col2:
                longitude = st.number_input("🌐 Longitude", value=0.0, format="%.6f")
            
            submitted = st.form_submit_button("➕ Add Property", type="primary", use_container_width=True)
            
            if submitted:
                if all([address, city, state, zip_code]):
                    with st.spinner("Adding property..."):
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
                            "description": description if description else None,
                            "latitude": float(latitude),
                            "longitude": float(longitude)
                        }
                        
                        result = api.post("/api/properties", data)
                        
                        if result:
                            st.success(f"✅ Property added! ID: {result.get('id')}")
                            st.balloons()
                            time.sleep(2)
                            st.rerun()
                else:
                    st.error("❌ Please fill all required fields (*)")

# ==================== TAB 2: NEIGHBORHOOD ANALYSIS ====================

with tab2:
    st.header("🗺️ Neighborhood Intelligence")
    
    st.markdown("Analyze any location for amenities, walkability, and local insights using OpenStreetMap data.")
    
    # Pre-fill from property if coming from there
    default_address = st.session_state.get('analyze_address', '')
    
    with st.form("neighborhood_form"):
        address = st.text_input(
            "📍 Enter Full Address",
            value=default_address,
            placeholder="e.g., MIT Campus, Manipal, Karnataka, India",
            help="More specific = better results"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            radius = st.slider("🔍 Search Radius (meters)", 500, 3000, 1000, 100)
        
        with col2:
            generate_map = st.checkbox("🗺️ Generate Interactive Map", value=True)
        
        st.markdown("### 🎯 Select Amenities")
        
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
        
        cols = st.columns(4)
        amenities_selected = []
        
        for idx, (label, value) in enumerate(amenity_options.items()):
            with cols[idx % 4]:
                default = value in ['restaurant', 'cafe', 'school', 'hospital']
                if st.checkbox(label, value=default, key=f"amenity_nb_{value}"):
                    amenities_selected.append(value)
        
        submitted = st.form_submit_button("🚀 Start Analysis", type="primary", use_container_width=True)
    
    if submitted:
        if not address:
            st.error("❌ Please enter an address")
        elif not amenities_selected:
            st.error("❌ Select at least one amenity")
        else:
            st.divider()
            st.subheader("⚙️ Running Analysis")
            
            # Start analysis
            with st.spinner("Creating analysis..."):
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
                
                st.success(f"✅ Analysis started!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**Analysis ID:** `{analysis_id}`")
                with col2:
                    st.info(f"**Task ID:** `{task_id}`")
                
                # Poll for results
                result = poll_task(task_id, max_wait=120)
                
                if result:
                    # Save to history
                    st.session_state.analysis_history.append({
                        'address': address,
                        'analysis_id': analysis_id,
                        'timestamp': datetime.now()
                    })
                    
                    st.divider()
                    st.subheader("📊 Analysis Results")
                    
                    # Key Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        walk_score = result.get('walk_score', 0)
                        st.metric("🚶 Walk Score", f"{walk_score:.1f}/100")
                    
                    with col2:
                        total = result.get('total_amenities', 0)
                        st.metric("📍 Amenities", total)
                    
                    with col3:
                        coords = result.get('coordinates')
                        if coords:
                            st.metric("🌍 Lat", f"{coords[0]:.4f}")
                    
                    with col4:
                        if coords:
                            st.metric("🌍 Lon", f"{coords[1]:.4f}")
                    
                    # Walkability Interpretation
                    st.divider()
                    
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
                    
                    # Amenities Breakdown
                    amenities = result.get('amenities', {})
                    
                    if amenities:
                        st.divider()
                        st.subheader("🎯 Amenities Found")
                        
                        # Chart
                        amenity_counts = {k.title(): len(v) for k, v in amenities.items() if v}
                        
                        if amenity_counts:
                            fig = px.bar(
                                x=list(amenity_counts.keys()),
                                y=list(amenity_counts.values()),
                                labels={'x': 'Amenity Type', 'y': 'Count'},
                                title="Amenity Distribution",
                                color=list(amenity_counts.values()),
                                color_continuous_scale='viridis'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Detailed Lists
                        st.markdown("### 📋 Nearby Places")
                        
                        cols = st.columns(3)
                        
                        for idx, (atype, items) in enumerate(amenities.items()):
                            if items:
                                with cols[idx % 3]:
                                    with st.expander(f"{atype.title()} ({len(items)})"):
                                        for i, item in enumerate(items[:10], 1):
                                            name = item.get('name', 'Unknown')
                                            dist = item.get('distance_km', 0)
                                            st.write(f"{i}. **{name}**")
                                            st.caption(f"   {dist:.2f} km away")
                    
                    # Interactive Map
                    if generate_map:
                        st.divider()
                        st.subheader("🗺️ Interactive Map")
                        
                        map_url = f"{API_URL}/api/neighborhood/{analysis_id}/map"
                        
                        try:
                            st.components.v1.iframe(map_url, height=600, scrolling=True)
                        except Exception as e:
                            st.error(f"Map error: {e}")
                            st.info(f"[Open map in new tab]({map_url})")
    
    # Recent Analyses
    st.divider()
    st.subheader("📜 Recent Analyses")
    
    recent = api.get("/api/neighborhood/recent", params={"limit": 10})
    
    if recent and len(recent) > 0:
        for analysis in recent:
            status = analysis.get('status', 'unknown')
            status_emoji = "✅" if status == 'completed' else "⏳" if status == 'processing' else "❌"
            
            with st.expander(f"{status_emoji} {analysis.get('address', 'Unknown')}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**Status:** {status}")
                    st.write(f"**Walk Score:** {analysis.get('walk_score', 'N/A')}")
                
                with col2:
                    st.write(f"**Amenities:** {analysis.get('total_amenities', 0)}")
                    st.write(f"**Map:** {'✅' if analysis.get('map_available') else '❌'}")
                
                with col3:
                    created = analysis.get('created_at', 'N/A')
                    st.write(f"**Created:** {created}")
                    
                    if st.button("📊 View Details", key=f"view_{analysis.get('analysis_id')}"):
                        st.info("Feature coming soon!")
    else:
        st.info("No analyses yet")

# ==================== TAB 3: AI ASSISTANT ====================

with tab3:
    st.header("🤖 AI Real Estate Assistant")
    
    st.markdown("Get instant investment analysis, property valuations, and market insights powered by AI.")
    
    # Example queries
    with st.expander("💡 Example Questions"):
        st.markdown("""
        **📊 Investment Analysis:**
        - `Calculate ROI for $300,000 property with $2,000 monthly rent`
        - `Investment analysis: $500K property, $2,800/month rent, 20% down`
        - `Analyze cash flow for $450K house with $2,500 rent`
        
        **💰 Property Valuation:**
        - `Is $450,000 a good price for a 3-bedroom house?`
        - `What's fair market value for $750K property?`
        - `Price analysis for $600K condo`
        
        **🏠 Rental Analysis:**
        - `Fair rent for $400K property?`
        - `Rental market for $2,500/month apartment`
        - `Expected rent for $350K house`
        """)
    
    # Query input
    default_query = st.session_state.get('ai_query', '')
    
    query = st.text_area(
        "💬 Ask Your Question",
        value=default_query,
        placeholder="e.g., Calculate ROI for $300,000 property with $2,000 monthly rent",
        height=100,
        key="ai_assistant_query"
    )
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if st.button("🚀 Ask AI Assistant", type="primary", use_container_width=True, key="ask_ai"):
            if query:
                with st.spinner("🤔 AI analyzing..."):
                    response = api.post("/api/agent/query", {"query": query})
                
                if response and response.get('success'):
                    # Save to history
                    st.session_state.agent_history.append({
                        'query': query,
                        'response': response,
                        'timestamp': datetime.now()
                    })
                    
                    st.success("✅ Analysis Complete")
                    
                    # Display answer
                    st.markdown("### 💡 AI Response")
                    
                    answer = response.get('answer', '')
                    st.markdown(answer)
                    
                    # Show calculations
                    calculations = response.get('calculations')
                    
                    if calculations:
                        st.divider()
                        st.subheader("📊 Investment Breakdown")
                        
                        # Key metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            if 'price' in calculations:
                                st.metric("🏠 Price", f"${calculations['price']:,.0f}")
                        
                        with col2:
                            if 'monthly_rent' in calculations:
                                st.metric("💵 Rent/Mo", f"${calculations['monthly_rent']:,.0f}")
                        
                        with col3:
                            if 'monthly_cash_flow' in calculations:
                                flow = calculations['monthly_cash_flow']
                                delta_color = "normal" if flow > 0 else "inverse"
                                st.metric("💰 Cash Flow", f"${flow:,.0f}", delta=None)
                        
                        with col4:
                            if 'cash_on_cash_roi' in calculations:
                                roi = calculations['cash_on_cash_roi']
                                st.metric("📈 ROI", f"{roi:.1f}%")
                        
                        # Additional details
                        with st.expander("📋 Full Financial Details"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**💰 Investment**")
                                st.write(f"Down Payment: ${calculations.get('down_payment', 0):,.0f}")
                                st.write(f"Down %: {calculations.get('down_payment_pct', 0):.0f}%")
                                st.write(f"Loan Amount: ${calculations.get('loan_amount', 0):,.0f}")
                                st.write(f"Interest Rate: {calculations.get('interest_rate', 0):.1f}%")
                            
                            with col2:
                                st.markdown("**📊 Monthly**")
                                st.write(f"Mortgage: ${calculations.get('monthly_mortgage', 0):,.0f}")
                                st.write(f"Expenses: ${calculations.get('monthly_expenses', 0):,.0f}")
                                st.write(f"Net Income: ${calculations.get('monthly_cash_flow', 0):,.0f}")
                            
                            st.markdown("**📈 Annual**")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.write(f"Gross Rent: ${calculations.get('annual_rent', 0):,.0f}")
                            
                            with col2:
                                st.write(f"Expenses: ${calculations.get('annual_expenses', 0):,.0f}")
                            
                            with col3:
                                st.write(f"NOI: ${calculations.get('annual_net_income', 0):,.0f}")
                            
                            st.markdown("**🎯 Key Ratios**")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.write(f"Rental Yield: {calculations.get('rental_yield', 0):.2f}%")
                            
                            with col2:
                                st.write(f"Cap Rate: {calculations.get('cap_rate', 0):.2f}%")
                            
                            with col3:
                                st.write(f"Break-Even: {calculations.get('break_even_occupancy', 0):.1f}%")
                    
                    # Confidence
                    confidence = response.get('confidence', 0)
                    if confidence:
                        st.divider()
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.progress(confidence)
                        
                        with col2:
                            st.metric("🎯 Confidence", f"{confidence*100:.0f}%")
            else:
                st.warning("⚠️ Please enter a question")
    
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.ai_query = ''
            st.rerun()
    
    with col3:
        if st.button("📜 History", use_container_width=True):
            st.session_state.show_ai_history = not st.session_state.get('show_ai_history', False)
    
    # Show history
    if st.session_state.get('show_ai_history', False):
        st.divider()
        st.subheader("📜 Query History")
        
        if st.session_state.agent_history:
            for idx, item in enumerate(reversed(st.session_state.agent_history[-10:])):
                with st.expander(f"🤖 {item['query'][:50]}..."):
                    st.write(f"**Asked:** {item['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                    st.markdown(item['response'].get('answer', 'No answer')[:200] + '...')
        else:
            st.info("No history yet")

# ==================== TAB 4: IMAGE ANALYSIS ====================

with tab4:
    st.header("📸 Computer Vision Analysis")
    
    st.markdown("""
    Upload images for AI-powered analysis:
    - **🚗 Street Scene:** Detect vehicles, pedestrians, traffic
    - **🌳 Green Space:** Calculate vegetation coverage
    """)
    
    # Analysis type
    analysis_type = st.radio(
        "🔬 Analysis Type",
        ["object_detection", "green_space"],
        format_func=lambda x: "🚗 Street Scene (Object Detection)" if x == "object_detection" else "🌳 Green Space Calculator",
        horizontal=True,
        key="img_analysis_type"
    )
    
    # File upload
    uploaded_file = st.file_uploader(
        "📤 Upload Image",
        type=['jpg', 'jpeg', 'png'],
        help="Max 10MB",
        key="img_upload"
    )
    
    # Preview
    if uploaded_file:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            st.info(f"**File:** {uploaded_file.name}")
            st.info(f"**Size:** {uploaded_file.size / 1024:.1f} KB")
            st.info(f"**Type:** {uploaded_file.type}")
        
        # Analyze button
        if st.button("🚀 Analyze Image", type="primary", use_container_width=True, key="analyze_img"):
            st.divider()
            
            # Upload to API
            with st.spinner("📤 Uploading..."):
                files = {
                    'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                }
                
                params = {'analysis_type': analysis_type}
                
                try:
                    response = requests.post(
                        f"{API_URL}/api/analysis/image",
                        files=files,
                        params=params,
                        timeout=30
                    )
                    
                    if response.status_code == 202:
                        result = response.json()
                        task_id = result.get('task_id')
                        
                        st.success(f"✅ Upload successful! Task: {task_id}")
                        
                        # Poll
                        st.subheader("⚙️ Processing Image")
                        
                        analysis_result = poll_task(task_id, max_wait=120)
                        
                        if analysis_result:
                            st.divider()
                            st.subheader("📊 Results")
                            
                            if analysis_type == "object_detection":
                                # Street scene
                                detections = analysis_result.get('detections', [])
                                class_counts = analysis_result.get('class_counts', {})
                                
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("🎯 Objects", len(detections))
                                
                                with col2:
                                    st.metric("🏷️ Types", len(class_counts))
                                
                                with col3:
                                    cars = class_counts.get('car', 0)
                                    st.metric("🚗 Vehicles", cars)
                                
                                # Chart
                                if class_counts:
                                    st.divider()
                                    
                                    df = pd.DataFrame(
                                        list(class_counts.items()),
                                        columns=['Object', 'Count']
                                    ).sort_values('Count', ascending=False)
                                    
                                    fig = px.bar(
                                        df, x='Object', y='Count',
                                        title="Detected Objects",
                                        color='Count',
                                        color_continuous_scale='blues'
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                # Annotated image
                                annotated = analysis_result.get('annotated_image_path')
                                if annotated:
                                    st.divider()
                                    st.subheader("🖼️ Annotated Image")
                                    
                                    try:
                                        st.image(annotated, use_container_width=True)
                                    except:
                                        st.info(f"Saved: {annotated}")
                            
                            else:
                                # Green space
                                green_pct = analysis_result.get('green_space_percentage', 0)
                                total_px = analysis_result.get('total_pixels', 0)
                                green_px = analysis_result.get('green_pixels', 0)
                                
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("🌳 Green %", f"{green_pct:.2f}%")
                                
                                with col2:
                                    st.metric("Green Pixels", f"{green_px:,}")
                                
                                with col3:
                                    st.metric("Total Pixels", f"{total_px:,}")
                                
                                # Gauge
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
                                        ]
                                    }
                                ))
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Interpretation
                                st.divider()
                                
                                if green_pct > 50:
                                    st.success("🌟 Excellent green coverage!")
                                elif green_pct > 25:
                                    st.info("✅ Good green coverage")
                                elif green_pct > 10:
                                    st.warning("⚠️ Moderate coverage")
                                else:
                                    st.error("❌ Low green coverage")
                                
                                # Viz
                                viz_path = analysis_result.get('visualization_path')
                                if viz_path:
                                    st.divider()
                                    st.subheader("🗺️ Visualization")
                                    
                                    try:
                                        st.image(viz_path, use_container_width=True)
                                    except:
                                        st.info(f"Saved: {viz_path}")
                    
                    else:
                        st.error(f"Upload failed: {response.status_code}")
                        st.error(response.text)
                
                except Exception as e:
                    st.error(f"Error: {e}")

# ==================== TAB 5: VECTOR SEARCH ====================

with tab5:
    st.header("🔍 Vector Similarity Search")
    
    # Check if available
    health = api.get("/health")
    vector_enabled = health and health.get('features', {}).get('vector_db', False)
    
    if not vector_enabled:
        st.warning("⚠️ Vector database not configured")
        st.info("""
        To enable vector search:
        1. Set up Supabase account
        2. Add credentials to backend/.env
        3. Install required packages
        4. Restart backend
        """)
    else:
        st.success("✅ Vector search enabled!")
        
        search_tab, store_tab, stats_tab = st.tabs(["🔍 Search", "💾 Store", "📊 Stats"])
        
        with search_tab:
            st.subheader("Find Similar Properties")
            
            uploaded = st.file_uploader("📤 Upload property image", type=['jpg', 'jpeg', 'png'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                limit = st.slider("Results", 1, 20, 5)
            
            with col2:
                threshold = st.slider("Similarity", 0.0, 1.0, 0.7, 0.05)
            
            if uploaded and st.button("🔍 Search", type="primary"):
                with st.spinner("Searching..."):
                    files = {
                        'file': (uploaded.name, uploaded.getvalue(), uploaded.type)
                    }
                    
                    params = {'limit': limit, 'threshold': threshold}
                    
                    try:
                        response = requests.post(
                            f"{API_URL}/api/vector/search",
                            files=files,
                            params=params
                        )
                        
                        if response.status_code == 200:
                            data = response.json()
                            results = data.get('results', [])
                            
                            if results:
                                st.success(f"Found {len(results)} similar properties")
                                
                                for r in results:
                                    with st.expander(f"📍 {r.get('address')} - Similarity: {r.get('similarity', 0):.2%}"):
                                        st.json(r.get('metadata', {}))
                            else:
                                st.info("No similar properties found")
                        else:
                            st.error(f"Search failed: {response.status_code}")
                    
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        with store_tab:
            st.subheader("Store Property Embedding")
            st.info("Feature requires additional setup")
        
        with stats_tab:
            st.subheader("Vector DB Statistics")
            
            stats = api.get("/api/vector/stats")
            
            if stats:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Total Properties", stats.get('total_properties', 0))
                
                with col2:
                    st.metric("Embedding Dim", stats.get('embedding_dimension', 0))

# ==================== TAB 6: DASHBOARD ====================

with tab6:
    st.header("📊 Analytics Dashboard")
    
    # Load data
    properties = api.get("/api/properties", params={"limit": 1000})
    stats = api.get("/api/stats")
    
    if properties and len(properties) > 0:
        df = pd.DataFrame(properties)
        
        # Top metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("🏠 Properties", len(df))
        
        with col2:
            if 'price' in df.columns:
                st.metric("💰 Total Value", f"${df['price'].sum():,.0f}")
        
        with col3:
            if 'price' in df.columns:
                st.metric("📊 Avg Price", f"${df['price'].mean():,.0f}")
        
        with col4:
            if 'city' in df.columns:
                st.metric("🏙️ Cities", df['city'].nunique())
        
        with col5:
            if stats:
                st.metric("🗺️ Analyses", stats.get('total_analyses', 0))
        
        st.divider()
        
        # Charts row 1
        col1, col2 = st.columns(2)
        
        with col1:
            if 'price' in df.columns:
                st.subheader("💰 Price Distribution")
                
                fig = px.histogram(
                    df, x='price',
                    nbins=30,
                    title="Property Prices",
                    color_discrete_sequence=['#667eea']
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'city' in df.columns and df['city'].nunique() > 1:
                st.subheader("🌆 By City")
                
                city_counts = df['city'].value_counts()
                
                fig = px.pie(
                    values=city_counts.values,
                    names=city_counts.index,
                    title="Distribution by City"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Charts row 2
        col1, col2 = st.columns(2)
        
        with col1:
            if 'property_type' in df.columns:
                st.subheader("🏘️ Property Types")
                
                type_counts = df['property_type'].value_counts()
                
                fig = px.bar(
                    x=type_counts.index,
                    y=type_counts.values,
                    labels={'x': 'Type', 'y': 'Count'},
                    color=type_counts.values,
                    color_continuous_scale='viridis'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'bedrooms' in df.columns:
                st.subheader("🛏️ Bedroom Distribution")
                
                bed_counts = df['bedrooms'].value_counts().sort_index()
                
                fig = px.bar(
                    x=bed_counts.index,
                    y=bed_counts.values,
                    labels={'x': 'Bedrooms', 'y': 'Count'},
                    color=bed_counts.values,
                    color_continuous_scale='blues'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Price vs Size
        st.divider()
        
        if 'price' in df.columns and 'square_feet' in df.columns:
            st.subheader("📐 Price vs Square Feet")
            
            fig = px.scatter(
                df,
                x='square_feet',
                y='price',
                color='city' if 'city' in df.columns else None,
                size='bedrooms' if 'bedrooms' in df.columns else None,
                hover_data=['address'] if 'address' in df.columns else None,
                title="Price vs Size Analysis"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Summary table
        st.divider()
        st.subheader("📋 Summary Statistics")
        
        if 'price' in df.columns:
            summary_df = df[['price', 'bedrooms', 'bathrooms', 'square_feet']].describe()
            st.dataframe(summary_df.style.format("{:.2f}"), use_container_width=True)
    
    else:
        st.info("📭 No data available")
        st.markdown("Add properties to see analytics!")

# ==================== FOOTER ====================

st.divider()

col1, col2, col3, col4 = st.columns(4)

with col1:
    health = api.get("/health")
    version = health.get('version', 'Unknown') if health else 'Offline'
    st.caption(f"🚀 GeoInsight AI v{version}")

with col2:
    st.caption(f"🔗 {API_URL}")

with col3:
    st.caption(f"📅 {datetime.now().strftime('%Y-%m-%d')}")

with col4:
    st.caption(f"🕐 {datetime.now().strftime('%H:%M:%S')}")