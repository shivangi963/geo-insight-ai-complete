import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime
import time

# ==================== PAGE CONFIG ====================

st.set_page_config(
    page_title="GeoInsight AI Platform",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================

st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #1E88E5;
        --secondary-color: #43A047;
        --accent-color: #F57C00;
        --background-color: #F5F7FA;
    }
    
    /* Header styling */
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: var(--primary-color);
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    /* Success message */
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    /* Info box */
    .info-box {
        background-color: #d1ecf1;
        border-left: 4px solid #0c5460;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        border-radius: 10px 10px 0 0;
        padding: 0 24px;
        font-weight: 600;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Cards */
    .element-container {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ==================== API CONFIGURATION ====================

API_URL = "http://localhost:8000"

# Initialize session state
if 'api_url' not in st.session_state:
    st.session_state.api_url = API_URL

if 'properties_data' not in st.session_state:
    st.session_state.properties_data = None

if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = None

# ==================== HELPER FUNCTIONS ====================

def fetch_data(endpoint, params=None):
    """Fetch data from API endpoint"""
    try:
        response = requests.get(
            f"{st.session_state.api_url}{endpoint}",
            params=params,
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: {e}")
        return None

def post_data(endpoint, data):
    """Post data to API endpoint"""
    try:
        response = requests.post(
            f"{st.session_state.api_url}{endpoint}",
            json=data,
            timeout=30
        )
        return response
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: {e}")
        return None

# ==================== SIDEBAR ====================

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/city.png", width=80)
    st.title("⚙️ Settings")
    
    # API Configuration
    st.subheader("🔌 API Configuration")
    api_url_input = st.text_input(
        "Backend URL",
        value=st.session_state.api_url,
        help="URL of your FastAPI backend"
    )
    
    if api_url_input != st.session_state.api_url:
        st.session_state.api_url = api_url_input
    
    # Test API Connection
    if st.button("🔍 Test Connection", use_container_width=True):
        with st.spinner("Testing connection..."):
            health = fetch_data("/health")
            if health:
                st.success(f"✅ Connected! Version: {health.get('version', 'unknown')}")
            else:
                st.error("❌ Cannot connect to API")
    
    st.divider()
    
    # Refresh Data
    if st.button("🔄 Refresh Data", use_container_width=True, type="primary"):
        st.session_state.properties_data = None
        st.session_state.last_refresh = datetime.now()
        st.rerun()
    
    if st.session_state.last_refresh:
        st.caption(f"Last refresh: {st.session_state.last_refresh.strftime('%H:%M:%S')}")
    
    st.divider()
    
    # Quick Stats
    st.subheader("📊 Quick Stats")
    stats = fetch_data("/api/stats")
    if stats:
        st.metric("Properties", stats.get('total_properties', 0))
        st.metric("Analyses", stats.get('total_analyses', 0))
        st.metric("Cities", stats.get('unique_cities', 0))
    
    st.divider()
    
    # Info
    st.info("""
    **💡 Quick Guide:**
    1. View properties in Browse tab
    2. Add new properties in Add Property tab
    3. Analyze neighborhoods in Analysis tab
    4. Ask AI questions in AI Assistant tab
    """)

# ==================== MAIN HEADER ====================

st.markdown('<h1 class="main-header">🏠 GeoInsight AI Platform</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Real Estate Intelligence & Neighborhood Analysis</p>', unsafe_allow_html=True)

# ==================== TABS ====================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏘️ Browse Properties",
    "➕ Add Property", 
    "🗺️ Neighborhood Analysis",
    "🤖 AI Assistant",
    "📈 Analytics Dashboard"
])

# ==================== TAB 1: BROWSE PROPERTIES ====================

with tab1:
    st.header("🏘️ Property Listings")
    
    # Fetch properties
    if st.session_state.properties_data is None:
        with st.spinner("Loading properties..."):
            st.session_state.properties_data = fetch_data("/api/properties")
    
    properties_data = st.session_state.properties_data
    
    if properties_data:
        df = pd.DataFrame(properties_data)
        
        # Top metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📍 Total Properties", len(df), delta=None)
        
        with col2:
            avg_price = df['price'].mean() if 'price' in df.columns else 0
            st.metric("💰 Avg Price", f"${avg_price:,.0f}", delta=None)
        
        with col3:
            avg_sqft = df['square_feet'].mean() if 'square_feet' in df.columns else 0
            st.metric("📏 Avg Size", f"{avg_sqft:,.0f} sq ft", delta=None)
        
        with col4:
            cities = df['city'].nunique() if 'city' in df.columns else 0
            st.metric("🌆 Cities", cities, delta=None)
        
        st.divider()
        
        # Filters
        st.subheader("🔍 Filter Properties")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'city' in df.columns:
                cities = ['All'] + sorted(df['city'].unique().tolist())
                selected_city = st.selectbox("City", cities)
            else:
                selected_city = 'All'
        
        with col2:
            if 'bedrooms' in df.columns:
                bedrooms = ['All'] + sorted(df['bedrooms'].dropna().unique().tolist())
                selected_beds = st.selectbox("Bedrooms", bedrooms)
            else:
                selected_beds = 'All'
        
        with col3:
            if 'property_type' in df.columns:
                prop_types = ['All'] + sorted(df['property_type'].dropna().unique().tolist())
                selected_type = st.selectbox("Type", prop_types)
            else:
                selected_type = 'All'
        
        # Apply filters
        filtered_df = df.copy()
        
        if selected_city != 'All':
            filtered_df = filtered_df[filtered_df['city'] == selected_city]
        
        if selected_beds != 'All':
            filtered_df = filtered_df[filtered_df['bedrooms'] == selected_beds]
        
        if selected_type != 'All':
            filtered_df = filtered_df[filtered_df['property_type'] == selected_type]
        
        # Price range slider
        if 'price' in filtered_df.columns and not filtered_df.empty:
            min_price = int(filtered_df['price'].min())
            max_price = int(filtered_df['price'].max())
            
            if min_price < max_price:
                price_range = st.slider(
                    "💵 Price Range",
                    min_value=min_price,
                    max_value=max_price,
                    value=(min_price, max_price),
                    format="$%d"
                )
                filtered_df = filtered_df[
                    (filtered_df['price'] >= price_range[0]) &
                    (filtered_df['price'] <= price_range[1])
                ]
        
        st.divider()
        
        # Display properties
        if not filtered_df.empty:
            st.success(f"✨ Showing {len(filtered_df)} of {len(df)} properties")
            
            # Property cards
            for idx, row in filtered_df.iterrows():
                with st.container():
                    col1, col2, col3 = st.columns([2, 2, 1])
                    
                    with col1:
                        st.markdown(f"### 🏠 {row.get('address', 'N/A')}")
                        st.caption(f"📍 {row.get('city', 'N/A')}, {row.get('state', 'N/A')}")
                    
                    with col2:
                        st.markdown(f"**💰 Price:** ${row.get('price', 0):,.0f}")
                        st.markdown(f"**🛏️ Beds:** {row.get('bedrooms', 'N/A')} | **🛁 Baths:** {row.get('bathrooms', 'N/A')}")
                        st.markdown(f"**📏 Size:** {row.get('square_feet', 'N/A'):,} sq ft")
                    
                    with col3:
                        if st.button("📋 Details", key=f"detail_{idx}", use_container_width=True):
                            st.session_state.selected_property = row
                            st.info(f"Property ID: {row.get('id', 'N/A')}")
                    
                    st.divider()
        else:
            st.warning("No properties match your filters. Try adjusting your criteria.")
    
    else:
        st.error("""
        ⚠️ **Cannot load properties**
        
        Please ensure:
        1. Backend API is running on http://localhost:8000
        2. MongoDB is running and connected
        3. Try clicking 'Test Connection' in the sidebar
        """)

# ==================== TAB 2: ADD PROPERTY ====================

with tab2:
    st.header("➕ Add New Property")
    
    st.markdown("""
    Fill in the property details below to add a new listing to the database.
    """)
    
    with st.form("add_property_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            address = st.text_input("🏠 Address *", placeholder="123 Main St")
            city = st.text_input("🌆 City *", placeholder="San Francisco")
            state = st.text_input("📍 State *", placeholder="CA")
            zip_code = st.text_input("📮 ZIP Code *", placeholder="94105")
        
        with col2:
            price = st.number_input("💰 Price ($) *", min_value=0, value=500000, step=10000)
            bedrooms = st.number_input("🛏️ Bedrooms *", min_value=0, value=3, step=1)
            bathrooms = st.number_input("🛁 Bathrooms *", min_value=0.0, value=2.0, step=0.5)
            square_feet = st.number_input("📏 Square Feet *", min_value=0, value=1500, step=100)
        
        col3, col4 = st.columns(2)
        
        with col3:
            property_type = st.selectbox(
                "🏘️ Property Type *",
                ["Single Family", "Condo", "Apartment", "Townhouse", "Multi-Family"]
            )
        
        with col4:
            st.write("📍 Coordinates (Optional)")
            latitude = st.number_input("Latitude", value=37.7749, format="%.6f")
            longitude = st.number_input("Longitude", value=-122.4194, format="%.6f")
        
        submitted = st.form_submit_button("✅ Add Property", use_container_width=True, type="primary")
        
        if submitted:
            if not all([address, city, state, zip_code]):
                st.error("❌ Please fill in all required fields (marked with *)")
            else:
                property_data = {
                    "address": address,
                    "city": city,
                    "state": state,
                    "zip_code": zip_code,
                    "price": price,
                    "bedrooms": int(bedrooms),
                    "bathrooms": float(bathrooms),
                    "square_feet": int(square_feet),
                    "property_type": property_type,
                    "latitude": latitude,
                    "longitude": longitude
                }
                
                with st.spinner("Adding property..."):
                    response = post_data("/api/properties", property_data)
                    
                    if response and response.status_code == 201:
                        result = response.json()
                        st.success(f"""
                        ✅ **Property added successfully!**
                        
                        - **ID:** {result.get('id')}
                        - **Address:** {result.get('address')}
                        - **Price:** ${result.get('price'):,}
                        """)
                        
                        # Clear cached data to show new property
                        st.session_state.properties_data = None
                        
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error("❌ Failed to add property. Check API connection.")

# ==================== TAB 3: NEIGHBORHOOD ANALYSIS ====================

with tab3:
    st.header("🗺️ Neighborhood Analysis")
    
    st.markdown("""
    Analyze any location to discover nearby amenities, calculate walkability scores, 
    and visualize the neighborhood on an interactive map.
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        analysis_address = st.text_input(
            "📍 Enter Address",
            placeholder="e.g., Manipal, Karnataka, India",
            help="Enter a complete address for best results"
        )
    
    with col2:
        radius = st.slider(
            "🔍 Search Radius (meters)",
            min_value=500,
            max_value=3000,
            value=1000,
            step=100
        )
    
    # Amenity selection
    st.subheader("🎯 Select Amenities to Find")
    
    col1, col2, col3, col4 = st.columns(4)
    
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
    
    selected_amenities = []
    
    with col1:
        if st.checkbox("🍽️ Restaurants", value=True):
            selected_amenities.append("restaurant")
        if st.checkbox("☕ Cafes", value=True):
            selected_amenities.append("cafe")
    
    with col2:
        if st.checkbox("🏫 Schools", value=True):
            selected_amenities.append("school")
        if st.checkbox("🏥 Hospitals", value=False):
            selected_amenities.append("hospital")
    
    with col3:
        if st.checkbox("🌳 Parks", value=False):
            selected_amenities.append("park")
        if st.checkbox("🛒 Supermarkets", value=False):
            selected_amenities.append("supermarket")
    
    with col4:
        if st.checkbox("🏦 Banks", value=False):
            selected_amenities.append("bank")
        if st.checkbox("💊 Pharmacies", value=False):
            selected_amenities.append("pharmacy")
    
    col1, col2 = st.columns(2)
    
    with col1:
        generate_map = st.checkbox("🗺️ Generate Interactive Map", value=True)
    
    with col2:
        include_buildings = st.checkbox("🏢 Include Building Data", value=True)
    
    if st.button("🚀 Start Analysis", use_container_width=True, type="primary"):
        if not analysis_address:
            st.error("❌ Please enter an address")
        elif not selected_amenities:
            st.error("❌ Please select at least one amenity type")
        else:
            with st.spinner("🔍 Analyzing neighborhood... This may take 10-30 seconds"):
                analysis_data = {
                    "address": analysis_address,
                    "radius_m": radius,
                    "amenity_types": selected_amenities,
                    "include_buildings": include_buildings,
                    "generate_map": generate_map
                }
                
                response = post_data("/api/neighborhood/analyze", analysis_data)
                
                if response and response.status_code == 202:
                    result = response.json()
                    
                    # Display results
                    st.success("✅ Analysis Complete!")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        walk_score = result.get('walk_score', 0)
                        st.metric(
                            "🚶 Walk Score",
                            f"{walk_score:.1f}/100",
                            delta="Good" if walk_score > 70 else "Fair" if walk_score > 50 else "Low"
                        )
                    
                    with col2:
                        st.metric("📍 Total Amenities", result.get('total_amenities', 0))
                    
                    with col3:
                        st.metric("📊 Analysis Status", result.get('status', 'unknown'))
                    
                    # Show amenities breakdown
                    st.subheader("🎯 Amenities Found")
                    
                    amenities = result.get('amenities', {})
                    if amenities:
                        cols = st.columns(4)
                        idx = 0
                        
                        for amenity_type, items in amenities.items():
                            with cols[idx % 4]:
                                st.markdown(f"""
                                **{amenity_type.title()}**  
                                🔢 {len(items)} found
                                """)
                                
                                if items:
                                    with st.expander("View Details"):
                                        for item in items[:5]:
                                            st.write(f"• {item.get('name', 'Unknown')} ({item.get('distance_km', 0):.2f} km)")
                            
                            idx += 1
                    
                    # Map link
                    if result.get('map_url'):
                        st.divider()
                        map_url = f"{st.session_state.api_url}{result['map_url']}"
                        st.markdown(f"""
                        ### 🗺️ Interactive Map
                        
                        Your interactive neighborhood map is ready!
                        
                        [🔗 Open Interactive Map]({map_url})
                        """)
                        
                        # Embed map in iframe
                        st.components.v1.iframe(map_url, height=600, scrolling=True)
                    
                    # Save analysis ID for later reference
                    st.session_state.last_analysis_id = result.get('analysis_id')
                    
                else:
                    st.error("❌ Analysis failed. Please check your API connection and try again.")
    
    # Recent analyses
    st.divider()
    st.subheader("📜 Recent Analyses")
    
    recent_analyses = fetch_data("/api/neighborhood/recent", params={"limit": 5})
    
    if recent_analyses:
        for analysis in recent_analyses:
            with st.expander(f"📍 {analysis.get('address', 'Unknown')} - {analysis.get('created_at', '')[:10]}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**Walk Score:** {analysis.get('walk_score', 'N/A')}")
                
                with col2:
                    st.write(f"**Amenities:** {analysis.get('total_amenities', 0)}")
                
                with col3:
                    st.write(f"**Status:** {analysis.get('status', 'unknown')}")
                
                if analysis.get('map_url'):
                    map_link = f"{st.session_state.api_url}{analysis['map_url']}"
                    st.markdown(f"[🗺️ View Map]({map_link})")

# ==================== TAB 4: AI ASSISTANT ====================

with tab4:
    st.header("🤖 AI Real Estate Assistant")
    
    st.markdown("""
    Ask me anything about real estate investments, property valuations, rental analysis, or market trends!
    I can help you make data-driven decisions.
    """)
    
    # Example queries
    with st.expander("💡 Example Questions"):
        st.markdown("""
        - Calculate ROI for a $300,000 property with $2,000 monthly rent
        - Is $450,000 a good price for a 3-bedroom house?
        - What's the expected rental income for a $500k property?
        - Investment analysis: $750k house, $3500 rent, 20% down payment
        - Should I invest in a property with 4% rental yield?
        """)
    
    # Query input
    user_query = st.text_area(
        "💬 Your Question",
        placeholder="e.g., Calculate investment returns for $300k property with $2000 monthly rent",
        height=100
    )
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        ask_button = st.button("🚀 Ask AI Assistant", use_container_width=True, type="primary")
    
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.rerun()
    
    if ask_button and user_query:
        with st.spinner("🤔 AI is thinking..."):
            response = post_data("/api/agent/query", {"query": user_query})
            
            if response and response.status_code == 200:
                result = response.json()
                
                st.success("✅ Analysis Complete!")
                
                # Display answer
                st.markdown("### 💡 AI Response")
                
                answer = result.get('answer', 'No response available')
                st.markdown(answer)
                
                # Show calculations if available
                if 'calculations' in result:
                    st.divider()
                    st.markdown("### 📊 Detailed Calculations")
                    
                    calc = result['calculations']
                    
                    cols = st.columns(len(calc))
                    for idx, (key, value) in enumerate(calc.items()):
                        with cols[idx]:
                            st.metric(
                                key.replace('_', ' ').title(),
                                f"{value:,.2f}" if isinstance(value, (int, float)) else str(value)
                            )
                
                # Show confidence
                if 'confidence' in result:
                    confidence = result['confidence']
                    st.progress(confidence)
                    st.caption(f"Confidence: {confidence*100:.0f}%")
                
            else:
                st.error("❌ Failed to get response from AI assistant")

# ==================== TAB 5: ANALYTICS ====================

with tab5:
    st.header("📈 Analytics Dashboard")
    
    if st.session_state.properties_data:
        df = pd.DataFrame(st.session_state.properties_data)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Total Properties", len(df))
        
        with col2:
            if 'price' in df.columns:
                st.metric("💰 Total Value", f"${df['price'].sum():,.0f}")
        
        with col3:
            if 'price' in df.columns:
                st.metric("📈 Avg Price", f"${df['price'].mean():,.0f}")
        
        with col4:
            if 'price' in df.columns:
                st.metric("📉 Median Price", f"${df['price'].median():,.0f}")
        
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
                    color_discrete_sequence=['#1E88E5']
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'city' in df.columns and len(df['city'].unique()) > 1:
                st.subheader("🌆 Properties by City")
                city_counts = df['city'].value_counts()
                fig = px.pie(
                    values=city_counts.values,
                    names=city_counts.index,
                    title="Property Distribution by City"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # More charts
        col1, col2 = st.columns(2)
        
        with col1:
            if 'bedrooms' in df.columns:
                st.subheader("🛏️ Bedrooms Distribution")
                bed_counts = df['bedrooms'].value_counts().sort_index()
                fig = px.bar(
                    x=bed_counts.index,
                    y=bed_counts.values,
                    labels={'x': 'Bedrooms', 'y': 'Count'},
                    color=bed_counts.values,
                    color_continuous_scale='blues'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'square_feet' in df.columns and 'price' in df.columns:
                st.subheader("📏 Price per Sq Ft")
                df['price_per_sqft'] = df['price'] / df['square_feet']
                fig = px.scatter(
                    df,
                    x='square_feet',
                    y='price',
                    size='price_per_sqft',
                    color='city' if 'city' in df.columns else None,
                    hover_data=['address'],
                    title="Property Size vs Price"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("📊 Load property data in the Browse Properties tab to view analytics")

# ==================== FOOTER ====================

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.caption(f"🔌 API: {st.session_state.api_url}")

with col2:
    st.caption(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

with col3:
    st.caption("🚀 GeoInsight AI v4.0")