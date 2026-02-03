"""
Properties Page
Browse and manage properties
"""
import streamlit as st
import pandas as pd
from api_client import api
from utils import (
    format_currency, format_number, calculate_price_per_sqft,
    show_success_message, show_error_message, init_session_state
)
from components.header import render_section_header
import time

def safe_filter_properties(properties, city_filter, type_filter, bedrooms_filter):
    """
    Safely filter properties with proper type handling
    
    Args:
        properties: List of property dictionaries
        city_filter: Selected city filter value
        type_filter: Selected property type filter value
        bedrooms_filter: Selected bedrooms filter value
    
    Returns:
        Filtered list of properties
    """
    filtered = properties
    
    # City filter - handle case-insensitive comparison
    if city_filter != "All" and city_filter:
        filtered = [
            p for p in filtered
            if str(p.get('city', '')).strip().lower() == str(city_filter).strip().lower()
        ]
    
    # Property type filter
    if type_filter != "All" and type_filter:
        filtered = [
            p for p in filtered
            if str(p.get('property_type', '')).strip().lower() == str(type_filter).strip().lower()
        ]
    
    # Bedrooms filter - handle both numeric and string comparisons
    if bedrooms_filter != "All" and bedrooms_filter:
        # Try to convert to int for numeric comparison
        try:
            bed_value = int(bedrooms_filter)
            filtered = [
                p for p in filtered
                if int(p.get('bedrooms', 0)) == bed_value
            ]
        except (ValueError, TypeError):
            # Fall back to string comparison
            filtered = [
                p for p in filtered
                if str(p.get('bedrooms', '')).strip() == str(bedrooms_filter).strip()
            ]
    
    return filtered



def render_properties_page():
    """Main properties page renderer"""
    render_section_header("Property Management", "🏘️")
    
    st.markdown("Browse and manage your real estate portfolio")
    
    # Tabs for different views
    browse_tab, add_tab = st.tabs(["📋 Browse Properties", "➕ Add Property"])
    
    with browse_tab:
        render_browse_properties()
    
    with add_tab:
        render_add_property()

def render_browse_properties():
    """Render properties browser"""
    # Refresh button
    col1, col2 = st.columns([3, 1])
    
    with col2:
        if st.button("🔄 Refresh", use_container_width=True, key="refresh_props"):
            st.cache_data.clear()
            st.rerun()
    
    # Load properties
    with st.spinner("Loading properties..."):
        properties = api.get_properties(limit=100)
    
    if not properties or len(properties) == 0:
        st.info("📭 No properties in database")
        render_no_properties_help()
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(properties)
    
    # Metrics Row
    render_property_metrics(df)
    
    st.divider()
    
    # Filters
    filtered_df = render_property_filters(df)
    
    st.success(f"📍 Showing {len(filtered_df)} properties")
    
    # Display properties
    render_property_list(filtered_df)

def render_property_metrics(df: pd.DataFrame):
    """Render property statistics"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total", len(df))
    
    with col2:
        if 'price' in df.columns:
            avg_price = df['price'].mean()
            st.metric("💰 Avg Price", format_currency(avg_price))
    
    with col3:
        if 'square_feet' in df.columns:
            avg_sqft = df['square_feet'].mean()
            st.metric("📐 Avg Size", f"{avg_sqft:,.0f} sqft")
    
    with col4:
        if 'city' in df.columns:
            unique_cities = df['city'].nunique()
            st.metric("🏙️ Cities", unique_cities)

def render_property_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Render filters and return filtered DataFrame - TYPE-SAFE VERSION"""
    col1, col2, col3 = st.columns(3)
    
    # Convert DataFrame to list of dicts for safe filtering
    properties = df.to_dict('records')
    
    # City filter
    with col1:
        if 'city' in df.columns:
            cities = list(set([str(p.get('city', 'Unknown')).strip() for p in properties if p.get('city')]))
            cities.sort()
            city_filter = st.selectbox("🏙️ City", ["All"] + cities, key="city_filter")
        else:
            city_filter = "All"
    
    # Property type filter
    with col2:
        if 'property_type' in df.columns:
            types = list(set([str(p.get('property_type', 'Unknown')).strip() for p in properties if p.get('property_type')]))
            types.sort()
            type_filter = st.selectbox("🏠 Type", ["All"] + types, key="type_filter")
        else:
            type_filter = "All"
    
    # Bedrooms filter
    with col3:
        if 'bedrooms' in df.columns:
            bedrooms = list(set([str(p.get('bedrooms', '0')).strip() for p in properties if p.get('bedrooms')]))
            bedrooms.sort(key=lambda x: int(x) if x.isdigit() else 0)
            bedrooms_filter = st.selectbox("🛏️ Bedrooms", ["All"] + bedrooms, key="bed_filter")
        else:
            bedrooms_filter = "All"
    
    # Apply safe filtering
    filtered_properties = safe_filter_properties(properties, city_filter, type_filter, bedrooms_filter)
    
    # Convert back to DataFrame
    filtered_df = pd.DataFrame(filtered_properties) if filtered_properties else pd.DataFrame()
    
    # Price range slider
    if 'price' in df.columns and len(df) > 0 and len(filtered_df) > 0:
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
    
    return filtered_df


def render_property_list(df: pd.DataFrame):
    """Render list of properties"""
    for idx, row in df.iterrows():
        render_property_card(row, idx)

def render_property_card(row: pd.Series, idx: int):
    """Render individual property card"""
    price = row.get('price', 0)
    address = row.get('address', 'N/A')
    
    with st.expander(f"🏠 {address} | {format_currency(price)}", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📍 Location**")
            st.write(f"{row.get('city', 'N/A')}, {row.get('state', 'N/A')}")
            st.write(f"ZIP: {row.get('zip_code', 'N/A')}")
        
        with col2:
            st.markdown("**🏘️ Details**")
            st.write(f"Type: {row.get('property_type', 'N/A')}")
            beds = row.get('bedrooms', 'N/A')
            baths = row.get('bathrooms', 'N/A')
            st.write(f"Beds: {beds} | Baths: {baths}")
        
        with col3:
            st.markdown("**📊 Metrics**")
            sqft = row.get('square_feet', 0)
            st.write(f"Size: {sqft:,} sqft")
            price_per_sqft = calculate_price_per_sqft(price, sqft)
            st.write(f"$/sqft: {format_currency(price_per_sqft)}")
        
        # Action buttons
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🗺️ Analyze", key=f"analyze_{idx}"):
                st.session_state.nav_to_analysis = address
                show_success_message("Switched to Analysis tab")
        
        with col2:
            if st.button("🤖 AI Analysis", key=f"ai_{idx}"):
                query = f"Investment analysis for ${price:,.0f} property at {address}"
                st.session_state.ai_query = query
                show_success_message("Switched to AI tab")

def render_add_property():
    """Render add property form"""
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
            placeholder="Beautiful property with...",
            height=100
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            latitude = st.number_input("🌐 Latitude", value=0.0, format="%.6f")
        
        with col2:
            longitude = st.number_input("🌐 Longitude", value=0.0, format="%.6f")
        
        submitted = st.form_submit_button(
            "➕ Add Property", 
            type="primary", 
            use_container_width=True
        )
        
        if submitted:
            handle_property_submission(
                address, city, state, zip_code, price, bedrooms,
                bathrooms, square_feet, property_type, description,
                latitude, longitude
            )

def handle_property_submission(address, city, state, zip_code, price, bedrooms,
                               bathrooms, square_feet, property_type, description,
                               latitude, longitude):
    """Handle property form submission"""
    if not all([address, city, state, zip_code]):
        show_error_message("Please fill all required fields (*)")
        return
    
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
        
        result = api.create_property(data)
        
        if result:
            show_success_message(f"Property added! ID: {result.get('id')}")
            st.balloons()
            time.sleep(2)
            st.rerun()

def render_no_properties_help():
    """Show help when no properties exist"""
    with st.expander("💡 How to Add Properties"):
        st.markdown("""
        **Option 1: Use the 'Add Property' tab above**
        
        **Option 2: Load from Kaggle CSV**
        ```bash
        cd backend
        python load_kaggle_data.py data/your_dataset.csv
        ```
        
        **Option 3: Use the API directly**
        ```bash
        curl -X POST "http://localhost:8000/api/properties" \\
          -H "Content-Type: application/json" \\
          -d '{"address": "...", ...}'
        ```
        """)