import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime
import numpy as np

st.set_page_config(
    page_title=" GeoInsight AI - Real Estate Dashboard",
    page_icon="",
    layout="wide"
)


st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header"> GeoInsight AI - Real Estate Dashboard</h1>', unsafe_allow_html=True)


with st.sidebar:
    st.header(" Settings")
    
    
    st.subheader("API Configuration")
    api_url = st.text_input(
        "FastAPI URL",
        value="http://localhost:8000",
        help="URL of your FastAPI backend"
    )
    
   
    if st.button(" Refresh Data", use_container_width=True):
        st.rerun()
    
    st.divider()
    

    st.info("""
    **Instructions:**
    1. Make sure your FastAPI backend is running
    2. Update API URL if needed
    3. Click refresh to load latest data
    """)

tab1, tab2, tab3 = st.tabs(["Properties", " Analytics", " Map View"])


def fetch_properties(api_url):
    """Fetch properties from FastAPI backend"""
    try:
        response = requests.get(f"{api_url}/api/properties", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: {e}")
        return None


properties_data = fetch_properties(api_url)

with tab1:
    st.header("Property Listings")
    
    if properties_data:

        df = pd.DataFrame(properties_data)
        

        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Properties", len(df))
        
        with col2:
            avg_price = df['price'].mean() if 'price' in df.columns else 0
            st.metric("Average Price", f"${avg_price:,.0f}")
        
        with col3:
            avg_size = df['square_feet'].mean() if 'square_feet' in df.columns else 0
            st.metric("Avg Size (sq ft)", f"{avg_size:,.0f}")
        
        with col4:

            unique_cities = df['city'].nunique() if 'city' in df.columns else 0
            st.metric("Cities", unique_cities)
        
   
        st.subheader(" Filter Properties")
        col1, col2 = st.columns(2)
        
        filtered_df = df.copy() 
        
        with col1:
            if 'city' in df.columns:
                cities = ['All'] + sorted(df['city'].unique().tolist())
                selected_city = st.selectbox("City", cities)
                
                if selected_city != 'All':
                    filtered_df = filtered_df[filtered_df['city'] == selected_city]
        
        with col2:
            if 'price' in df.columns and not filtered_df.empty:
                min_price = int(filtered_df['price'].min())
                max_price = int(filtered_df['price'].max())
                
          
                if min_price == max_price:
        
                    min_price = max(0, min_price - 100000)
                    max_price = max_price + 100000
                
           
                try:
                    price_range = st.slider(
                        "Price Range",
                        min_value=min_price,
                        max_value=max_price,
                        value=(min_price, max_price)
                    )
                    filtered_df = filtered_df[(filtered_df['price'] >= price_range[0]) & (filtered_df['price'] <= price_range[1])]
                except Exception as e:
                    st.warning(f"Could not create price slider: {e}")
    
                    st.info(f"All properties are priced at: ${min_price:,.0f}")
        

        if not filtered_df.empty:
            st.write(f"Showing {len(filtered_df)} of {len(df)} properties")
            

            display_df = filtered_df.copy()

            if 'price' in display_df.columns:
                display_df['price'] = display_df['price'].apply(lambda x: f"${x:,.0f}")
            
            if 'square_feet' in display_df.columns:
                display_df['square_feet'] = display_df['square_feet'].apply(lambda x: f"{x:,} sq ft")
            

            columns_to_show = ['address', 'city', 'state', 'price', 'bedrooms', 'bathrooms', 'square_feet']
            columns_to_show = [col for col in columns_to_show if col in display_df.columns]
            
            st.dataframe(
                display_df[columns_to_show],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No properties match the selected filters.")
            

            if not df.empty:
                with st.expander("View all properties"):
                    st.dataframe(df, use_container_width=True)
    else:
        st.warning("No data available. Please check your API connection.")
        st.info("""
        **Troubleshooting:**
        1. Make sure your FastAPI server is running: `uvicorn app.main:app --reload --port 8000`
        2. Check the API URL in the sidebar
        3. Test the API directly: `curl http://localhost:8000/api/properties`
        """)


with tab2:
    st.header("Property Analytics")
    
    if properties_data and len(properties_data) > 0:
        df = pd.DataFrame(properties_data)
        
        if not df.empty:
            st.subheader("Data Overview")
            
     
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'price' in df.columns:
                    st.metric("Min Price", f"${df['price'].min():,.0f}")
            
            with col2:
                if 'price' in df.columns:
                    st.metric("Max Price", f"${df['price'].max():,.0f}")
            
            with col3:
                if 'price' in df.columns:
                    st.metric("Median Price", f"${df['price'].median():,.0f}")
            
    
            col1, col2 = st.columns(2)
            
            with col1:
         
                if 'price' in df.columns and len(df) > 1:
                    st.subheader("Price Distribution")
                    fig1 = px.histogram(
                        df,
                        x='price',
                        nbins=min(20, len(df)),
                        title="Property Price Distribution",
                        labels={'price': 'Price ($)'}
                    )
                    fig1.update_layout(
                        showlegend=False,
                        xaxis_title="Price",
                        yaxis_title="Count"
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                elif 'price' in df.columns:
                    st.info(f"Single property price: ${df['price'].iloc[0]:,.0f}")
            
            with col2:
      
                if 'property_type' in df.columns:
                    st.subheader("Property Types")
                    type_counts = df['property_type'].value_counts()
                    if not type_counts.empty:
                        fig2 = px.pie(
                            values=type_counts.values,
                            names=type_counts.index,
                            title="Property Type Distribution"
                        )
                        st.plotly_chart(fig2, use_container_width=True)
                    else:
                        st.info("No property type data available")
            

            if 'city' in df.columns and len(df['city'].unique()) > 1:
                st.subheader("Location Analysis")
                
                city_stats = df.groupby('city').agg({
                    'price': ['mean', 'count'],
                    'square_feet': 'mean'
                }).round(0)
                
          
                city_stats.columns = ['avg_price', 'property_count', 'avg_sqft']
                city_stats = city_stats.reset_index()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig3 = px.bar(
                        city_stats,
                        x='city',
                        y='avg_price',
                        title="Average Price by City",
                        text='avg_price'
                    )
                    fig3.update_traces(texttemplate='$%{text:,}')
                    fig3.update_layout(
                        xaxis_title="City",
                        yaxis_title="Average Price ($)"
                    )
                    st.plotly_chart(fig3, use_container_width=True)
                
                with col2:
                    fig4 = px.bar(
                        city_stats,
                        x='city',
                        y='property_count',
                        title="Number of Properties by City",
                        text='property_count'
                    )
                    fig4.update_layout(
                        xaxis_title="City",
                        yaxis_title="Property Count"
                    )
                    st.plotly_chart(fig4, use_container_width=True)
            elif 'city' in df.columns:
                st.info(f"All properties are in: {df['city'].iloc[0]}")
        else:
            st.info("No data available for analytics.")
    else:
        st.info("Load data in the Properties tab first.")

with tab3:
    st.header(" Property Map")
    
    if properties_data and len(properties_data) > 0:
        df = pd.DataFrame(properties_data)
        
        if not df.empty:
    
            has_coords = all(col in df.columns for col in ['latitude', 'longitude'])
            
            if has_coords and not df[['latitude', 'longitude']].isnull().all().all():
           
                try:
                  
                    center_lat = df['latitude'].median() if not df['latitude'].isnull().all() else 39.8283
                    center_lon = df['longitude'].median() if not df['longitude'].isnull().all() else -98.5795
                    
                    fig = px.scatter_mapbox(
                        df.dropna(subset=['latitude', 'longitude']),
                        lat="latitude",
                        lon="longitude",
                        hover_name="address",
                        hover_data=["city", "state", "price", "square_feet", "bedrooms"],
                        color="price" if 'price' in df.columns else None,
                        color_continuous_scale=px.colors.cyclical.IceFire,
                        size_max=15,
                        zoom=3,
                        title="Property Locations",
                        height=600,
                        center={"lat": center_lat, "lon": center_lon}
                    )
                    
                    fig.update_layout(
                        mapbox_style="open-street-map",
                        margin={"r":0,"t":30,"l":0,"b":0}
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                  
                    with st.expander("Map Data Summary"):
                        st.write(f"Properties on map: {len(df.dropna(subset=['latitude', 'longitude']))}")
                        if 'price' in df.columns:
                            st.write(f"Price range: ${df['price'].min():,.0f} - ${df['price'].max():,.0f}")
                        if 'city' in df.columns:
                            cities = df['city'].unique()
                            if len(cities) <= 5:
                                st.write(f"Cities: {', '.join(cities)}")
                            else:
                                st.write(f"{len(cities)} different cities")
                except Exception as e:
                    st.error(f"Error creating map: {e}")
                    st.info("Showing data in table format instead:")
                    st.dataframe(df[['address', 'city', 'state', 'price']].head(10))
            else:
                st.warning("Location data not available for mapping.")
                st.info("Properties in dataset:")
                
                cols_to_show = ['address', 'city', 'state']
                if 'price' in df.columns:
                    cols_to_show.append('price')
                if 'square_feet' in df.columns:
                    cols_to_show.append('square_feet')
                
                st.dataframe(df[cols_to_show].head(10))
        else:
            st.info("No properties to display on map.")
    else:
        st.info("Load data in the Properties tab first.")

st.divider()
st.header("AI Property Analyst")

query = st.text_input(
    "Ask the AI agent about properties or investments:",
    placeholder="E.g., 'Which properties have the best value?' or 'Analyze investment potential'"
)

if query and st.button("Get AI Analysis", type="primary"):
    with st.spinner("Analyzing with AI..."):
        try:
            response = requests.post(
                f"{api_url}/api/agent/query",
                json={"query": query},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                st.success(" AI Analysis Complete")
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown("### Insights")
                    st.write(result.get('answer', 'No analysis provided'))
                
                with col2:
                    st.markdown("### Details")
                    if 'confidence' in result:
                        st.metric("Confidence", f"{result['confidence']:.0%}")
                    if 'timestamp' in result:
                        st.caption(f"Generated: {result['timestamp']}")
                
            
                if 'suggestions' in result and result['suggestions']:
                    with st.expander(" Recommendations"):
                        for suggestion in result['suggestions']:
                            st.write(f"• {suggestion}")
                
            else:
                st.error(f"Agent error: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            st.error(f"Connection error: {e}")
            st.info("Make sure your FastAPI server is running and the agent endpoint is working.")

st.divider()
st.caption(f" API: {api_url} |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.caption("GeoInsight AI Dashboard v1.0 | Phase 3: AI-Powered Tools & UIs")