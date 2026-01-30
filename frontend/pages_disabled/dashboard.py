"""
Dashboard Page
Analytics and visualizations
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from api_client import api
from utils import format_currency, format_number
from components.header import render_section_header

def render_dashboard_page():
    """Main dashboard page"""
    render_section_header("Analytics Dashboard", "📊")
    
    # Load data
    properties = api.get_properties(limit=1000)
    stats = api.get_stats()
    
    if not properties or len(properties) == 0:
        render_no_data_message()
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(properties)
    
    # Top metrics
    render_top_metrics(df, stats)
    
    st.divider()
    
    # Charts
    render_dashboard_charts(df)

def render_no_data_message():
    """Show message when no data available"""
    st.info("📭 No data available for analytics")
    
    st.markdown("""
    ### Get Started
    
    1. Add properties via the **Properties** tab
    2. Or load data from CSV:
       ```bash
       cd backend
       python load_kaggle_data.py data/your_dataset.csv
       ```
    3. Return here to see analytics!
    """)

def render_top_metrics(df: pd.DataFrame, stats: dict):
    """Render top-level metrics"""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("🏠 Total Properties", format_number(len(df)))
    
    with col2:
        if 'price' in df.columns:
            total_value = df['price'].sum()
            st.metric("💰 Total Value", format_currency(total_value))
    
    with col3:
        if 'price' in df.columns:
            avg_price = df['price'].mean()
            st.metric("📊 Avg Price", format_currency(avg_price))
    
    with col4:
        if 'city' in df.columns:
            unique_cities = df['city'].nunique()
            st.metric("🏙️ Cities", format_number(unique_cities))
    
    with col5:
        if stats:
            total_analyses = stats.get('total_analyses', 0)
            st.metric("🗺️ Analyses", format_number(total_analyses))

def render_dashboard_charts(df: pd.DataFrame):
    """Render all dashboard charts"""
    # Row 1: Price Distribution & City Breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        render_price_distribution(df)
    
    with col2:
        render_city_breakdown(df)
    
    # Row 2: Property Types & Bedroom Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        render_property_types(df)
    
    with col2:
        render_bedroom_distribution(df)
    
    # Row 3: Price vs Size Scatter
    render_price_vs_size(df)
    
    # Row 4: Summary Statistics
    render_summary_statistics(df)

def render_price_distribution(df: pd.DataFrame):
    """Render price distribution histogram"""
    if 'price' not in df.columns:
        return
    
    st.subheader("💰 Price Distribution")
    
    fig = px.histogram(
        df, 
        x='price',
        nbins=30,
        title="Property Price Distribution",
        labels={'price': 'Price ($)', 'count': 'Number of Properties'},
        color_discrete_sequence=['#667eea']
    )
    
    fig.update_layout(
        showlegend=False,
        xaxis_title="Price ($)",
        yaxis_title="Count"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_city_breakdown(df: pd.DataFrame):
    """Render city breakdown pie chart"""
    if 'city' not in df.columns:
        return
    
    st.subheader("🌆 Distribution by City")
    
    city_counts = df['city'].value_counts()
    
    # Show top 10 cities
    if len(city_counts) > 10:
        top_cities = city_counts.head(10)
        other_count = city_counts[10:].sum()
        if other_count > 0:
            top_cities['Other'] = other_count
        city_counts = top_cities
    
    fig = px.pie(
        values=city_counts.values,
        names=city_counts.index,
        title="Properties by City (Top 10)"
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    
    st.plotly_chart(fig, use_container_width=True)

def render_property_types(df: pd.DataFrame):
    """Render property types bar chart"""
    if 'property_type' not in df.columns:
        return
    
    st.subheader("🏘️ Property Types")
    
    type_counts = df['property_type'].value_counts()
    
    fig = px.bar(
        x=type_counts.index,
        y=type_counts.values,
        labels={'x': 'Property Type', 'y': 'Count'},
        title="Properties by Type",
        color=type_counts.values,
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(showlegend=False)
    
    st.plotly_chart(fig, use_container_width=True)

def render_bedroom_distribution(df: pd.DataFrame):
    """Render bedroom distribution"""
    if 'bedrooms' not in df.columns:
        return
    
    st.subheader("🛏️ Bedroom Distribution")
    
    bed_counts = df['bedrooms'].value_counts().sort_index()
    
    fig = px.bar(
        x=bed_counts.index,
        y=bed_counts.values,
        labels={'x': 'Number of Bedrooms', 'y': 'Count'},
        title="Properties by Bedroom Count",
        color=bed_counts.values,
        color_continuous_scale='blues'
    )
    
    fig.update_layout(showlegend=False)
    
    st.plotly_chart(fig, use_container_width=True)

def render_price_vs_size(df: pd.DataFrame):
    """Render price vs square feet scatter plot"""
    if 'price' not in df.columns or 'square_feet' not in df.columns:
        return
    
    st.divider()
    st.subheader("📐 Price vs Square Feet")
    
    # Add price per sqft column
    df_copy = df.copy()
    df_copy['price_per_sqft'] = df_copy['price'] / df_copy['square_feet']
    
    # Create scatter plot
    fig = px.scatter(
        df_copy,
        x='square_feet',
        y='price',
        color='city' if 'city' in df_copy.columns else None,
        size='bedrooms' if 'bedrooms' in df_copy.columns else None,
        hover_data=['address'] if 'address' in df_copy.columns else None,
        title="Price vs Size Analysis",
        labels={'square_feet': 'Square Feet', 'price': 'Price ($)'}
    )
    
    # Add trendline
    fig.add_trace(
        go.Scatter(
            x=df_copy['square_feet'],
            y=df_copy['square_feet'] * df_copy['price_per_sqft'].median(),
            mode='lines',
            name='Median $/sqft',
            line=dict(color='red', dash='dash')
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Additional insights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        median_price_sqft = df_copy['price_per_sqft'].median()
        st.metric("Median $/sqft", format_currency(median_price_sqft))
    
    with col2:
        avg_size = df_copy['square_feet'].mean()
        st.metric("Avg Size", f"{avg_size:,.0f} sqft")
    
    with col3:
        # Find most efficient (lowest $/sqft in top 25% by size)
        large_props = df_copy[df_copy['square_feet'] >= df_copy['square_feet'].quantile(0.75)]
        if len(large_props) > 0:
            best_value = large_props['price_per_sqft'].min()
            st.metric("Best Value (Large)", format_currency(best_value))

def render_summary_statistics(df: pd.DataFrame):
    """Render summary statistics table"""
    st.divider()
    st.subheader("📋 Summary Statistics")
    
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    if len(numeric_cols) == 0:
        st.info("No numeric data available")
        return
    
    # Generate summary
    summary = df[numeric_cols].describe()
    
    # Format currency columns
    currency_cols = ['price']
    for col in currency_cols:
        if col in summary.columns:
            summary[col] = summary[col].apply(lambda x: f"${x:,.0f}")
    
    # Format number columns
    number_cols = ['square_feet', 'bedrooms', 'bathrooms']
    for col in number_cols:
        if col in summary.columns:
            summary[col] = summary[col].apply(lambda x: f"{x:,.1f}")
    
    st.dataframe(summary, use_container_width=True)
    
    # Additional insights
    render_additional_insights(df)

def render_additional_insights(df: pd.DataFrame):
    """Render additional insights"""
    st.divider()
    st.subheader("💡 Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏆 Top 5 Most Expensive")
        if 'price' in df.columns and 'address' in df.columns:
            top_expensive = df.nlargest(5, 'price')[['address', 'city', 'price']]
            for idx, row in top_expensive.iterrows():
                st.write(f"**{row['address']}** - {format_currency(row['price'])}")
                st.caption(f"📍 {row.get('city', 'N/A')}")
        else:
            st.info("Price data not available")
    
    with col2:
        st.markdown("### 💎 Top 5 Best Value ($/sqft)")
        if 'price' in df.columns and 'square_feet' in df.columns and 'address' in df.columns:
            df_copy = df.copy()
            df_copy['price_per_sqft'] = df_copy['price'] / df_copy['square_feet']
            top_value = df_copy.nsmallest(5, 'price_per_sqft')[['address', 'city', 'price_per_sqft']]
            for idx, row in top_value.iterrows():
                st.write(f"**{row['address']}** - {format_currency(row['price_per_sqft'])}/sqft")
                st.caption(f"📍 {row.get('city', 'N/A')}")
        else:
            st.info("Size data not available")