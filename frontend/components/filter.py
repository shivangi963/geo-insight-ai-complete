"""
Filter Components
Reusable filter widgets
"""
import streamlit as st
import pandas as pd
from typing import List, Optional, Dict, Any

def render_city_filter(df: pd.DataFrame, key: str = "city_filter") -> Optional[str]:
    """Render city dropdown filter"""
    if 'city' not in df.columns:
        return None
    
    cities = ['All'] + sorted(df['city'].dropna().unique().tolist())
    selected = st.selectbox("🏙️ City", cities, key=key)
    
    return None if selected == 'All' else selected

def render_property_type_filter(df: pd.DataFrame, key: str = "type_filter") -> Optional[str]:
    """Render property type dropdown filter"""
    if 'property_type' not in df.columns:
        return None
    
    types = ['All'] + sorted(df['property_type'].dropna().unique().tolist())
    selected = st.selectbox("🏠 Property Type", types, key=key)
    
    return None if selected == 'All' else selected

def render_bedroom_filter(df: pd.DataFrame, key: str = "bedroom_filter") -> Optional[int]:
    """Render bedroom dropdown filter"""
    if 'bedrooms' not in df.columns:
        return None
    
    bedrooms = ['All'] + sorted(df['bedrooms'].dropna().unique().tolist())
    selected = st.selectbox("🛏️ Bedrooms", bedrooms, key=key)
    
    return None if selected == 'All' else int(selected)

def render_price_range_filter(df: pd.DataFrame, key: str = "price_range") -> Optional[tuple]:
    """Render price range slider"""
    if 'price' not in df.columns or len(df) == 0:
        return None
    
    min_price = int(df['price'].min())
    max_price = int(df['price'].max())
    
    if min_price >= max_price:
        return None
    
    selected = st.slider(
        "💵 Price Range",
        min_price, max_price,
        (min_price, max_price),
        key=key
    )
    
    return selected

def render_size_range_filter(df: pd.DataFrame, key: str = "size_range") -> Optional[tuple]:
    """Render size range slider"""
    if 'square_feet' not in df.columns or len(df) == 0:
        return None
    
    min_size = int(df['square_feet'].min())
    max_size = int(df['square_feet'].max())
    
    if min_size >= max_size:
        return None
    
    selected = st.slider(
        "📐 Size Range (sqft)",
        min_size, max_size,
        (min_size, max_size),
        key=key
    )
    
    return selected

def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """Apply multiple filters to DataFrame"""
    filtered = df.copy()
    
    # City filter
    if filters.get('city'):
        filtered = filtered[filtered['city'] == filters['city']]
    
    # Property type filter
    if filters.get('property_type'):
        filtered = filtered[filtered['property_type'] == filters['property_type']]
    
    # Bedroom filter
    if filters.get('bedrooms') is not None:
        filtered = filtered[filtered['bedrooms'] == filters['bedrooms']]
    
    # Price range filter
    if filters.get('price_range'):
        min_p, max_p = filters['price_range']
        filtered = filtered[
            (filtered['price'] >= min_p) &
            (filtered['price'] <= max_p)
        ]
    
    # Size range filter
    if filters.get('size_range'):
        min_s, max_s = filters['size_range']
        filtered = filtered[
            (filtered['square_feet'] >= min_s) &
            (filtered['square_feet'] <= max_s)
        ]
    
    return filtered

def render_filter_summary(original_count: int, filtered_count: int):
    """Display filter results summary"""
    if filtered_count == original_count:
        st.info(f"📊 Showing all {original_count} properties")
    else:
        st.success(f"📊 Showing {filtered_count} of {original_count} properties")
        
        if filtered_count == 0:
            st.warning("⚠️ No properties match the current filters. Try adjusting your criteria.")

def render_reset_filters_button(key: str = "reset_filters") -> bool:
    """Render reset filters button"""
    return st.button("🔄 Reset Filters", key=key, use_container_width=True)