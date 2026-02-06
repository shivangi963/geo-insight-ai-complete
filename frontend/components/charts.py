"""
Chart Components
Reusable chart/visualization components
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional

def create_price_distribution_chart(df: pd.DataFrame, title: str = "Price Distribution") -> go.Figure:
    """Create price distribution histogram"""
    fig = px.histogram(
        df,
        x='price',
        nbins=30,
        title=title,
        labels={'price': 'Price ($)', 'count': 'Count'},
        color_discrete_sequence=['#667eea']
    )
    
    fig.update_layout(
        showlegend=False,
        xaxis_title="Price ($)",
        yaxis_title="Count"
    )
    
    return fig

def create_pie_chart(data: Dict[str, int], title: str = "Distribution") -> go.Figure:
    """Create pie chart from dictionary data"""
    fig = px.pie(
        values=list(data.values()),
        names=list(data.keys()),
        title=title
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    
    return fig

def create_bar_chart(data: Dict[str, int], title: str = "Comparison", 
                     x_label: str = "Category", y_label: str = "Count") -> go.Figure:
    """Create bar chart from dictionary data"""
    fig = px.bar(
        x=list(data.keys()),
        y=list(data.values()),
        labels={'x': x_label, 'y': y_label},
        title=title,
        color=list(data.values()),
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(showlegend=False)
    
    return fig

def create_scatter_plot(df: pd.DataFrame, x_col: str, y_col: str,
                       color_col: Optional[str] = None,
                       size_col: Optional[str] = None,
                       title: str = "Scatter Plot") -> go.Figure:
    """Create scatter plot"""
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        size=size_col,
        title=title,
        hover_data=df.columns.tolist()
    )
    
    return fig


def create_gauge_chart(value: float, title: str = "Metric",
                      min_val: float = 0, max_val: float = 100,
                      thresholds: Optional[List[float]] = None) -> go.Figure:
    """Create gauge chart"""
    if thresholds is None:
        thresholds = [min_val + (max_val - min_val) * i / 3 for i in range(4)]
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        gauge={
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [thresholds[0], thresholds[1]], 'color': "lightgray"},
                {'range': [thresholds[1], thresholds[2]], 'color': "yellow"},
                {'range': [thresholds[2], thresholds[3]], 'color': "lightgreen"}
            ]
        }
    ))
    
    return fig


def render_metric_cards(metrics: Dict[str, Dict[str, any]]):
    """Render multiple metric cards in columns"""
    cols = st.columns(len(metrics))
    
    for col, (label, data) in zip(cols, metrics.items()):
        with col:
            value = data.get('value', 'N/A')
            delta = data.get('delta')
            help_text = data.get('help')
            
            st.metric(label, value, delta=delta, help=help_text)

