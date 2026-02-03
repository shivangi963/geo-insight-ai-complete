"""
Vector Search Page
Image-based property similarity search
"""
import streamlit as st
from api_client import api
from utils import (
    validate_file_size, show_success_message, 
    show_error_message, format_percentage
)
from components.header import render_section_header
from config import feature_config

def search_similar_properties(api_url, query_image, limit=3, threshold=0.7):
    """
    Search for visually similar properties
    
    Args:
        api_url: Base API URL
        query_image: Streamlit UploadedFile object
        limit: Maximum number of results
        threshold: Similarity threshold
    
    Returns:
        API response dictionary
    """
    import requests
    
    if query_image is None:
        st.error("Please upload a query image first")
        return None
    
    try:
        # Reset file pointer
        query_image.seek(0)
        
        # Create multipart payload
        files = {
            'file': (
                query_image.name,
                query_image,
                query_image.type
            )
        }
        
        params = {
            'limit': limit,
            'threshold': threshold
        }
        
        response = requests.post(
            f"{api_url}/api/vector/search",
            files=files,
            params=params,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            error_detail = response.json().get('detail', f'HTTP {response.status_code}')
            st.error(f"❌ Search failed: {error_detail}")
            return None
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None
    

def render_vector_search_page():
    """Main vector search page"""
    render_section_header("Vector Similarity Search", "🔍")
    
    # Check if vector DB is available
    health = api.health_check()
    vector_enabled = health and health.get('features', {}).get('vector_db', False)
    
    if not vector_enabled:
        render_vector_not_available()
        return
    
    show_success_message("Vector search enabled!")
    
    # Main tabs
    search_tab, store_tab, stats_tab = st.tabs([
        "🔍 Search", 
        "💾 Store", 
        "📊 Statistics"
    ])
    
    with search_tab:
        render_search_tab()
    
    with store_tab:
        render_store_tab()
    
    with stats_tab:
        render_stats_tab()

def render_vector_not_available():
    """Show message when vector DB not available"""
    st.warning("⚠️ Vector database not configured")
    
    st.info("""
    ### How to Enable Vector Search
    
    Vector search requires Supabase configuration:
    
    1. **Create Supabase Account**
       - Visit https://supabase.com
       - Create a new project
    
    2. **Set up Vector Extension**
       ```sql
       CREATE EXTENSION IF NOT EXISTS vector;
       ```
    
    3. **Add Credentials to Backend**
       Edit `backend/.env`:
       ```
       SUPABASE_URL=your_project_url
       SUPABASE_KEY=your_anon_key
       ```
    
    4. **Install Dependencies**
       ```bash
       pip install supabase transformers torch
       ```
    
    5. **Restart Backend**
       ```bash
       uvicorn app.main:app --reload
       ```
    """)

def render_search_tab():
    """Render search interface"""
    st.subheader("🔍 Find Similar Properties")
    
    st.markdown("""
    Upload a property image to find visually similar properties in the database.
    Uses AI-powered image embeddings for similarity matching.
    """)
    
    # File uploader
    uploaded = st.file_uploader(
        "📤 Upload Property Image", 
        type=['jpg', 'jpeg', 'png'],
        help="Upload an image to find similar properties",
        key="vector_search_upload"
    )
    
    if uploaded:
        render_search_image_preview(uploaded)
    
    # Search parameters
    col1, col2 = st.columns(2)
    
    with col1:
        limit = st.slider(
            "📊 Number of Results", 
            min_value=1, 
            max_value=20, 
            value=5,
            help="Maximum number of similar properties to return"
        )
    
    with col2:
        threshold = st.slider(
            "🎯 Similarity Threshold", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.7, 
            step=0.05,
            help="Minimum similarity score (0-1)"
        )
    
    # Search button
    if uploaded and st.button("🔍 Search Similar Properties", type="primary", use_container_width=True):
        handle_vector_search(uploaded, limit, threshold)

def render_search_image_preview(uploaded_file):
    """Display uploaded image preview"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.image(uploaded_file, caption="Query Image", use_container_width=True)
    
    with col2:
        st.markdown("### 📋 File Info")
        st.info(f"**Name:** {uploaded_file.name}")
        st.info(f"**Size:** {uploaded_file.size / 1024:.1f} KB")

def handle_vector_search(uploaded_file, limit: int, threshold: float):
    """Execute vector similarity search - FIXED VERSION"""
    from api_client import api
    
    if not validate_file_size(uploaded_file, 10):
        return
    
    st.divider()
    
    # Use the new search function
    with st.spinner("🔍 Searching for similar properties..."):
        result = search_similar_properties(
            api_url=api.base_url,
            query_image=uploaded_file,
            limit=limit,
            threshold=threshold
        )
    
    if not result:
        return
    
    # Display results
    results = result.get('results', [])
    
    if not results:
        st.info("❌ No similar properties found")
        st.caption(f"Try lowering the similarity threshold (currently {threshold})")
        return
    
    show_success_message(f"Found {len(results)} similar properties")
    render_search_results(results)

def render_search_results(results: list):
    """Display search results"""
    st.subheader("📊 Similar Properties")
    
    for idx, result in enumerate(results, 1):
        render_result_card(result, idx)

def render_result_card(result: dict, idx: int):
    """Render individual result card"""
    similarity = result.get('similarity', 0)
    address = result.get('address', 'Unknown')
    property_id = result.get('property_id', '')
    metadata = result.get('metadata', {})
    
    with st.expander(f"#{idx} - {address} | Similarity: {format_percentage(similarity * 100)}"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📍 Property Details**")
            st.write(f"**Address:** {address}")
            st.write(f"**ID:** {property_id}")
            st.write(f"**Similarity:** {format_percentage(similarity * 100)}")
        
        with col2:
            st.markdown("**📊 Metadata**")
            if metadata:
                for key, value in metadata.items():
                    st.write(f"**{key.title()}:** {value}")
            else:
                st.caption("No additional metadata")
        
        # Similarity score bar
        st.progress(similarity)
        
        # Action buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 View Full Details", key=f"view_vector_{idx}"):
                st.info("Feature coming soon: Full property details")
        
        with col2:
            if st.button("🗺️ Analyze Area", key=f"analyze_vector_{idx}"):
                st.session_state.nav_to_analysis = address
                show_success_message("Switched to Neighborhood tab")

def render_store_tab():
    """Render vector storage interface"""
    st.subheader("💾 Store Property Embedding")
    
    st.info("""
    This feature allows you to add property images to the vector database.
    
    **Note:** This requires additional setup and is typically done via the backend API.
    For now, use the backend endpoints directly or the batch processing feature.
    """)
    
    with st.expander("📚 How to Add Properties to Vector DB"):
        st.markdown("""
        ### Option 1: Via Backend API
        
        ```bash
        curl -X POST "http://localhost:8000/api/vector/store" \\
          -H "Content-Type: application/json" \\
          -d '{
            "property_id": "prop_123",
            "address": "123 Main St",
            "image_path": "/path/to/image.jpg",
            "metadata": {"price": 500000, "beds": 3}
          }'
        ```
        
        ### Option 2: Batch Processing
        
        ```bash
        curl -X POST "http://localhost:8000/api/vector/batch-store?limit=100"
        ```
        
        This will process all properties in the database.
        
        ### Option 3: Python Script
        
        ```python
        import requests
        
        response = requests.post(
            "http://localhost:8000/api/vector/store",
            json={
                "property_id": "prop_123",
                "address": "123 Main St",
                "image_path": "image.jpg",
                "metadata": {"price": 500000}
            }
        )
        ```
        """)

def render_stats_tab():
    """Render vector DB statistics"""
    st.subheader("📊 Vector Database Statistics")
    
    with st.spinner("Loading statistics..."):
        stats = api.get("/api/vector/stats")
    
    if not stats:
        st.error("Failed to load statistics")
        return
    
    # Display stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total = stats.get('total_properties', 0)
        st.metric("🏠 Total Properties", total)
    
    with col2:
        dim = stats.get('embedding_dimension', 0)
        st.metric("🧮 Embedding Dimension", dim)
    
    with col3:
        table = stats.get('table_name', 'N/A')
        st.metric("📋 Table", table)
    
    # Additional info
    st.divider()
    
    with st.expander("ℹ️ About Vector Search"):
        st.markdown("""
        ### How Vector Search Works
        
        1. **Image Encoding**
           - Images are converted to numerical vectors (embeddings)
           - Uses CLIP model (512 dimensions)
        
        2. **Similarity Calculation**
           - Cosine similarity between vectors
           - Range: 0 (different) to 1 (identical)
        
        3. **Search Process**
           - Query image → embedding
           - Compare with all stored embeddings
           - Return top matches above threshold
        
        ### Benefits
        
        - 🎯 Find visually similar properties
        - 🏗️ Group similar architectural styles
        - 📊 Market analysis by visual features
        - 🔍 Reverse image search for properties
        
        ### Performance
        
        - **Fast:** Vectorized operations
        - **Scalable:** Handles thousands of properties
        - **Accurate:** AI-powered matching
        """)
    
    # Refresh button
    if st.button("🔄 Refresh Statistics", use_container_width=True):
        st.rerun()