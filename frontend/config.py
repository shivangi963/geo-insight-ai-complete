"""
Frontend Configuration
Centralized settings for the Streamlit application
"""
import os
from dataclasses import dataclass
from typing import List

@dataclass
class APIConfig:
    """API connection settings"""
    base_url: str = os.getenv("BACKEND_URL", "http://localhost:8000")
    timeout: int = 30
    max_retries: int = 3
    
@dataclass
class UIConfig:
    """UI/UX settings"""
    page_title: str = "GeoInsight AI - Real Estate Intelligence"
    page_icon: str = "🏠"
    layout: str = "wide"
    initial_sidebar_state: str = "expanded"
    
    # Theme colors
    primary_color: str = "#667eea"
    secondary_color: str = "#764ba2"
    success_color: str = "#28a745"
    warning_color: str = "#ffc107"
    error_color: str = "#dc3545"
    
@dataclass
class FeatureConfig:
    """Feature flags"""
    enable_vector_search: bool = True
    enable_image_analysis: bool = True
    enable_ai_agent: bool = True
    enable_maps: bool = True
    max_file_size_mb: int = 10
    
@dataclass
class MapConfig:
    """Map visualization settings"""
    default_zoom: int = 15
    default_radius: int = 1000
    min_radius: int = 100
    max_radius: int = 5000
    
    amenity_types: List[str] = None
    
    def __post_init__(self):
        if self.amenity_types is None:
            self.amenity_types = [
                'restaurant', 'cafe', 'school', 'hospital',
                'park', 'supermarket', 'bank', 'pharmacy',
                'gym', 'library', 'transit_station'
            ]

@dataclass
class PaginationConfig:
    """Pagination settings"""
    default_page_size: int = 20
    max_page_size: int = 100
    
# Global instances
api_config = APIConfig()
ui_config = UIConfig()
feature_config = FeatureConfig()
map_config = MapConfig()
pagination_config = PaginationConfig()

# Polling settings for async tasks
TASK_POLL_INTERVAL = 2  # seconds
TASK_MAX_WAIT = 120  # seconds
TASK_PROGRESS_BAR_ENABLED = True