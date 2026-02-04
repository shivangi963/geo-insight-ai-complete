from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime


class PropertyBase(BaseModel):
    address: str
    city: str
    state: str
    zip_code: str
    price: Optional[float] = None
    bedrooms: Optional[int] = None
    bathrooms: Optional[float] = None
    square_feet: Optional[int] = None
    property_type: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None

class PropertyCreate(PropertyBase):
    pass

class PropertyUpdate(BaseModel):
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    price: Optional[float] = None
    bedrooms: Optional[int] = None
    bathrooms: Optional[float] = None
    square_feet: Optional[int] = None
    property_type: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None

class PropertyResponse(PropertyBase):
    id: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    model_config = ConfigDict(from_attributes=True)

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str



class Coordinates(BaseModel):
 
    latitude: float = Field(..., ge=-90, le=90, description="Latitude in decimal degrees")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude in decimal degrees")

class Amenity(BaseModel):
  
    name: str = Field(..., description="Name of the amenity")
    type: str = Field(..., description="Type of amenity (restaurant, park, etc.)")
    coordinates: Coordinates
    distance_km: float = Field(..., description="Distance in kilometers")
    @field_validator('coordinates', mode='before')
    @classmethod
    def create_coordinates(cls, v):
        if isinstance(v, dict):
            if 'latitude' in v and 'longitude' in v:
                return v
            elif 'lat' in v and 'lon' in v:
                return {'latitude': v['lat'], 'longitude': v['lon']}
        return v

class BuildingFootprint(BaseModel):
 
    building_id: str
    building_type: str
    geometry_type: str
    area_sq_m: Optional[float] = None
    centroid: Coordinates
    @field_validator('centroid', mode='before')
    @classmethod
    def normalize_centroid(cls, v):
        if isinstance(v, dict):
            if 'lat' in v and 'lon' in v:
                return {'latitude': v['lat'], 'longitude': v['lon']}
            elif 'latitude' in v and 'longitude' in v:
                return v
        return v

class NeighborhoodAnalysis(BaseModel):
    
    address: str
    coordinates: Optional[Dict[str, float]] = None  # Accept dict instead of strict Coordinates object
    search_radius_m: int = Field(default=1000, description="Search radius in meters")
    amenities: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)  # Accept generic dicts
    building_footprints: List[Dict[str, Any]] = Field(default_factory=list)  # Accept generic dicts
    walk_score: Optional[float] = Field(None, ge=0, le=100, description="Walkability score")
    map_path: Optional[str] = Field(None, description="Path to interactive map HTML file")
    status: Optional[str] = Field(default="pending")
    progress: Optional[int] = Field(default=0)
    analysis_date: Optional[datetime] = Field(default_factory=datetime.now)
    total_amenities: Optional[int] = None
    amenity_categories: Optional[int] = None
    
    model_config = ConfigDict(from_attributes=True)

class NeighborhoodAnalysisRequest(BaseModel):
   
    address: str
    radius_m: Optional[int] = Field(default=1000, ge=100, le=5000)
    amenity_types: Optional[List[str]] = Field(
        default=[
            'restaurant', 'cafe', 'school', 'hospital',
            'park', 'supermarket', 'bank', 'pharmacy'
        ]
    )
    include_buildings: Optional[bool] = Field(default=True)
    generate_map: Optional[bool] = Field(default=True)

class NeighborhoodAnalysisResponse(BaseModel):
    analysis_id: str
    status: str
    address: Optional[str] = None
    walk_score: Optional[float] = None
    total_amenities: Optional[int] = None  
    building_count: Optional[int] = None  
    map_path: Optional[str] = None
    coordinates: Optional[Tuple[float, float]] = None
    amenities: Optional[Dict[str, List[Dict[str, Any]]]] = None
    timestamp: Optional[str] = None
    task_id: Optional[str] = None
    message: Optional[str] = None