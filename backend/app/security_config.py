"""
Security and Configuration Module
Handles rate limiting, CORS, request validation, and secrets management
"""

import os
from typing import List
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

# Load environment variables (kept for convenience; secrets are managed elsewhere)
load_dotenv()


class CORSSettings:
    """
    CORS Configuration
    Customize based on your frontend deployment
    """
    
    # Development CORS origins
    DEVELOPMENT_ORIGINS = [
        "http://localhost:3000",
        "http://localhost:8501",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8501",
        "http://127.0.0.1:8000",
    ]
    
    # Production CORS origins (update with your actual domains)
    PRODUCTION_ORIGINS = [
        "https://yourdomain.com",
        "https://app.yourdomain.com",
        "https://api.yourdomain.com",
    ]
    
    # Staging CORS origins
    STAGING_ORIGINS = [
        "https://staging.yourdomain.com",
        "https://staging-app.yourdomain.com",
    ]
    
    @staticmethod
    def get_cors_config(environment: str = "development") -> dict:
        """Get CORS configuration based on environment"""
        
        if environment == "production":
            origins = CORSSettings.PRODUCTION_ORIGINS
            allow_credentials = True
            allow_methods = ["GET", "POST", "PUT", "DELETE"]
            allow_headers = ["*"]
        elif environment == "staging":
            origins = CORSSettings.STAGING_ORIGINS
            allow_credentials = True
            allow_methods = ["GET", "POST", "PUT", "DELETE", "PATCH"]
            allow_headers = ["*"]
        else:  # development
            origins = CORSSettings.DEVELOPMENT_ORIGINS
            allow_credentials = True
            allow_methods = ["*"]
            allow_headers = ["*"]
        
        return {
            "allow_origins": origins,
            "allow_credentials": allow_credentials,
            "allow_methods": allow_methods,
            "allow_headers": allow_headers,
            "max_age": 600,  # 10 minutes
        }


class RateLimitSettings:
    """
    Rate Limiting Configuration
    Using slowapi (already in requirements.txt)
    """
    
    # Default rate limits per endpoint
    DEFAULT_RATE_LIMIT = "60/minute"
    
    # Specific endpoint rate limits
    RATE_LIMITS = {
        # Auth endpoints
        "/auth/login": "5/minute",
        "/auth/register": "5/minute",
        
        # Heavy computation endpoints
        "/api/neighborhood/analyze": "10/minute",
        "/api/image/analyze": "20/minute",
        "/api/vector-search": "30/minute",
        
        # Normal endpoints
        "/api/properties": "60/minute",
        "/api/properties/{id}": "60/minute",
        
        # Download endpoints
        "/api/download": "20/minute",
    }


class RequestValidationSettings:
    """
    Request validation configuration
    """
    
    # Maximum request body size (bytes)
    MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10MB
    
    # Maximum file upload size (bytes)
    MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB
    
    # Allowed file extensions for uploads
    ALLOWED_FILE_EXTENSIONS = {
        "images": {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"},
        "documents": {".pdf", ".txt", ".csv", ".xlsx", ".json"},
        "data": {".csv", ".json", ".geojson", ".shp"},
    }
    
    # Request validation rules
    VALIDATION_RULES = {
        "min_query_length": 1,
        "max_query_length": 500,
        "min_password_length": 8,
        "max_batch_size": 1000,
        "min_lat": -90,
        "max_lat": 90,
        "min_lon": -180,
        "max_lon": 180,
    }
def get_cors_config(environment: str = "development") -> dict:
    """Convenience wrapper to get CORS config based on environment"""
    return CORSSettings.get_cors_config(environment=environment)
