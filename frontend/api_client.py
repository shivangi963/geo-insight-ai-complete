"""
API Client for Backend Communication
Handles all HTTP requests to the FastAPI backend
"""
import streamlit as st
import requests
from typing import Optional, Dict, Any, List
from config import api_config
import time

class APIClient:
    """Unified API client with error handling and retry logic"""
    
    def __init__(self, base_url: str = None):
        self.base_url = base_url or api_config.base_url
        self.timeout = api_config.timeout
        self.max_retries = api_config.max_retries
    
    def _handle_error(self, error: Exception, endpoint: str):
        """Centralized error handling"""
        if isinstance(error, requests.exceptions.ConnectionError):
            st.error(f"🔌 **Connection Failed**")
            st.error(f"Cannot connect to backend at {self.base_url}")
            st.info("💡 Make sure backend is running: `uvicorn app.main:app --reload`")
        elif isinstance(error, requests.exceptions.Timeout):
            st.error(f"⏱️ **Request Timeout** ({self.timeout}s)")
            st.warning("Backend is taking longer than expected")
        elif isinstance(error, requests.exceptions.HTTPError):
            try:
                detail = error.response.json().get('detail', error.response.text)
                st.error(f"❌ **HTTP {error.response.status_code}**")
                st.error(f"Details: {detail}")
            except:
                st.error(f"❌ HTTP Error: {error.response.text}")
        else:
            st.error(f"❌ **Unexpected Error**: {str(error)}")
    
    def get(self, endpoint: str, params: Dict = None, show_errors: bool = True) -> Optional[Dict]:
        """GET request with retry logic"""
        url = f"{self.base_url}{endpoint}"
        
        for attempt in range(self.max_retries):
            try:
                response = requests.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()
                return response.json()
            
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    time.sleep(1 * (attempt + 1))  # Exponential backoff
                    continue
                
                if show_errors:
                    self._handle_error(e, endpoint)
                return None
    
    def post(self, endpoint: str, data: Dict = None, files: Dict = None, 
             show_errors: bool = True) -> Optional[Dict]:
        """POST request"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            if files:
                response = requests.post(url, files=files, timeout=self.timeout)
            else:
                response = requests.post(url, json=data, timeout=self.timeout)
            
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            if show_errors:
                self._handle_error(e, endpoint)
            return None
    
    def put(self, endpoint: str, data: Dict, show_errors: bool = True) -> Optional[Dict]:
        """PUT request"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = requests.put(url, json=data, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            if show_errors:
                self._handle_error(e, endpoint)
            return None
    
    def delete(self, endpoint: str, show_errors: bool = True) -> Optional[Dict]:
        """DELETE request"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = requests.delete(url, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            if show_errors:
                self._handle_error(e, endpoint)
            return None
    
    # ==================== Convenience Methods ====================
    
    def health_check(self) -> Optional[Dict]:
        """Check backend health"""
        return self.get("/health", show_errors=False)
    
    def get_stats(self) -> Optional[Dict]:
        """Get system statistics"""
        return self.get("/api/stats")
    
    def get_properties(self, skip: int = 0, limit: int = 100, 
                       city: str = None) -> Optional[List[Dict]]:
        """Get properties with filters"""
        params = {"skip": skip, "limit": limit}
        if city:
            params["city"] = city
        return self.get("/api/properties", params=params)
    
    def create_property(self, property_data: Dict) -> Optional[Dict]:
        """Create new property"""
        return self.post("/api/properties", data=property_data)
    
    def start_neighborhood_analysis(self, analysis_data: Dict) -> Optional[Dict]:
        """Start neighborhood analysis"""
        return self.post("/api/neighborhood/analyze", data=analysis_data)
    
    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """Get async task status"""
        return self.get(f"/api/tasks/{task_id}", show_errors=False)
    
    def get_analysis(self, analysis_id: str) -> Optional[Dict]:
        """Get completed analysis"""
        return self.get(f"/api/neighborhood/{analysis_id}")
    
    def query_ai_agent(self, query: str) -> Optional[Dict]:
        """Query AI assistant"""
        return self.post("/api/agent/query", data={"query": query})
    
    def analyze_image(self, file_content: bytes, filename: str, 
                     analysis_type: str) -> Optional[Dict]:
        """Upload image for analysis"""
        files = {'file': (filename, file_content)}
        params = {'analysis_type': analysis_type}
        
        url = f"{self.base_url}/api/analysis/image"
        try:
            response = requests.post(url, files=files, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self._handle_error(e, "/api/analysis/image")
            return None
    
    def vector_search(self, file_content: bytes, filename: str,
                     limit: int = 5, threshold: float = 0.7) -> Optional[Dict]:
        """Search similar properties using vector DB"""
        files = {'file': (filename, file_content)}
        params = {'limit': limit, 'threshold': threshold}
        
        url = f"{self.base_url}/api/vector/search"
        try:
            response = requests.post(url, files=files, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self._handle_error(e, "/api/vector/search")
            return None

# Global instance
api = APIClient()