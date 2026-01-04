import asyncio
import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_neighborhood_analysis():
    
    print("Testing neighborhood analysis...")

    request_data = {
        "address": "Manipal, Karnataka, India",
        "radius_m": 1500,
        "amenity_types": ["restaurant", "cafe", "park", "school"],
        "include_buildings": True,
        "generate_map": True
    }
    
    try:

        response = requests.post(
            f"{BASE_URL}/api/neighborhood/analyze",
            json=request_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"Analysis started successfully!")
            print(f"Analysis ID: {result['analysis_id']}")
            print(f"Status: {result['status']}")
            print(f"Walk Score: {result.get('walk_score', 'N/A')}")
            print(f"Total Amenities: {result['total_amenities']}")
            

            if result['status'] == 'processing':
                print("Waiting for map generation...")
                time.sleep(5)
            
            return result['analysis_id']
        else:
            print(f"API Error: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"Request failed: {e}")
        return None

def test_get_analysis_results(analysis_id: str):

    print(f"\nGetting analysis results for ID: {analysis_id}")
    
    try:
        response = requests.get(f"{BASE_URL}/api/neighborhood/{analysis_id}")
        
        if response.status_code == 200:
            result = response.json()
            print("Analysis results retrieved successfully!")
            

            print(f"Address: {result['address']}")
            print(f"Walk Score: {result.get('walk_score', 'N/A')}")
            
            amenities = result.get('amenities', {})
            print(" Amenities found:")
            for amenity_type, items in amenities.items():
                if items:
                    print(f"  - {amenity_type}: {len(items)}")
            
            print(f"Total buildings: {len(result.get('building_footprints', []))}")
            return True
        else:
            print(f"Failed to get results: {response.status_code}")
            return False
            
    except Exception as e:
        print(f" Request failed: {e}")
        return False


def test_get_analysis_map(analysis_id: str):
    print(f"\nGetting map for analysis: {analysis_id}")
    
    try:
        response = requests.get(f"{BASE_URL}/api/neighborhood/map/{analysis_id}")
        
        if response.status_code == 200:
            map_filename = f"neighborhood_map_{analysis_id}.html"
            with open(map_filename, 'wb') as f:
                f.write(response.content)
            
            print(f"Map saved as: {map_filename}")
            print("Open this file in your browser to view the interactive map")
            return True
        else:
            print(f"Failed to get map: {response.status_code}")
            return False
            
    except Exception as e:
        print(f" Request failed: {e}")
        return False

def test_amenity_search():
    print("\nTesting amenity search...")
    
    params = {
        "address": "Udupi, Karnataka, India",
        "amenity_type": "restaurant",
        "radius_m": 1000
    }
    
    try:
        response = requests.get(f"{BASE_URL}/api/neighborhood/search", params=params)
        
        if response.status_code == 200:
            result = response.json()
            print(f"Found {result['total_found']} restaurants near {params['address']}")
            
            if result['amenities']:
                print(" Top 3 restaurants:")
                for i, restaurant in enumerate(result['amenities'][:3], 1):
                    print(f"  {i}. {restaurant['name']} ({restaurant['distance_km']}km)")
            return True
        else:
            print(f"Search failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"Request failed: {e}")
        return False

def test_recent_analyses():
    print("\nTesting recent analyses...")
    
    try:
        response = requests.get(f"{BASE_URL}/api/neighborhood/recent?limit=5")
        
        if response.status_code == 200:
            analyses = response.json()
            print(f"Retrieved {len(analyses)} recent analyses:")
            
            for i, analysis in enumerate(analyses, 1):
                print(f"   {i}. {analysis['address']} - Score: {analysis.get('walk_score', 'N/A')}")
            return True
        else:
            print(f" Failed to get recent analyses: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"equest failed: {e}")
        return False

def main():
    """Run all tests"""
    print("-" * 60)
    print("Neighborhood Analyzer API Tests")
    print("-" * 60)

    print("Checking API health...")
    try:
        health_response = requests.get(f"{BASE_URL}/health")
        if health_response.status_code == 200:
            print(" API is running!")
        else:
            print(" API is not responding. Make sure to run: uvicorn app.main:app --reload")
            return
    except:
        print("Cannot connect to API. Make sure it's running on localhost:8000")
        return
    
    analysis_id = test_neighborhood_analysis()
    
    if analysis_id:
        time.sleep(3)
        
        test_get_analysis_results(analysis_id)
        test_get_analysis_map(analysis_id)
    
    test_amenity_search()
    test_recent_analyses()
    
    print("\n" + "-" * 60)
    print("All tests completed!")
    print("-" * 60)

if __name__ == "__main__":
    main()