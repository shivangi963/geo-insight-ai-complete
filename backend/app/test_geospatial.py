from geospatial import (
    LocationGeocoder,
    OpenStreetMapClient,
    calculate_walk_score
)

def test_geocoding():
    print("Testing geocoding")
    geocoder = LocationGeocoder()
    
    address = "Udupi, Karnataka, India"
    coordinates = geocoder.address_to_coordinates(address)
    
    if coordinates:
        print(f"Coordinates for '{address}': {coordinates}")
        
        reverse_address = geocoder.coordinates_to_address(*coordinates)
        print(f"Reverse geocode: {reverse_address[:100]}...")
    else:
        print("Geocoding failed")

def test_amenities_search():
  
    print("\nTesting amenities search...")
    osm_client = OpenStreetMapClient()
    

    address = "Manipal, Karnataka, India"
    amenities_data = osm_client.get_nearby_amenities(
        address=address,
        radius=2000,  
        amenity_types=['restaurant', 'cafe', 'school', 'park']
    )
    
    if "error" not in amenities_data:
        print(f"Found amenities for '{address}'")
        

        for amenity_type, items in amenities_data.get('amenities', {}).items():
            if items:
                print(f"  {amenity_type.title()}: {len(items)} found")
                for item in items[:2]: 
                    print(f"    - {item['name']} ({item['distance_km']}km)")
    else:
        print(f"Error: {amenities_data['error']}")

def test_walk_score():
    print("\nTesting walk score calculation")
    osm_client = OpenStreetMapClient()
    
    address = "Manipal, Karnataka, India"
    amenities_data = osm_client.get_nearby_amenities(address, radius=1500)
    
    if "error" not in amenities_data:
        coordinates = amenities_data.get("coordinates")
        if coordinates:
            walk_score = calculate_walk_score(coordinates, amenities_data)
            print(f"Walk score for '{address}': {walk_score}/100")
    else:
        print(f"Error: {amenities_data['error']}")

def test_map_creation():
    print("\nTesting map creation")
    osm_client = OpenStreetMapClient()
    
    address = "Manipal, Karnataka, India"
    amenities_data = osm_client.get_nearby_amenities(
        address=address,
        radius=1000,
        amenity_types=['restaurant', 'cafe', 'park']
    )
    
    if "error" not in amenities_data:
        map_path = osm_client.create_map_visualization(
            address=address,
            amenities_data=amenities_data,
            save_path="test_map.html"
        )
        
        if map_path:
            print(f" Map created: {map_path}")
            print("  Open this file in your browser to see the interactive map!")
        else:
            print("Map creation failed")
    else:
        print(f"Error: {amenities_data['error']}")

if __name__ == "__main__":
    print("=" * 50)
    print("Geospatial Module Tests")
    print("=" * 50)
    
    test_geocoding()
    test_amenities_search()
    test_walk_score()
    test_map_creation()
    
    print("\n" + "=" * 50)
    print("All tests completed!")
    print("=" * 50)