import requests
import json
import time
import os

BASE_URL = "http://localhost:8000"

def test_complete_workflow():

    print("-" * 60)
    print("Testing Complete Phase 4 Workflow")
    print("-" * 60)
 
    print("\n1. Testing Neighborhood Analysis API")
    print("-" * 40)
    
    request_data = {
        "address": "Manipal University, Karnataka, India",
        "radius_m": 1200,
        "amenity_types": ["restaurant", "cafe", "park", "university"],
        "generate_map": True
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/neighborhood/analyze",
            json=request_data
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"Analysis started: {result['analysis_id']}")
            print(f" Walk Score: {result.get('walk_score', 'Calculating...')}")
            
            time.sleep(5)
            
            results_response = requests.get(
                f"{BASE_URL}/api/neighborhood/{result['analysis_id']}"
            )
            
            if results_response.status_code == 200:
                details = results_response.json()
                print(f" Analysis completed!")
                print(f" Status: {details.get('status')}")
                print(f" Total amenities found: {sum(len(v) for v in details.get('amenities', {}).values())}")
                
                map_response = requests.get(
                    f"{BASE_URL}/api/neighborhood/map/{result['analysis_id']}"
                )
                
                if map_response.status_code == 200:
                    with open(f"neighborhood_map_{result['analysis_id']}.html", "wb") as f:
                        f.write(map_response.content)
                    print(f"Map saved to: neighborhood_map_{result['analysis_id']}.html")
                else:
                    print(f" Could not download map")
                    
            else:
                print(f"Could not get analysis results")
                
        else:
            print(f"Analysis failed: {response.status_code}")
            
    except Exception as e:
        print(f"Error: {e}")
    
  
    print("\n2. Testing Async Task Processing")
    print("-" * 40)
    
    try:

        task_response = requests.post(
            f"{BASE_URL}/api/tasks/neighborhood/async",
            json=request_data
        )
        
        if task_response.status_code == 200:
            task_data = task_response.json()
            task_id = task_data["task_id"]
            print(f"Async task created: {task_id}")

            for i in range(10):
                status_response = requests.get(f"{BASE_URL}/api/tasks/{task_id}")
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    print(f"   Status: {status_data['status']}")
                    
                    if status_data['status'] == 'SUCCESS':
                        print(f"Task completed successfully!")
                        break
                    elif status_data['status'] == 'FAILURE':
                        print(f"Task failed")
                        break
                
                time.sleep(2)
        else:
            print(f" Failed to create async task")
            
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n3. Testing Workflow Integration")
    print("-" * 40)
    
    try:
        workflow_data = {
            "address": "Udupi, Karnataka, India",
            "radius_m": 1500,
            "email": "test@example.com"
        }
        
        workflow_response = requests.post(
            f"{BASE_URL}/api/workflow/trigger-analysis",
            json=workflow_data
        )
        
        if workflow_response.status_code == 200:
            workflow_result = workflow_response.json()
            print(f" Workflow triggered: {workflow_result['workflow_id']}")
            print(f" Status: {workflow_result['status']}")
            print(f" Email notification: {'Sent' if workflow_result['email_sent'] else 'Not sent'}")
        else:
            print(f" Workflow failed: {workflow_response.status_code}")
            
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n4. Testing Recent Analyses")
    print("-" * 40)
    
    try:
        recent_response = requests.get(f"{BASE_URL}/api/neighborhood/recent?limit=3")
        
        if recent_response.status_code == 200:
            recent_analyses = recent_response.json()
            print(f"Found {len(recent_analyses)} recent analyses:")
            
            for i, analysis in enumerate(recent_analyses, 1):
                print(f"   {i}. {analysis['address'][:50]}...")
                print(f"   Score: {analysis.get('walk_score', 'N/A')}, Amenities: {analysis['total_amenities']}")
        else:
            print(f"Failed to get recent analyses")
            
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "-" * 60)
    print("Phase 4 Testing Complete!")
    print("-" * 60)

def test_supabase_vector_db():
   
    print("\nTesting Supabase Vector Database")
    print("-" * 40)
    
    print(" Supabase testing requires actual Supabase project setup")
    print("   Please configure your .env file with:")
    print("   SUPABASE_URL=your-project-url")
    print("   SUPABASE_KEY=your-anon-key")
    print("\n   Then uncomment the vector DB tests in the code")

if __name__ == "__main__":

    try:
        health = requests.get(f"{BASE_URL}/health")
        if health.status_code == 200:
            test_complete_workflow()
            test_supabase_vector_db()
        else:
            print("API not running. Start it with: uvicorn app.main:app --reload")
    except:
        print("Cannot connect to API. Make sure it's running on localhost:8000")