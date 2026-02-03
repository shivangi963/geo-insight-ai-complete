import requests
import json

def test_n8n_workflow():
    """Test n8n workflow integration"""
    
    # n8n webhook URL
    webhook_url = "http://localhost:5678/webhook/geoinsight-analysis"
    
    # Test payload
    payload = {
        "address": "MIT Campus, Manipal, Karnataka",
        "radius_m": 1500,
        "email": "test@example.com"
    }
    
    print("🚀 Triggering n8n workflow...")
    print(f"📍 Address: {payload['address']}")
    
    try:
        response = requests.post(webhook_url, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ Workflow triggered successfully!")
            print(f"📊 Response: {json.dumps(result, indent=2)}")
        else:
            print(f"\n❌ Error: {response.status_code}")
            print(response.text)
    
    except requests.exceptions.ConnectionError:
        print("\n❌ Cannot connect to n8n. Is it running on port 5678?")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    test_n8n_workflow()