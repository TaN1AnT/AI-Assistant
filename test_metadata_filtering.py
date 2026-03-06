import requests
import json

def test_metadata_filtering():
    url = "http://localhost:8000/v1/webhook/n8n"
    headers = {
        "X-Webhook-Secret": "password",
        "Content-Type": "application/json"
    }
    
    # Query specifically asking for "my tasks" using Mykola's metadata
    payload = {
        "query": "Я Mykola Demchuk, покажи мої завдання.",
        "session_id": "metadata-filtering-test",
        "user_id": "0238c601-1415-4c51-80b6-d9dfb592ea57",
        "user_email": "mykola.demchuk@modern-expo.com",
        "user_name": "Mykola Demchuk"
    }
    
    print(f"Sending Metadata Filtering query: {payload['query']}")
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        print("\n--- FULL RESPONSE ---")
        print(json.dumps(data, indent=2, ensure_ascii=False))
        
        tools_used = data.get("tools_used", [])
        print(f"\nTools used: {tools_used}")
        
        if "suppa_search_instances" in tools_used or "suppa_list_entities" in tools_used:
            print("\n✅ SUCCESS: Agent attempted to use Suppa tools with metadata!")
        else:
            print("\n❌ FAILURE: Agent did not use Suppa tools as expected.")
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")

if __name__ == "__main__":
    test_metadata_filtering()
