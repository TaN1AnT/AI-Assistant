import requests
import json
import time

def test_rag_sources():
    url = "http://localhost:8000/v1/webhook/n8n"
    headers = {
        "X-Webhook-Secret": "password",
        "Content-Type": "application/json"
    }
    
    # Query testing both RAG and Suppa with user metadata
    payload = {
        "query": "Що таке конвінік і дай посилання на джерела документації.",
        "session_id": "full-workflow-metadata-test",
        "user_id": "0238c601-1415-4c51-80b6-d9dfb592ea57",
        "user_email": "mykola.demchuk@modern-expo.com",
        "user_name": "Mykola Demchuk"
    }
    
    print(f"Sending Full Workflow query: {payload['query']}")
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        print("\n--- FULL RESPONSE ---")
        print(json.dumps(data, indent=2, ensure_ascii=False))
        
        print("\n--- AI ANSWER ---")
        print(data.get("answer"))
        print("\n--- SOURCES FIELD ---")
        print(data.get("sources"))
        print("\n--- TOOLS USED ---")
        print(data.get("tools_used"))
        
        if data.get("sources"):
            print("\n✅ SUCCESS: Sources were extracted!")
        else:
            print("\n❌ FAILURE: No sources found in the response model.")
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")

if __name__ == "__main__":
    test_rag_sources()
