import requests
import json

def test_greeting():
    url = "http://127.0.0.1:8000/v1/webhook/n8n"
    headers = {
        "X-Webhook-Secret": "password",
        "Content-Type": "application/json"
    }
    payload = {
        "query": "hello",
        "session_id": "test-greeting-session"
    }
    
    print(f"Sending greeting: {payload['query']}")
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        print("\n--- AI ANSWER ---")
        print(data.get("answer"))
    except Exception as e:
        print(f"\n--- ERROR ---")
        print(e)

if __name__ == "__main__":
    test_greeting()
