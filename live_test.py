"""Quick live test: sends 2 queries to the running server and prints results."""
import requests, json, sys

BASE = "http://localhost:8000"
HEADERS = {"X-Webhook-Secret": "password", "Content-Type": "application/json"}

tests = [
    {"name": "1. Simple greeting (no tools)", "body": {"message": "Привіт! Що ти вмієш?", "session_id": "test-greeting"}},
    {"name": "2. Suppa query (tools)", "body": {"message": "List all entities in Suppa", "session_id": "test-suppa"}},
]

for t in tests:
    print(f"\n{'='*60}")
    print(f"TEST: {t['name']}")
    print(f"{'='*60}")
    try:
        r = requests.post(f"{BASE}/v1/webhook/n8n", json=t["body"], headers=HEADERS, timeout=90)
        data = r.json()
        print(f"Status: {r.status_code}")
        print(f"Tools Used: {data.get('tools_used', [])}")
        print(f"Iterations: {data.get('iterations', 0)}")
        print(f"Time: {data.get('execution_time_ms', 0)}ms")
        print(f"Answer: {data.get('answer', '')[:500]}")
    except Exception as e:
        print(f"ERROR: {e}")

print("\n\nDone!")
sys.exit(0)
