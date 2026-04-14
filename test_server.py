"""
Test server startup and basic API responses
"""

import sys
import time
import threading
import requests

print("Testing server startup and API responses...")

# Start server in background thread
def run_server():
    from app import app
    from backend.state import load_model_state
    load_model_state()
    app.run(host='127.0.0.1', port=3000, debug=False, use_reloader=False)

server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()

# Wait for server to start
print("Waiting for server to start...")
time.sleep(3)

# Test endpoints
base_url = "http://127.0.0.1:3000"

print("\n[1/4] Testing GET /api/config...")
try:
    r = requests.get(f"{base_url}/api/config", timeout=5)
    print(f"  Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"  [OK] Config: {list(data.keys())}")
    else:
        print(f"  [FAIL] Unexpected status")
except Exception as e:
    print(f"  [FAIL] {e}")

print("\n[2/4] Testing GET /api/train/status...")
try:
    r = requests.get(f"{base_url}/api/train/status", timeout=5)
    print(f"  Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"  [OK] Model loaded: {data.get('model_loaded')}")
    else:
        print(f"  [FAIL] Unexpected status")
except Exception as e:
    print(f"  [FAIL] {e}")

print("\n[3/4] Testing POST /api/predict...")
try:
    r = requests.post(
        f"{base_url}/api/predict",
        json={"words": ["i", "want", "to"], "top_k": 5},
        timeout=5
    )
    print(f"  Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"  [OK] Predictions: {[p['word'] for p in data.get('predictions', [])]}")
    elif r.status_code == 503:
        print(f"  [WARN] Model not trained yet (expected if no model)")
    else:
        print(f"  [FAIL] Unexpected status")
        print(f"  Response: {r.text}")
except Exception as e:
    print(f"  [FAIL] {e}")

print("\n[4/4] Testing GET / (homepage)...")
try:
    r = requests.get(base_url, timeout=5)
    print(f"  Status: {r.status_code}")
    if r.status_code == 200 and 'FedKeyboard' in r.text:
        print(f"  [OK] Homepage loads")
    else:
        print(f"  [FAIL] Unexpected response")
except Exception as e:
    print(f"  [FAIL] {e}")

print("\n" + "="*60)
print("Server is running at http://127.0.0.1:3000")
print("Open this URL in your browser to test the UI")
print("Press Ctrl+C to stop")
print("="*60)

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nShutting down...")
    sys.exit(0)
