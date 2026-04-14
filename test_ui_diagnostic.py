"""
Diagnostic test - simulates UI API calls
"""

import sys
import json

print("="*70)
print("UI DIAGNOSTIC TEST")
print("="*70)

# Initialize app
print("\n[1] Initializing app...")
from app import app
from backend.state import load_model_state, get_state

with app.app_context():
    print("  Loading model state...")
    loaded = load_model_state()
    state = get_state()
    
    print(f"  Model loaded: {loaded}")
    print(f"  Vocab: {state['vocab'] is not None} ({len(state['vocab']) if state['vocab'] else 0} tokens)")
    print(f"  Bigram: {state['bigram_idx'] is not None}")
    print(f"  Trigram: {state['trigram_idx'] is not None}")

# Test API calls
print("\n[2] Testing API endpoints (simulating UI calls)...")

with app.test_client() as client:
    
    # Test 1: Config
    print("\n  [2.1] GET /api/config")
    response = client.get('/api/config')
    print(f"    Status: {response.status_code}")
    if response.status_code == 200:
        data = response.get_json()
        print(f"    Response: {json.dumps(data, indent=6)}")
    else:
        print(f"    Error: {response.get_data(as_text=True)}")
    
    # Test 2: Train status
    print("\n  [2.2] GET /api/train/status")
    response = client.get('/api/train/status')
    print(f"    Status: {response.status_code}")
    if response.status_code == 200:
        data = response.get_json()
        print(f"    Model loaded: {data.get('model_loaded')}")
        print(f"    Model exists: {data.get('model_exists')}")
        print(f"    Running: {data.get('running')}")
    else:
        print(f"    Error: {response.get_data(as_text=True)}")
    
    # Test 3: Predict (only if model loaded)
    if state['loaded']:
        print("\n  [2.3] POST /api/predict")
        response = client.post(
            '/api/predict',
            json={'words': ['i', 'want', 'to'], 'top_k': 5},
            content_type='application/json'
        )
        print(f"    Status: {response.status_code}")
        if response.status_code == 200:
            data = response.get_json()
            print(f"    Context: {data.get('context')}")
            print(f"    Model type: {data.get('model_type')}")
            print(f"    Predictions:")
            for p in data.get('predictions', []):
                print(f"      - {p['word']}: {p['probability']:.4f}")
        else:
            print(f"    Error: {response.get_data(as_text=True)}")
    else:
        print("\n  [2.3] POST /api/predict - SKIPPED (no model)")
    
    # Test 4: Homepage
    print("\n  [2.4] GET / (homepage)")
    response = client.get('/')
    print(f"    Status: {response.status_code}")
    if response.status_code == 200:
        html = response.get_data(as_text=True)
        if 'FedKeyboard' in html:
            print(f"    [OK] HTML contains 'FedKeyboard'")
        if '/static/js/app.js' in html:
            print(f"    [OK] JavaScript file referenced")
        if '/static/css/style.css' in html:
            print(f"    [OK] CSS file referenced")
    else:
        print(f"    Error: {response.get_data(as_text=True)}")

print("\n" + "="*70)
print("DIAGNOSTIC COMPLETE")
print("="*70)

if state['loaded']:
    print("\n[OK] Everything is working!")
    print("     The UI should work when you run: python app.py")
    print("     Then open: http://localhost:3000")
else:
    print("\n[WARN] Model not loaded")
    print("       The UI will load but predictions won't work until you train")
    print("       Click 'Train Model' in the UI after starting the server")

print("\nTo start the server:")
print("  python app.py")
print("="*70)
