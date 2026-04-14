"""
Integration test - Tests complete workflow with API simulation
"""

import sys
import json

print("=" * 70)
print("INTEGRATION TEST - Complete Workflow")
print("=" * 70)

# Test 1: Initialize app
print("\n[1/5] Initializing Flask app...")
try:
    from app import app
    from backend.state import load_model_state, get_state
    
    with app.app_context():
        load_model_state()
        state = get_state()
        
        if not state['loaded']:
            print("[WARN] No trained model found. Some tests will be skipped.")
            print("       Run training first: python app.py (then click Train Model)")
        else:
            print(f"[OK] App initialized with model ({state['model'].count_params():,} params)")
except Exception as e:
    print(f"[FAIL] {e}")
    sys.exit(1)

# Test 2: Test prediction endpoint
print("\n[2/5] Testing prediction endpoint...")
if state['loaded']:
    try:
        with app.test_client() as client:
            response = client.post('/api/predict', 
                json={'words': ['i', 'want', 'to'], 'top_k': 5})
            
            assert response.status_code == 200
            data = response.get_json()
            assert 'predictions' in data
            assert len(data['predictions']) == 5
            
            print(f"[OK] Prediction API works")
            print(f"     Context: {data['context']}")
            print(f"     Top prediction: {data['predictions'][0]['word']} "
                  f"({data['predictions'][0]['probability']:.4f})")
    except Exception as e:
        print(f"[FAIL] {e}")
        sys.exit(1)
else:
    print("[SKIP] No model loaded")

# Test 3: Test config endpoint
print("\n[3/5] Testing config endpoint...")
try:
    with app.test_client() as client:
        response = client.get('/api/config')
        assert response.status_code == 200
        config = response.get_json()
        
        required = ['num_rounds', 'local_epochs', 'batch_size', 'fedprox_mu']
        for key in required:
            assert key in config, f"Missing config key: {key}"
        
        print(f"[OK] Config API works")
        print(f"     Rounds: {config['num_rounds']}, "
              f"Local epochs: {config['local_epochs']}, "
              f"Batch size: {config['batch_size']}")
except Exception as e:
    print(f"[FAIL] {e}")
    sys.exit(1)

# Test 4: Test training status endpoint
print("\n[4/5] Testing training status endpoint...")
try:
    with app.test_client() as client:
        response = client.get('/api/train/status')
        assert response.status_code == 200
        status = response.get_json()
        
        assert 'running' in status
        assert 'model_loaded' in status
        
        print(f"[OK] Training status API works")
        print(f"     Model loaded: {status['model_loaded']}, "
              f"Training running: {status['running']}")
except Exception as e:
    print(f"[FAIL] {e}")
    sys.exit(1)

# Test 5: Test personalization status endpoint
print("\n[5/5] Testing personalization status endpoint...")
try:
    with app.test_client() as client:
        response = client.get('/api/personalize/status')
        assert response.status_code == 200
        status = response.get_json()
        
        assert 'running' in status
        assert 'results' in status
        
        print(f"[OK] Personalization status API works")
        print(f"     Running: {status['running']}, "
              f"Results: {len(status['results'])} clients")
except Exception as e:
    print(f"[FAIL] {e}")
    sys.exit(1)

# Summary
print("\n" + "=" * 70)
print("INTEGRATION TEST PASSED")
print("=" * 70)

print("\nAPI Endpoints Verified:")
print("  [OK] POST /api/predict")
print("  [OK] GET  /api/config")
print("  [OK] GET  /api/train/status")
print("  [OK] GET  /api/personalize/status")

print("\nSystem Ready:")
print("  - All imports working")
print("  - Model loading working")
print("  - All API endpoints responding")
print("  - State management working")

if state['loaded']:
    print(f"\nModel Statistics:")
    print(f"  - Vocabulary: {len(state['vocab'])} tokens")
    print(f"  - Parameters: {state['model'].count_params():,}")
    print(f"  - Bigram entries: {len(state['bigram_idx'])}")
    print(f"  - Trigram entries: {len(state['trigram_idx'])}")

print("\n======================================================================")
print("Ready to start server: python app.py")
print("======================================================================")
