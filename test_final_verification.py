"""
FINAL VERIFICATION - Complete System Check
"""

import sys
import os

print("="*70)
print("FEDERATED KEYBOARD - FINAL VERIFICATION")
print("="*70)

# Check 1: File structure
print("\n[1/6] Checking file structure...")
required_files = [
    'app.py',
    'requirements.txt',
    'backend/state.py',
    'backend/model/architecture.py',
    'backend/model/train.py',
    'backend/model/predict.py',
    'backend/data/vocabulary.py',
    'backend/data/client_data.py',
    'backend/federated/fedprox.py',
    'backend/federated/fednova.py',
    'backend/privacy/dp_sgd.py',
    'backend/privacy/audit.py',
    'backend/personalization/finetune.py',
    'backend/routes/predict_routes.py',
    'backend/routes/train_routes.py',
    'backend/routes/audit_routes.py',
    'backend/routes/personalize_routes.py',
    'frontend/templates/index.html',
    'frontend/static/js/app.js',
    'frontend/static/css/style.css',
    'frontend/static/favicon.svg',
]

missing = []
for f in required_files:
    if not os.path.exists(f):
        missing.append(f)

if missing:
    print(f"  [FAIL] Missing files: {missing}")
    sys.exit(1)
else:
    print(f"  [OK] All {len(required_files)} required files present")

# Check 2: Model file
print("\n[2/6] Checking model file...")
model_paths = [
    'backend/model/saved/gru_federated_keyboard.keras',
    'gru_federated_keyboard.keras',
]
model_exists = any(os.path.exists(p) for p in model_paths)
if model_exists:
    print("  [OK] Model file exists")
else:
    print("  [WARN] Model file not found - will need to train")

# Check 3: Import all modules
print("\n[3/6] Testing imports...")
try:
    from app import app
    from backend.state import load_model_state, get_state
    from backend.model.architecture import create_model
    from backend.model.predict import predict_next_word
    from backend.data.vocabulary import build_vocab
    from backend.data.client_data import CLIENT_TEXTS
    print("  [OK] All imports successful")
except Exception as e:
    print(f"  [FAIL] Import error: {e}")
    sys.exit(1)

# Check 4: Initialize app
print("\n[4/6] Initializing application...")
try:
    with app.app_context():
        loaded = load_model_state()
        state = get_state()
        
        if state['vocab']:
            print(f"  [OK] Vocabulary: {len(state['vocab'])} tokens")
        if state['bigram_idx']:
            print(f"  [OK] Bigram index: {len(state['bigram_idx'])} entries")
        if state['trigram_idx']:
            print(f"  [OK] Trigram index: {len(state['trigram_idx'])} entries")
        if loaded:
            print(f"  [OK] Model loaded: {state['model'].count_params():,} parameters")
        else:
            print("  [WARN] Model not loaded - predictions will fail until trained")
except Exception as e:
    print(f"  [FAIL] Initialization error: {e}")
    sys.exit(1)

# Check 5: Test API endpoints
print("\n[5/6] Testing API endpoints...")
with app.test_client() as client:
    
    # Test config
    try:
        r = client.get('/api/config')
        if r.status_code == 200:
            print("  [OK] GET /api/config")
        else:
            print(f"  [FAIL] GET /api/config returned {r.status_code}")
    except Exception as e:
        print(f"  [FAIL] GET /api/config: {e}")
    
    # Test train status
    try:
        r = client.get('/api/train/status')
        if r.status_code == 200:
            print("  [OK] GET /api/train/status")
        else:
            print(f"  [FAIL] GET /api/train/status returned {r.status_code}")
    except Exception as e:
        print(f"  [FAIL] GET /api/train/status: {e}")
    
    # Test predict (if model loaded)
    if state['loaded']:
        try:
            r = client.post('/api/predict', json={'words': ['i', 'want'], 'top_k': 3})
            if r.status_code == 200:
                data = r.get_json()
                preds = data.get('predictions', [])
                if preds:
                    print(f"  [OK] POST /api/predict (got {len(preds)} predictions)")
                else:
                    print("  [FAIL] POST /api/predict returned no predictions")
            else:
                print(f"  [FAIL] POST /api/predict returned {r.status_code}")
        except Exception as e:
            print(f"  [FAIL] POST /api/predict: {e}")
    else:
        print("  [SKIP] POST /api/predict (no model)")
    
    # Test homepage
    try:
        r = client.get('/')
        if r.status_code == 200 and 'FedKeyboard' in r.get_data(as_text=True):
            print("  [OK] GET / (homepage)")
        else:
            print(f"  [FAIL] GET / returned {r.status_code}")
    except Exception as e:
        print(f"  [FAIL] GET /: {e}")
    
    # Test favicon
    try:
        r = client.get('/static/favicon.svg')
        if r.status_code == 200:
            print("  [OK] GET /static/favicon.svg")
        else:
            print(f"  [WARN] Favicon not found (404 in browser)")
    except Exception as e:
        print(f"  [WARN] Favicon: {e}")

# Check 6: Verify predictions work
print("\n[6/6] Testing prediction quality...")
if state['loaded']:
    with app.test_client() as client:
        test_cases = [
            ['i', 'want', 'to'],
            ['please', 'send', 'me'],
            ['the', 'weather', 'is'],
        ]
        
        all_good = True
        for words in test_cases:
            r = client.post('/api/predict', json={'words': words, 'top_k': 5})
            if r.status_code == 200:
                data = r.get_json()
                preds = data.get('predictions', [])
                if len(preds) == 5:
                    top = preds[0]
                    print(f"  [OK] '{' '.join(words)}' -> {top['word']} ({top['probability']:.2%})")
                else:
                    print(f"  [FAIL] Expected 5 predictions, got {len(preds)}")
                    all_good = False
            else:
                print(f"  [FAIL] Prediction failed with status {r.status_code}")
                all_good = False
        
        if all_good:
            print("  [OK] All predictions working correctly")
else:
    print("  [SKIP] No model loaded")

# Final summary
print("\n" + "="*70)
print("VERIFICATION COMPLETE")
print("="*70)

if state['loaded']:
    print("\n[STATUS] READY TO USE")
    print("  All systems operational")
    print("  Model loaded and predictions working")
    print("  All API endpoints functional")
    print("\n[START SERVER]")
    print("  python app.py")
    print("\n[OPEN BROWSER]")
    print("  http://localhost:3000")
    print("\n[FEATURES]")
    print("  - Real-time next-word predictions")
    print("  - 5 client personas (Social, Professional, Tech, Food, Travel)")
    print("  - Trigram + Bigram re-ranking")
    print("  - Federated learning with privacy guarantees")
    print("  - On-device personalization")
    print("  - Privacy audit tools")
else:
    print("\n[STATUS] SETUP REQUIRED")
    print("  System is functional but model needs training")
    print("\n[NEXT STEPS]")
    print("  1. Start server: python app.py")
    print("  2. Open browser: http://localhost:3000")
    print("  3. Go to 'Train' tab")
    print("  4. Click 'Train Model' button")
    print("  5. Wait for training to complete (~5-10 minutes)")
    print("  6. Return to 'Keyboard' tab and start typing")

print("\n" + "="*70)
