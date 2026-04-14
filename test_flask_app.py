"""
Quick test to verify Flask app initialization and routes.
"""

import sys

print("Testing Flask app initialization...")

try:
    from app import app
    from backend.state import get_state, load_model_state
    
    print("[OK] Flask app imported successfully")
    
    # Load state (normally done in app.py main)
    load_model_state()
    
    # Check registered blueprints
    blueprints = list(app.blueprints.keys())
    print(f"[OK] Registered blueprints: {blueprints}")
    
    expected = ['predict', 'train', 'audit', 'personalize']
    for bp in expected:
        if bp in blueprints:
            print(f"  [OK] {bp} blueprint registered")
        else:
            print(f"  [FAIL] {bp} blueprint missing")
            sys.exit(1)
    
    # Check routes
    routes = []
    for rule in app.url_map.iter_rules():
        if rule.endpoint != 'static':
            routes.append(f"{rule.rule} [{', '.join(rule.methods - {'HEAD', 'OPTIONS'})}]")
    
    print(f"\n[OK] Available routes ({len(routes)}):")
    for route in sorted(routes):
        print(f"  {route}")
    
    # Check state
    state = get_state()
    print(f"\n[OK] State initialized:")
    print(f"  - Vocab loaded: {state['vocab'] is not None}")
    print(f"  - Bigram index: {state['bigram_idx'] is not None}")
    print(f"  - Trigram index: {state['trigram_idx'] is not None}")
    print(f"  - Model loaded: {state['loaded']}")
    
    if state['vocab']:
        print(f"  - Vocab size: {len(state['vocab'])}")
    
    print("\n[OK] Flask app is ready to run!")
    print("     Start with: python app.py")
    
except Exception as e:
    print(f"[FAIL] Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
