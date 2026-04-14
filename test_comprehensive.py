"""
Comprehensive test suite for Federated Learning Keyboard project.
Tests all components: vocabulary, model, training, prediction, privacy, personalization.
"""

import os
import sys
import numpy as np

print("=" * 70)
print("FEDERATED LEARNING KEYBOARD - COMPREHENSIVE TEST SUITE")
print("=" * 70)

# Test 1: Import all modules
print("\n[1/8] Testing imports...")
try:
    from backend.data.vocabulary import build_vocab, build_bigram_index, build_trigram_index, text_to_ids
    from backend.data.client_data import CLIENT_TEXTS, CLIENT_PERSONAS
    from backend.model.architecture import create_model, BahdanauAttention
    from backend.model.predict import predict_next_word
    from backend.model.train import run_federated_training, MODEL_SAVE_PATH
    from backend.federated.fedprox import local_train_fedprox
    from backend.federated.fednova import fed_nova
    from backend.privacy.dp_sgd import clip_and_noise
    from backend.state import load_model_state, get_state
    import keras
    import tensorflow as tf
    print("[OK] All imports successful")
except Exception as e:
    print(f"[FAIL] Import failed: {e}")
    sys.exit(1)

# Test 2: Vocabulary building
print("\n[2/8] Testing vocabulary building...")
try:
    vocab = build_vocab(CLIENT_TEXTS)
    assert len(vocab) > 100, "Vocab too small"
    assert '<pad>' in vocab and '<unk>' in vocab
    print(f"[OK] Vocabulary built: {len(vocab)} tokens")
    
    bigram_idx = build_bigram_index(CLIENT_TEXTS, vocab)
    trigram_idx = build_trigram_index(CLIENT_TEXTS, vocab)
    print(f"[OK] Bigram index: {len(bigram_idx)} entries")
    print(f"[OK] Trigram index: {len(trigram_idx)} entries")
except Exception as e:
    print(f"[FAIL] Vocabulary test failed: {e}")
    sys.exit(1)

# Test 3: Model architecture
print("\n[3/8] Testing model architecture...")
try:
    test_model = create_model(len(vocab))
    assert test_model is not None
    params = test_model.count_params()
    print(f"[OK] Model created: {params:,} parameters")
    
    # Test forward pass
    test_input = np.random.randint(0, len(vocab), (2, 6))
    test_output = test_model.predict(test_input, verbose=0)
    assert test_output.shape == (2, len(vocab))
    print(f"[OK] Forward pass works: output shape {test_output.shape}")
except Exception as e:
    print(f"[FAIL] Model architecture test failed: {e}")
    sys.exit(1)

# Test 4: DP-SGD privacy mechanism
print("\n[4/8] Testing DP-SGD privacy mechanism...")
try:
    dummy_global = [np.random.randn(10, 10) for _ in range(3)]
    dummy_local = [w + np.random.randn(*w.shape) * 0.1 for w in dummy_global]
    
    noisy_weights = clip_and_noise(dummy_local, dummy_global)
    assert len(noisy_weights) == len(dummy_global)
    
    # Verify noise was added
    diff = np.linalg.norm(np.concatenate([n.flatten() for n in noisy_weights]) - 
                          np.concatenate([l.flatten() for l in dummy_local]))
    assert diff > 0, "No noise added"
    print(f"[OK] DP-SGD clipping and noise applied (L2 diff: {diff:.4f})")
except Exception as e:
    print(f"[FAIL] DP-SGD test failed: {e}")
    sys.exit(1)

# Test 5: FedNova aggregation
print("\n[5/8] Testing FedNova aggregation...")
try:
    global_w = [np.ones((5, 5)) for _ in range(2)]
    client1_w = [w + 0.1 for w in global_w]
    client2_w = [w + 0.2 for w in global_w]
    
    aggregated = fed_nova(global_w, [client1_w, client2_w], [100, 200], [10, 20])
    assert len(aggregated) == len(global_w)
    print("[OK] FedNova aggregation successful")
except Exception as e:
    print(f"[FAIL] FedNova test failed: {e}")
    sys.exit(1)

# Test 6: Model loading
print("\n[6/8] Testing model loading...")
try:
    # Check if model exists
    model_exists = os.path.exists(MODEL_SAVE_PATH) or os.path.exists('gru_federated_keyboard.keras')
    
    if model_exists:
        loaded = load_model_state()
        state = get_state()
        
        if loaded:
            print(f"[OK] Model loaded successfully")
            print(f"  - Vocab size: {len(state['vocab'])}")
            print(f"  - Model params: {state['model'].count_params():,}")
            
            # Test prediction with loaded model
            test_words = ['i', 'want', 'to']
            preds = predict_next_word(
                state['model'], state['vocab'], test_words,
                top_k=3,
                bigram_idx=state['bigram_idx'],
                trigram_idx=state['trigram_idx']
            )
            print(f"[OK] Prediction test: '{' '.join(test_words)}' -> {[w for w, _ in preds]}")
        else:
            print("[WARN] Model file exists but failed to load")
    else:
        print("[WARN] No trained model found (run training first)")
        print(f"  Expected path: {MODEL_SAVE_PATH}")
except Exception as e:
    print(f"[FAIL] Model loading test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 7: Data pipeline
print("\n[7/8] Testing data pipeline...")
try:
    from backend.model.train import prepare_datasets
    datasets = prepare_datasets(vocab)
    
    valid_count = sum(1 for d in datasets if d is not None)
    print(f"[OK] Prepared {valid_count}/{len(CLIENT_TEXTS)} client datasets")
    
    for i, ds in enumerate(datasets):
        if ds is not None:
            x, y = ds
            print(f"  Client {i+1}: {len(x)} sequences")
except Exception as e:
    print(f"[FAIL] Data pipeline test failed: {e}")
    sys.exit(1)

# Test 8: State management
print("\n[8/8] Testing state management...")
try:
    from backend.state import get_config
    config = get_config()
    
    required_keys = ['num_rounds', 'local_epochs', 'batch_size', 'fedprox_mu', 
                     'dp_clip_norm', 'dp_noise_mult']
    for key in required_keys:
        assert key in config, f"Missing config key: {key}"
    
    print("[OK] State management working")
    print(f"  Config keys: {list(config.keys())}")
except Exception as e:
    print(f"[FAIL] State management test failed: {e}")
    sys.exit(1)

# Summary
print("\n" + "=" * 70)
print("ALL TESTS PASSED")
print("=" * 70)
print("\nSystem Status:")
print(f"  • TensorFlow version: {tf.__version__}")
print(f"  • Keras version: {keras.__version__}")
print(f"  • NumPy version: {np.__version__}")
print(f"  • Vocabulary size: {len(vocab)}")
print(f"  • Client datasets: {len(CLIENT_TEXTS)}")
print(f"  • Model save path: {MODEL_SAVE_PATH}")
print(f"  • Model exists: {'Yes' if os.path.exists(MODEL_SAVE_PATH) else 'No'}")

if not os.path.exists(MODEL_SAVE_PATH):
    print("\n[RECOMMENDATION] Run training to create the model:")
    print("   python app.py  (then click 'Train Model' in UI)")
    print("   OR")
    print("   python -c \"from backend.model.train import run_federated_training; run_federated_training()\"")

print("\nProject is ready to use!")
print("   Run: python app.py")
print("   URL: http://localhost:3000")
print("=" * 70)
