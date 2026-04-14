"""Quick test to verify model loading"""
import os
import keras
from backend.model.architecture import BahdanauAttention
from backend.model.train import MODEL_SAVE_PATH

print(f"Checking model at: {MODEL_SAVE_PATH}")
print(f"File exists: {os.path.exists(MODEL_SAVE_PATH)}")

if os.path.exists(MODEL_SAVE_PATH):
    try:
        model = keras.models.load_model(
            MODEL_SAVE_PATH,
            compile=False,
            custom_objects={'BahdanauAttention': BahdanauAttention}
        )
        print("[OK] Model loaded successfully!")
        print(f"  Model name: {model.name}")
        print(f"  Total params: {model.count_params():,}")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
else:
    print("[ERROR] Model file not found!")
