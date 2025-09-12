#!/usr/bin/env python3
"""
Test script to diagnose model loading issues
"""
import tensorflow as tf
from pathlib import Path
import sys

print(f"TensorFlow version: {tf.__version__}")
print(f"Python version: {sys.version}")

# Test loading one model file
model_path = Path("models/16s_genus_classifier.keras")
print(f"Model path exists: {model_path.exists()}")
print(f"Model file size: {model_path.stat().st_size if model_path.exists() else 'N/A'} bytes")

if model_path.exists():
    try:
        print("Attempting to load model...")
        model = tf.keras.models.load_model(str(model_path), compile=False)
        print("✅ Model loaded successfully!")
        print(f"Model summary: {model.summary()}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print(f"Error type: {type(e).__name__}")
        
        # Try alternative loading methods
        try:
            print("Trying with custom_objects={}...")
            model = tf.keras.models.load_model(str(model_path), compile=False, custom_objects={})
            print("✅ Model loaded with custom_objects!")
        except Exception as e2:
            print(f"❌ Still failed: {e2}")
            
            # Check if it's actually an HDF5 file that needs .h5 extension
            try:
                print("Trying to load as HDF5...")
                import h5py
                with h5py.File(str(model_path), 'r') as f:
                    print(f"HDF5 keys: {list(f.keys())}")
            except Exception as e3:
                print(f"❌ HDF5 check failed: {e3}")
