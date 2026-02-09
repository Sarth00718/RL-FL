"""
Test script to verify Flower setup without TensorFlow interference
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

print("Testing Flower FL setup...")
print("-" * 50)

# Test 1: Core packages
try:
    import torch
    import torchvision
    import numpy as np
    import matplotlib.pyplot as plt
    import sklearn
    print("✓ Core ML packages (torch, torchvision, numpy, matplotlib, sklearn)")
except Exception as e:
    print(f"✗ Core ML packages failed: {e}")

# Test 2: Flower datasets
try:
    import flwr_datasets
    print("✓ Flower datasets package")
except Exception as e:
    print(f"✗ Flower datasets failed: {e}")

# Test 3: Flower client (without server components that need TensorFlow)
try:
    from flwr.client import NumPyClient
    from flwr.common import NDArrays, Scalar
    print("✓ Flower client components")
except Exception as e:
    print(f"✗ Flower client failed: {e}")

# Test 4: Utils file
try:
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from utils2 import SimpleModel, train_model, evaluate_model
    print("✓ Utils2.py functions")
except Exception as e:
    print(f"✗ Utils2.py failed: {e}")

# Test 5: Create a simple model
try:
    model = SimpleModel()
    print(f"✓ SimpleModel created successfully")
except Exception as e:
    print(f"✗ SimpleModel creation failed: {e}")

# Test 6: Load MNIST dataset
try:
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    # Just test if we can create the dataset object (don't download)
    print("✓ Dataset transforms configured")
except Exception as e:
    print(f"✗ Dataset setup failed: {e}")

print("-" * 50)
print("Flower FL setup test completed!")
print("\nNote: TensorFlow-based Flower server components are skipped")
print("For FL practicals, use simulation mode or client-only code")
