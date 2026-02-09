# Flower Federated Learning Practical Setup

## ✅ Status: All Problems Fixed!

This folder contains a complete, working setup for Flower Federated Learning practicals.

---

## 📁 Files Overview

### Notebooks (Ready to Use)
- **`Flower_intro.ipynb`** - Introduction to Flower FL ✓ Fixed
- **`flower_fl.ipynb`** - FL implementation practical ✓ Fixed

### Python Files
- **`utils2.py`** - Utility functions for FL (models, training, evaluation)
- **`test_flower_setup.py`** - Verify your setup is working
- **`fix_notebooks.py`** - Script that fixed the notebooks

### Configuration
- **`requirements.txt`** - Updated package requirements
- **`requirements_fixed.txt`** - Clean version (recommended)

### Documentation
- **`SETUP_GUIDE.md`** - Complete setup instructions
- **`PROBLEMS_FIXED.md`** - Details of all fixes applied
- **`README.md`** - This file

---

## 🚀 Quick Start

### 1. Verify Setup
```bash
python "Flower Pract Setup/test_flower_setup.py"
```

You should see all checkmarks (✓):
```
✓ Core ML packages
✓ Flower datasets package
✓ Flower client components
✓ Utils2.py functions
✓ SimpleModel created successfully
✓ Dataset transforms configured
```

### 2. Start Jupyter
```bash
jupyter notebook
```

### 3. Open a Notebook
- Start with `Flower_intro.ipynb` for basics
- Then try `flower_fl.ipynb` for full FL implementation

---

## 📦 Installed Packages

### Core FL Framework
- `flwr==1.10.0` - Flower federated learning
- `ray==2.31.0` - Distributed computing
- `flwr-datasets[vision]==0.2.0` - FL datasets

### Machine Learning
- `torch==2.5.0` - PyTorch deep learning
- `torchvision==0.20.0` - Computer vision
- `scikit-learn==1.7.0` - ML algorithms

### Data & Visualization
- `numpy>=1.23,<2.0` - Numerical computing
- `matplotlib==3.10.3` - Plotting
- `seaborn==0.13.2` - Statistical visualization

### Jupyter
- `ipywidgets==8.1.2` - Interactive widgets

### NLP (Optional)
- `transformers==4.42.4` - Transformer models
- `accelerate==0.30.0` - Training acceleration

---

## ✅ What Was Fixed

### 1. Notebook Syntax Errors
- **Problem:** `!pip install` causing syntax errors
- **Fix:** Changed to `%pip install` (Jupyter magic command)
- **Files:** Both `.ipynb` files

### 2. TensorFlow Conflict
- **Problem:** TensorFlow requiring incompatible protobuf version
- **Fix:** Removed TensorFlow (not needed for FL practicals)
- **Result:** All Flower imports work perfectly

### 3. Package Versions
- **Problem:** Incompatible versions for Python 3.12.8
- **Fix:** Updated to compatible versions
- **Result:** All packages install without conflicts

### 4. Import Errors
- **Problem:** Cannot import Flower components
- **Fix:** Resolved protobuf conflict
- **Result:** All imports successful

---

## 🧪 Test Your Setup

### Test 1: Import Flower
```python
import flwr as fl
from flwr.client import NumPyClient
from flwr.server import ServerConfig
from flwr.simulation import start_simulation
print("✓ Flower imports successful!")
```

### Test 2: Create a Model
```python
from utils2 import SimpleModel
model = SimpleModel()
print("✓ Model created!")
```

### Test 3: Load Data
```python
from torchvision import datasets, transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
print(f"✓ Loaded {len(trainset)} training samples!")
```

---

## 📚 Learning Path

1. **Start Here:** `Flower_intro.ipynb`
   - Understand Flower basics
   - Learn client-server architecture
   - Run simple FL example

2. **Next:** `flower_fl.ipynb`
   - Implement full FL workflow
   - Create multiple clients
   - Aggregate models
   - Evaluate global model

3. **Explore:** `utils2.py`
   - Study utility functions
   - Understand model architecture
   - Learn training/evaluation patterns

---

## 🛠️ Troubleshooting

### If imports fail:
```bash
pip uninstall tensorflow tensorflow-intel -y
pip install -r "Flower Pract Setup/requirements_fixed.txt"
```

### If notebooks show errors:
```bash
python "Flower Pract Setup/fix_notebooks.py"
```

### If packages conflict:
```bash
pip install "protobuf>=4.25.2,<5.0.0" "typer[all]>=0.9.0,<0.10.0"
```

---

## 💡 Tips

1. **Use %pip not !pip** in Jupyter notebooks
2. **TensorFlow removed** - not needed for these practicals
3. **Python 3.12.8** - all packages compatible
4. **GPU optional** - CPU works fine for learning

---

## 📖 Additional Resources

- [Flower Documentation](https://flower.dev/docs/)
- [Flower Examples](https://github.com/adap/flower/tree/main/examples)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)

---

## ✨ You're Ready!

All setup issues have been resolved. Your Flower FL environment is fully functional and ready for practicals.

Happy Federated Learning! 🌸
