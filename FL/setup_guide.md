# Python Environment Setup Guide
## Practical 1: Introduction to Python and Libraries - Supervised Learning (Regression)

### Prerequisites
- Python 3.7 or higher installed on your system
- pip (Python package installer)

### Step 1: Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv ml_env

# Activate virtual environment
# On Windows:
ml_env\Scripts\activate
# On macOS/Linux:
source ml_env/bin/activate
```

### Step 2: Install Required Libraries
```bash
# Install all required packages
pip install -r requirements.txt

# Or install individually:
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

### Step 3: Verify Installation
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
print("All libraries installed successfully!")
```

### Step 4: Run the Practical
```bash
python practical_1_regression.py
```

### Dataset Information
- **File**: `Student_Performance.csv`
- **Size**: 1000 students
- **Features**: 
  - Hours Studied
  - Previous Scores
  - Extracurricular Activities (Yes/No)
  - Sleep Hours
  - Sample Question Papers Practiced
- **Target**: Performance Index (0-100)

### Learning Objectives
1. Set up Python environment with ML libraries
2. Load and explore datasets using Pandas
3. Perform data preprocessing and visualization
4. Build linear regression models with Scikit-Learn
5. Evaluate model performance with metrics
6. Visualize results and interpret findings

### Expected Outputs
- Dataset exploration statistics
- Correlation analysis
- Multiple visualization plots
- Trained linear regression model
- Performance metrics (R², RMSE, MAE)
- Feature importance analysis
- Predictions on new data

### Troubleshooting
- If matplotlib plots don't show, try: `pip install --upgrade matplotlib`
- For Jupyter notebook support: `pip install jupyter ipykernel`
- If import errors occur, ensure virtual environment is activated