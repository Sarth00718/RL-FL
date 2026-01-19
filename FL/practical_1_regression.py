# Practical 1: Introduction to Python and Libraries - Supervised Learning (Regression)
# Student Performance Dataset Analysis

# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("PRACTICAL 1: SUPERVISED LEARNING - LINEAR REGRESSION")
print("Dataset: Student Performance Analysis")
print("=" * 60)

# Step 1: Load and explore the dataset
print("\n1. LOADING AND EXPLORING THE DATASET")
print("-" * 40)

# Load the dataset
df = pd.read_csv('FL/Student_Performance.csv')

print(f"Dataset shape: {df.shape}")
print(f"Number of students: {df.shape[0]}")
print(f"Number of features: {df.shape[1]}")

print("\nDataset columns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset info:")
print(df.info())

print("\nBasic statistics:")
print(df.describe())

# Step 2: Data preprocessing and exploration
print("\n\n2. DATA PREPROCESSING AND EXPLORATION")
print("-" * 40)

# Check for missing values
print("Missing values:")
print(df.isnull().sum())

# Check data types
print("\nData types:")
print(df.dtypes)

# Handle categorical variable (Extracurricular Activities)
le = LabelEncoder()
df['Extracurricular_Encoded'] = le.fit_transform(df['Extracurricular Activities'])
print("\nExtracurricular Activities encoding:")
print("Yes -> 1, No -> 0")

# Display correlation matrix
print("\nCorrelation with Performance Index:")
correlations = df[['Hours Studied', 'Previous Scores', 'Extracurricular_Encoded', 
                   'Sleep Hours', 'Sample Question Papers Practiced', 'Performance Index']].corr()
print(correlations['Performance Index'].sort_values(ascending=False))

# Step 3: Data visualization
print("\n\n3. DATA VISUALIZATION")
print("-" * 40)

# Set up the plotting style
plt.style.use('default')
fig = plt.figure(figsize=(15, 12))

# 1. Distribution of Performance Index
plt.subplot(2, 3, 1)
plt.hist(df['Performance Index'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Distribution of Performance Index')
plt.xlabel('Performance Index')
plt.ylabel('Frequency')

# 2. Hours Studied vs Performance Index
plt.subplot(2, 3, 2)
plt.scatter(df['Hours Studied'], df['Performance Index'], alpha=0.6, color='green')
plt.title('Hours Studied vs Performance Index')
plt.xlabel('Hours Studied')
plt.ylabel('Performance Index')

# 3. Previous Scores vs Performance Index
plt.subplot(2, 3, 3)
plt.scatter(df['Previous Scores'], df['Performance Index'], alpha=0.6, color='orange')
plt.title('Previous Scores vs Performance Index')
plt.xlabel('Previous Scores')
plt.ylabel('Performance Index')

# 4. Sleep Hours vs Performance Index
plt.subplot(2, 3, 4)
plt.scatter(df['Sleep Hours'], df['Performance Index'], alpha=0.6, color='red')
plt.title('Sleep Hours vs Performance Index')
plt.xlabel('Sleep Hours')
plt.ylabel('Performance Index')

# 5. Sample Papers vs Performance Index
plt.subplot(2, 3, 5)
plt.scatter(df['Sample Question Papers Practiced'], df['Performance Index'], alpha=0.6, color='purple')
plt.title('Sample Papers vs Performance Index')
plt.xlabel('Sample Question Papers Practiced')
plt.ylabel('Performance Index')

# 6. Extracurricular Activities vs Performance Index
plt.subplot(2, 3, 6)
df.boxplot(column='Performance Index', by='Extracurricular Activities', ax=plt.gca())
plt.title('Performance by Extracurricular Activities')
plt.suptitle('')  # Remove default title

plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = df[['Hours Studied', 'Previous Scores', 'Extracurricular_Encoded', 
                        'Sleep Hours', 'Sample Question Papers Practiced', 'Performance Index']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5)
plt.title('Correlation Matrix of Features')
plt.tight_layout()
plt.show()
# Step 4: Prepare data for machine learning
print("\n\n4. PREPARING DATA FOR MACHINE LEARNING")
print("-" * 40)

# Define features (X) and target variable (y)
feature_columns = ['Hours Studied', 'Previous Scores', 'Extracurricular_Encoded', 
                   'Sleep Hours', 'Sample Question Papers Practiced']
X = df[feature_columns]
y = df['Performance Index']

print("Features (X):")
print(X.head())
print(f"\nFeature matrix shape: {X.shape}")

print(f"\nTarget variable (y) shape: {y.shape}")
print(f"Target variable range: {y.min():.1f} to {y.max():.1f}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")
print(f"Training/Testing split: {X_train.shape[0]/(X_train.shape[0]+X_test.shape[0])*100:.1f}% / {X_test.shape[0]/(X_train.shape[0]+X_test.shape[0])*100:.1f}%")

# Step 5: Build and train the linear regression model
print("\n\n5. BUILDING AND TRAINING LINEAR REGRESSION MODEL")
print("-" * 40)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

print("Model trained successfully!")
print(f"Model intercept: {model.intercept_:.4f}")
print("\nModel coefficients:")
for feature, coef in zip(feature_columns, model.coef_):
    print(f"  {feature}: {coef:.4f}")

# Step 6: Make predictions
print("\n\n6. MAKING PREDICTIONS")
print("-" * 40)

# Predict on training and testing sets
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print("Predictions completed!")
print(f"Training predictions range: {y_train_pred.min():.2f} to {y_train_pred.max():.2f}")
print(f"Testing predictions range: {y_test_pred.min():.2f} to {y_test_pred.max():.2f}")

# Display some sample predictions
print("\nSample predictions vs actual values (first 10 test samples):")
comparison_df = pd.DataFrame({
    'Actual': y_test.iloc[:10].values,
    'Predicted': y_test_pred[:10],
    'Difference': y_test.iloc[:10].values - y_test_pred[:10]
})
print(comparison_df.round(2))
# Step 7: Evaluate model performance
print("\n\n7. MODEL PERFORMANCE EVALUATION")
print("-" * 40)

# Calculate evaluation metrics
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("TRAINING SET METRICS:")
print(f"  Mean Squared Error (MSE): {train_mse:.4f}")
print(f"  Root Mean Squared Error (RMSE): {train_rmse:.4f}")
print(f"  Mean Absolute Error (MAE): {train_mae:.4f}")
print(f"  R-squared (R²): {train_r2:.4f}")

print("\nTESTING SET METRICS:")
print(f"  Mean Squared Error (MSE): {test_mse:.4f}")
print(f"  Root Mean Squared Error (RMSE): {test_rmse:.4f}")
print(f"  Mean Absolute Error (MAE): {test_mae:.4f}")
print(f"  R-squared (R²): {test_r2:.4f}")

print(f"\nModel Performance Summary:")
print(f"  The model explains {test_r2*100:.1f}% of the variance in student performance")
print(f"  Average prediction error: ±{test_rmse:.2f} points")

# Step 8: Visualize results
print("\n\n8. VISUALIZING RESULTS")
print("-" * 40)

# Create visualization plots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Actual vs Predicted (Training)
axes[0, 0].scatter(y_train, y_train_pred, alpha=0.6, color='blue')
axes[0, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Actual Performance Index')
axes[0, 0].set_ylabel('Predicted Performance Index')
axes[0, 0].set_title(f'Training Set: Actual vs Predicted\nR² = {train_r2:.4f}')
axes[0, 0].grid(True, alpha=0.3)

# 2. Actual vs Predicted (Testing)
axes[0, 1].scatter(y_test, y_test_pred, alpha=0.6, color='green')
axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 1].set_xlabel('Actual Performance Index')
axes[0, 1].set_ylabel('Predicted Performance Index')
axes[0, 1].set_title(f'Testing Set: Actual vs Predicted\nR² = {test_r2:.4f}')
axes[0, 1].grid(True, alpha=0.3)

# 3. Residuals plot (Training)
train_residuals = y_train - y_train_pred
axes[1, 0].scatter(y_train_pred, train_residuals, alpha=0.6, color='blue')
axes[1, 0].axhline(y=0, color='r', linestyle='--')
axes[1, 0].set_xlabel('Predicted Performance Index')
axes[1, 0].set_ylabel('Residuals')
axes[1, 0].set_title('Training Set: Residuals Plot')
axes[1, 0].grid(True, alpha=0.3)

# 4. Residuals plot (Testing)
test_residuals = y_test - y_test_pred
axes[1, 1].scatter(y_test_pred, test_residuals, alpha=0.6, color='green')
axes[1, 1].axhline(y=0, color='r', linestyle='--')
axes[1, 1].set_xlabel('Predicted Performance Index')
axes[1, 1].set_ylabel('Residuals')
axes[1, 1].set_title('Testing Set: Residuals Plot')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Feature importance visualization
plt.figure(figsize=(10, 6))
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Coefficient': model.coef_,
    'Abs_Coefficient': np.abs(model.coef_)
}).sort_values('Abs_Coefficient', ascending=True)

plt.barh(feature_importance['Feature'], feature_importance['Coefficient'], 
         color=['red' if x < 0 else 'green' for x in feature_importance['Coefficient']])
plt.xlabel('Coefficient Value')
plt.title('Feature Importance (Linear Regression Coefficients)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
# Step 9: Make predictions on new data
print("\n\n9. MAKING PREDICTIONS ON NEW DATA")
print("-" * 40)

# Create sample new student data
new_students = pd.DataFrame({
    'Hours Studied': [6, 8, 3, 9],
    'Previous Scores': [85, 92, 65, 88],
    'Extracurricular_Encoded': [1, 0, 1, 1],  # 1=Yes, 0=No
    'Sleep Hours': [7, 6, 8, 5],
    'Sample Question Papers Practiced': [5, 8, 2, 7]
})

print("New student data:")
print(new_students)

# Make predictions
new_predictions = model.predict(new_students)

print("\nPredicted Performance Index for new students:")
for i, pred in enumerate(new_predictions):
    print(f"Student {i+1}: {pred:.2f}")

# Step 10: Model interpretation and insights
print("\n\n10. MODEL INTERPRETATION AND INSIGHTS")
print("-" * 40)

print("KEY FINDINGS:")
print("1. Feature Impact Analysis:")
for feature, coef in zip(feature_columns, model.coef_):
    impact = "positive" if coef > 0 else "negative"
    print(f"   • {feature}: {coef:.4f} ({impact} impact)")

print(f"\n2. Model Performance:")
print(f"   • R² Score: {test_r2:.4f} ({test_r2*100:.1f}% variance explained)")
print(f"   • RMSE: {test_rmse:.2f} points")
print(f"   • MAE: {test_mae:.2f} points")

print(f"\n3. Most Important Features (by absolute coefficient value):")
importance_ranking = sorted(zip(feature_columns, np.abs(model.coef_)), 
                          key=lambda x: x[1], reverse=True)
for i, (feature, importance) in enumerate(importance_ranking, 1):
    print(f"   {i}. {feature}: {importance:.4f}")

print(f"\n4. Model Equation:")
equation = f"Performance Index = {model.intercept_:.2f}"
for feature, coef in zip(feature_columns, model.coef_):
    sign = "+" if coef >= 0 else ""
    equation += f" {sign}{coef:.2f}*{feature}"
print(f"   {equation}")

print("\n" + "=" * 60)
print("PRACTICAL 1 COMPLETED SUCCESSFULLY!")
print("You have successfully:")
print("• Set up Python environment with required libraries")
print("• Loaded and explored the Student Performance dataset")
print("• Performed data preprocessing and visualization")
print("• Built and trained a Linear Regression model")
print("• Evaluated model performance with multiple metrics")
print("• Visualized results with regression plots")
print("• Made predictions on new data")
print("=" * 60)