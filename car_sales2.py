import streamlit as st 
import pickle
import os
import pandas as pd
import numpy as np

# Import necessary libraries from scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
# --- CHANGE 1: Import Random Forest Regressor ---
from sklearn.ensemble import RandomForestRegressor 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

# --- 1. Load Data and Define X (Features) and y (Target) ---
# Assuming the CSV is in a 'dataset' folder relative to the script
file_path = 'dataset/car-sales-extended-missing-data.csv' 
try:
    # Since the dataset is about car sales, I gave the variable as cs!
    cs = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: File not found at {file_path}. Please create the 'dataset' folder and place the CSV inside.")
    exit()

# Define X (Features) and y (Target) based on your chosen columns
X = cs[['Make', 'Colour', 'Odometer (KM)', 'Doors']]
y = cs['Price']

# --- 2. Feature Engineering (Extraction) ---
# Create the binary 'Is_4_Door' feature
X['Is_4_Door'] = np.where(X['Doors'] == 4.0, 1, 0)

# Drop the original 'Doors' column
X = X.drop('Doors', axis=1)

# --- 3. Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- 4. Define Column Sets for Preprocessing ---
numerical_features = ['Odometer (KM)']
categorical_features = ['Make', 'Colour']
binary_features = ['Is_4_Door'] 

# --- 5. Build Preprocessing Pipelines ---

# Numerical Pipeline: KNN Imputation + Standardization
numerical_pipeline = Pipeline([
    ('knn_imputer', KNNImputer(n_neighbors=5)), 
    ('scaler', StandardScaler()) 
])

# Categorical Pipeline: Simple Imputation (Mode) + One-Hot Encoding
categorical_pipeline = Pipeline([
    ('simple_imputer', SimpleImputer(strategy='most_frequent')), 
    ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) 
])

# Binary Pipeline: Simple Imputation (Mode)
# Scaling is unnecessary for binary features (0 or 1)
binary_pipeline = Pipeline([
    ('simple_imputer', SimpleImputer(strategy='most_frequent'))
])

# --- 6. Combine Pipelines with ColumnTransformer ---
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_pipeline, numerical_features),
    ('cat', categorical_pipeline, categorical_features),
    ('bin', binary_pipeline, binary_features)
]) 


# --- 7. Build Final Machine Learning Pipeline ---
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    # --- CHANGE 2: Replace LinearRegression with RandomForestRegressor ---
    # Using default parameters for simplicity
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42)) 
])


# Imputing missing values in y_train before fitting (CRITICAL for regression with NaN target)
y_train_clean = y_train.fillna(y_train.mean())
print("\nTraining model...")

# The pipeline handles all preprocessing on X_train and fits the model
model_pipeline.fit(X_train, y_train_clean) 
print("Model training complete.")

# --- 9. Model Evaluation ---

y_pred = model_pipeline.predict(X_test)

# Since we have not cleaned y_test yet! We impute y_test with its mean so all rows can be compared for evaluation.
y_test_clean = y_test.fillna(y_test.mean())

mae = mean_absolute_error(y_test_clean, y_pred)
r2 = r2_score(y_test_clean, y_pred)




# --- 10. Model Persistence (Saving for Streamlit App) ---

# Update the save file name to reflect the new model type
model_save_path = 'random_forest_pipeline.joblib' 
with open(model_save_path, 'wb') as f:
    pickle.dump(model_pipeline, f)

print(f"\nPipeline successfully saved to '{model_save_path}' using pickle.")