import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Load the dataset
path = os.path.join("Data", "Disease_symptom_and_patient_profile_dataset.csv")
df = pd.read_csv(path)

print(f"--- Training Disease Presence Predictor (Outcome Variable) ---")

# 2. Define Features and Target
# Target is 'Outcome Variable' (Positive/Negative)
X = df.drop(columns=["Outcome Variable"])
y = df["Outcome Variable"].map({"Negative": 0, "Positive": 1})

# 3. Encoding Features
# Binary encoding for symptoms and gender
binary_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
binary_cols = ["Fever", "Cough", "Fatigue", "Difficulty Breathing", "Gender"]

for col in binary_cols:
    X[col] = X[col].map(binary_map)

# Ordinal encoding for clinical indicators
ordinal_map = {"Low": 0, "Normal": 1, "High": 2}
X["Blood Pressure"] = X["Blood Pressure"].map(ordinal_map)
X["Cholesterol Level"] = X["Cholesterol Level"].map(ordinal_map)

# One-Hot Encode Disease
# We keep the column names for the app to match later
X = pd.get_dummies(X, columns=["Disease"], drop_first=False) # Keep all for easier matching

# 4. Scale Age
scaler = StandardScaler()
X["Age"] = scaler.fit_transform(X[["Age"]])

# 5. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

# 6. Model Training
print("Training Random Forest for Presence Prediction...")
rf_presence = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
rf_presence.fit(X_train, y_train)

# 7. Evaluation
score = rf_presence.score(X_test, y_test)
print(f"Presence Prediction Accuracy: {score:.4f}")

# 8. Save everything
joblib.dump(rf_presence, 'rf_presence_model.pkl')
joblib.dump(scaler, 'scaler_presence.pkl')
joblib.dump(X.columns.tolist(), 'feature_columns.pkl') # Save column order

print("Models and transformers saved.")
