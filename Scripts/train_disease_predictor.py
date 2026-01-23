import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Load the dataset
path = os.path.join("Data", "Disease_symptom_and_patient_profile_dataset.csv")
df = pd.read_csv(path)

print(f"--- Predicting Disease (Multi-class Classification) ---")

# 2. Define Features and Target
# Now 'Disease' is the target (y), and 'Outcome Variable' is a feature (X)
X = df.drop(columns=["Disease"])
y = df["Disease"]

# 3. Encoding Features
# Binary encoding for symptoms, gender, and outcome
binary_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0, "Positive": 1, "Negative": 0}
binary_cols = ["Fever", "Cough", "Fatigue", "Difficulty Breathing", "Gender", "Outcome Variable"]

for col in binary_cols:
    X[col] = X[col].map(binary_map)

# Ordinal encoding for clinical indicators
ordinal_map = {"Low": 0, "Normal": 1, "High": 2}
X["Blood Pressure"] = X["Blood Pressure"].map(ordinal_map)
X["Cholesterol Level"] = X["Cholesterol Level"].map(ordinal_map)

# 4. Scale Age
scaler = StandardScaler()
X["Age"] = scaler.fit_transform(X[["Age"]])

# 5. Encode Target (Disease)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 6. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, 
    test_size=0.2, 
    random_state=42, 
    stratify=None # Stratify might fail due to single-sample classes
)

# 7. Model Training
print("Training Random Forest for Disease Prediction...")
rf_disease = RandomForestClassifier(n_estimators=500, random_state=42)
rf_disease.fit(X_train, y_train)

# 8. Evaluation
y_pred = rf_disease.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Disease Prediction Accuracy: {acc:.4f}")

# 9. Save everything for the app
joblib.dump(rf_disease, 'rf_disease_model.pkl')
joblib.dump(le, 'disease_label_encoder.pkl')
joblib.dump(scaler, 'scaler_disease.pkl')

print("Models and transformers saved.")
