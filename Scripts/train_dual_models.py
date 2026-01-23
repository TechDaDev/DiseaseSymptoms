import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Load the dataset
path = os.path.join(os.getcwd(), "Data", "Disease_symptom_and_patient_profile_dataset.csv")
if not os.path.exists(path):
    path = os.path.join("Data", "Disease_symptom_and_patient_profile_dataset.csv")
df = pd.read_csv(path)

print("--- Step 1: Preprocessing for Multi-Target Prediction ---")

# FEATURES (X): Everything EXCEPT Disease and Outcome Variable
# as requested: "Outcome Variable is not with the features"
X_df = df.drop(columns=["Disease", "Outcome Variable"])

# TARGETS (y)
y_disease = df["Disease"]
y_outcome = df["Outcome Variable"].map({"Negative": 0, "Positive": 1})

# Encoding Features (Binary & Ordinal)
binary_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
binary_cols = ["Fever", "Cough", "Fatigue", "Difficulty Breathing", "Gender"]

for col in binary_cols:
    X_df[col] = X_df[col].map(binary_map)

ordinal_map = {"Low": 0, "Normal": 1, "High": 2}
X_df["Blood Pressure"] = X_df["Blood Pressure"].map(ordinal_map)
X_df["Cholesterol Level"] = X_df["Cholesterol Level"].map(ordinal_map)

# Encoding Target Disease
le_disease = LabelEncoder()
y_disease_encoded = le_disease.fit_transform(y_disease)

# Scaling Age
scaler = StandardScaler()
X_df["Age"] = scaler.fit_transform(X_df[["Age"]])

# 2. Train/Test Splits
# Split for Disease
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
    X_df, y_disease_encoded, test_size=0.2, random_state=42
)

# Split for Outcome (Presence)
X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(
    X_df, y_outcome, test_size=0.2, random_state=42, stratify=y_outcome
)

# 3. Training Models
print("Training Disease Classifier...")
rf_disease = RandomForestClassifier(n_estimators=500, random_state=42, class_weight="balanced")
rf_disease.fit(X_train_d, y_train_d)

print("Training Outcome/Presence Classifier...")
rf_outcome = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
rf_outcome.fit(X_train_o, y_train_o)

# 4. Evaluation
print(f"Disease Model Accuracy: {rf_disease.score(X_test_d, y_test_d):.4f}")
print(f"Outcome Model Accuracy: {rf_outcome.score(X_test_o, y_test_o):.4f}")

# 5. Save everything
model_dir = "Models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

joblib.dump(rf_disease, os.path.join(model_dir, 'rf_disease_only.pkl'))
joblib.dump(rf_outcome, os.path.join(model_dir, 'rf_outcome_only.pkl'))
joblib.dump(le_disease, os.path.join(model_dir, 'label_encoder_disease.pkl'))
joblib.dump(scaler, os.path.join(model_dir, 'scaler_global.pkl'))

print(f"All models and transformers saved to {model_dir}")
