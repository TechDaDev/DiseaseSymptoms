import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

# 1. Load the dataset (from Step 1)
path = os.path.join("Data", "Disease_symptom_and_patient_profile_dataset.csv")
df = pd.read_csv(path)

print(f"--- Starting Step 2: Feature Encoding & Preprocessing ---")

# 2.1 Separate features and target
X = df.drop(columns=["Outcome Variable"])
y = df["Outcome Variable"]

# Encode target: Negative: 0, Positive: 1
y = y.map({"Negative": 0, "Positive": 1})
print(f"Target encoded. Distribution:\n{y.value_counts()}")

# 2.2 Binary encoding (Symptoms + Gender)
binary_cols = ["Fever", "Cough", "Fatigue", "Difficulty Breathing", "Gender"]
binary_map = {
    "Yes": 1,
    "No": 0,
    "Male": 1,
    "Female": 0
}

for col in binary_cols:
    X[col] = X[col].map(binary_map)

print(f"\nBinary encoding completed for: {binary_cols}")

# 2.3 Ordinal encoding (Clinical indicators)
ordinal_map = {
    "Low": 0,
    "Normal": 1,
    "High": 2
}

X["Blood Pressure"] = X["Blood Pressure"].map(ordinal_map)
X["Cholesterol Level"] = X["Cholesterol Level"].map(ordinal_map)

print(f"Ordinal encoding completed for: Blood Pressure, Cholesterol Level")

# 2.4 One-Hot Encode Disease
# This will create dummy variables for each disease type except the first one (drop_first=True)
X = pd.get_dummies(X, columns=["Disease"], drop_first=True)

print(f"One-Hot encoding completed for 'Disease' column.")

# 2.5 Scale numerical feature (Age only)
scaler = StandardScaler()
X["Age"] = scaler.fit_transform(X[["Age"]])

print(f"Age feature scaled using StandardScaler.")

# 2.6 Final sanity check
print(f"\n--- Final Preprocessing Sanity Check ---")
print(f"Shape of X: {X.shape}")
print(f"Missing values in X: {X.isna().sum().sum()}")
print(f"Data types overview:\n{X.dtypes.value_counts()}")

print("\n--- First 5 rows of processed features (X) ---")
print(X.head())

# Save the processed data for Step 3
# It's good practice to keep them separate for now or use them in memory
# If you want to proceed to modeling, we can keep X and y in memory or save them.
# X.to_csv("X_processed.csv", index=False)
# y.to_csv("y_processed.csv", index=False)
