import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.preprocessing import StandardScaler

# --- STEP 1 & 2 RECAP (to get feature names) ---
path = os.path.join("Data", "Disease_symptom_and_patient_profile_dataset.csv")
df = pd.read_csv(path)

X = df.drop(columns=["Outcome Variable"])
binary_cols = ["Fever", "Cough", "Fatigue", "Difficulty Breathing", "Gender"]
binary_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
for col in binary_cols:
    X[col] = X[col].map(binary_map)

ordinal_map = {"Low": 0, "Normal": 1, "High": 2}
X["Blood Pressure"] = X["Blood Pressure"].map(ordinal_map)
X["Cholesterol Level"] = X["Cholesterol Level"].map(ordinal_map)

X = pd.get_dummies(X, columns=["Disease"], drop_first=True)
feature_names = X.columns

# --- LOAD MODEL ---
rf = joblib.load('random_forest_model.pkl')

print("--- Starting Step 4: Model Explainability & Clinical Interpretation ---")

# 4.1 Extract feature importance
importances = rf.feature_importances_
feature_importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\n--- Top 15 Features (Overall) ---")
print(feature_importance_df.head(15))

# 4.2 Focus on medically meaningful features (Non-Disease columns)
important_features = feature_importance_df[
    ~feature_importance_df["Feature"].str.startswith("Disease_")
]

print("\n--- Top Medically Meaningful Features ---")
print(important_features)

# 4.3 Visualize top features
top_features = important_features.head(10)

plt.figure(figsize=(10, 6))
plt.barh(top_features["Feature"], top_features["Importance"], color='skyblue')
plt.gca().invert_yaxis()
plt.title("Top Medical Features Influencing Disease Outcome (Random Forest)")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig('feature_importance.png')
print("\nFeature importance plot saved as 'feature_importance.png'")

# Explicitly print the values for the final report
for index, row in top_features.iterrows():
    print(f"{row['Feature']}: {row['Importance']:.4f}")
