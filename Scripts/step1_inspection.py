import pandas as pd
import os

# Set dataset path
path = os.path.join("Data", "Disease_symptom_and_patient_profile_dataset.csv")

print(f"--- Loading Dataset from {path} ---")
df = pd.read_csv(path)

print(f"\nShape of dataset: {df.shape}")
print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Basic Info & Data Types ---")
print(df.info())

print("\n--- Unique Values per Column ---")
for col in df.columns:
    print(f"\n{col}:")
    print(df[col].unique())

print("\n--- Missing Values Check ---")
print(df.isna().sum())

print("\n--- Target Distribution (Outcome Variable) ---")
print(df["Outcome Variable"].value_counts(normalize=True))
print(df["Outcome Variable"].value_counts())
