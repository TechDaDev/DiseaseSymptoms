import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score

# --- STEP 1 & 2: LOAD AND PREPROCESS ---
path = os.path.join("Data", "Disease_symptom_and_patient_profile_dataset.csv")
df = pd.read_csv(path)

X = df.drop(columns=["Outcome Variable"])
y = df["Outcome Variable"].map({"Negative": 0, "Positive": 1})

binary_cols = ["Fever", "Cough", "Fatigue", "Difficulty Breathing", "Gender"]
binary_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
for col in binary_cols:
    X[col] = X[col].map(binary_map)

ordinal_map = {"Low": 0, "Normal": 1, "High": 2}
X["Blood Pressure"] = X["Blood Pressure"].map(ordinal_map)
X["Cholesterol Level"] = X["Cholesterol Level"].map(ordinal_map)

X = pd.get_dummies(X, columns=["Disease"], drop_first=True)

scaler = StandardScaler()
X["Age"] = scaler.fit_transform(X[["Age"]])

# --- STEP 3: BASELINE PREDICTIVE MODELS ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

results = []

def evaluate_model(model, name, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return f"--- {name} ---\nAccuracy: {acc:.4f}\n{report}\n"

# Model 1
logreg = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")
logreg.fit(X_train, y_train)
results.append(evaluate_model(logreg, "Logistic Regression", X_test, y_test))

# Model 2
rf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
rf.fit(X_train, y_train)
results.append(evaluate_model(rf, "Random Forest", X_test, y_test))

# Model 3
gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train, y_train)
results.append(evaluate_model(gb, "Gradient Boosting", X_test, y_test))

# Output to console and file
final_output = "\n".join(results)
print(final_output)

with open("model_results.txt", "w") as f:
    f.write(final_output)

import joblib
joblib.dump(rf, 'random_forest_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
