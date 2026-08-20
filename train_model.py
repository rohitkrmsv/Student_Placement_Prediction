import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# 1. CREATE A LARGER DATASET
# ============================================================

np.random.seed(42)

n = 1000

data = pd.DataFrame({
    "CGPA": np.round(np.random.uniform(5.0, 9.8, n), 2),
    "Attendance": np.random.randint(55, 101, n),
    "Coding_Score": np.random.randint(30, 101, n),
    "Aptitude_Score": np.random.randint(30, 101, n),
    "Communication_Score": np.random.randint(35, 101, n),
    "Internships": np.random.randint(0, 4, n),
    "Projects": np.random.randint(0, 5, n)
})


# ============================================================
# 2. CREATE PLACEMENT TARGET
# ============================================================

score = (
    data["CGPA"] * 10
    + data["Attendance"] * 0.15
    + data["Coding_Score"] * 0.30
    + data["Aptitude_Score"] * 0.20
    + data["Communication_Score"] * 0.15
    + data["Internships"] * 5
    + data["Projects"] * 4
)

# Add some randomness so the model doesn't get an unrealistically
# perfect relationship
noise = np.random.normal(0, 8, n)

data["Placed"] = ((score + noise) >= 125).astype(int)


# Save generated dataset
data.to_csv("data/students_large.csv", index=False)

print("Dataset created successfully!")
print("Dataset shape:", data.shape)


# ============================================================
# 3. FEATURES AND TARGET
# ============================================================

X = data.drop("Placed", axis=1)
y = data["Placed"]


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 5. CREATE MULTIPLE MODELS
# ============================================================

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    )
}


# ============================================================
# 6. TRAIN AND COMPARE MODELS
# ============================================================

results = {}

print("\n========== MODEL RESULTS ==========")

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    results[name] = accuracy

    print(f"{name}: {accuracy * 100:.2f}%")


# ============================================================
# 7. SELECT BEST MODEL
# ============================================================

best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

print("\n====================================")
print("Best Model:", best_model_name)
print("Accuracy:", f"{results[best_model_name] * 100:.2f}%")
print("====================================")


# ============================================================
# 8. DETAILED EVALUATION
# ============================================================

best_predictions = best_model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, best_predictions))


# ============================================================
# 9. SAVE BEST MODEL
# ============================================================

joblib.dump(best_model, "placement_model.pkl")

print("\nBest model saved as: placement_model.pkl")
print("Project training completed successfully!")