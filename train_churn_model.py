
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

# File paths
BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "churn_model_data.csv"

# Load customer data
df = pd.read_csv(DATA_FILE)

print("Dataset loaded successfully!")
print("Total customers:", len(df))
print("\nChurn distribution:")
print(df["Churn"].value_counts())

# Features must come only from the observation period
features = [
    "Recency",
    "Frequency",
    "Monetary",
    "Total_Quantity",
]

X = df[features]
y = df["Churn"]

# Split customers into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Train the model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    min_samples_leaf=5,
)

model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluate performance
print("\n===== CHURN MODEL RESULTS =====")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 4))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Purchased (0)", "No Purchase (1)"],
        zero_division=0,
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance
importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_,
}).sort_values("Importance", ascending=False)

print("\nFeature Importance:")
print(importance.to_string(index=False))

# Save trained model
import joblib

MODEL_FILE = BASE_DIR / "churn_model.pkl"
joblib.dump(model, MODEL_FILE)

print("\nModel saved to:", MODEL_FILE)
