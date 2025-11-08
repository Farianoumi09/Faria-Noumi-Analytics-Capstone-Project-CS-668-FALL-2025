import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------
# Load dataset
# -------------------------------
data = pd.read_csv("student_data_clean.csv")
print("✅ Dataset loaded successfully!")
print("Shape:", data.shape)
print("Columns:", data.columns.tolist())

# -------------------------------
# Define target
# -------------------------------
# If 'at_risk' exists, use it. Otherwise, create it from G3.
if "at_risk" in data.columns:
    target = "at_risk"
else:
    # Example rule: G3 < 10 → At Risk (1), else Low Risk (0)
    target = "at_risk"
    data[target] = np.where(data["G3"] < 10, 1, 0)
    print(f"⚠️ 'at_risk' column not found, created synthetic target using G3 < 10.")

# -------------------------------
# Select features
# -------------------------------
features = ["G1", "G2", "studytime", "absences", "famsup"]
X = data[features].copy()
# Convert famsup to numeric if not already
X["famsup"] = X["famsup"].map({"yes":1, "no":0, "Yes":1, "No":0}).fillna(0)
y = data[target]

# -------------------------------
# Train/test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# Train model
# -------------------------------
clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# -------------------------------
# Evaluate model
# -------------------------------
y_pred = clf.predict(X_test)
print("\n✅ Model Training Complete!")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 2))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------------------------------
# Optional: Feature importance plot
# -------------------------------
importance = clf.feature_importances_
importance_df = pd.DataFrame({"Feature": features, "Importance": importance}).sort_values(by="Importance", ascending=True)

plt.figure(figsize=(6,4))
sns.barplot(data=importance_df, x="Importance", y="Feature", palette="viridis")
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
print("📊 Feature importance plot saved as feature_importance.png")

# -------------------------------
# Save model
# -------------------------------
joblib.dump(clf, "student_risk_model.pkl")
print("💾 Model saved successfully as 'student_risk_model.pkl'")
print("🎉 Training complete! You can now run Streamlit using:\n   streamlit run app.py")
