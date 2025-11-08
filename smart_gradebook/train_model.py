# train_model.py
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

print("✅ Loading dataset...")
data = pd.read_csv("student_data_clean.csv")
print("Shape:", data.shape)
print("Columns:", list(data.columns))

# ---------------------------
# Target and features
# ---------------------------
target = 'risk'  # Column indicating Low Risk / At Risk
if target not in data.columns:
    raise ValueError(f"❌ No target column '{target}' found in dataset!")

# Encode target if it is categorical
if data[target].dtype == 'object':
    le = LabelEncoder()
    data[target] = le.fit_transform(data[target])
    print(f"Target '{target}' encoded.")

# Select features (exclude target and G3)
features = [col for col in data.columns if col not in ['G3', target]]

# Convert categorical columns to numeric
for col in data.select_dtypes(include=['object']).columns:
    if col != target:
        data[col] = data[col].map({"Yes":1, "No":0, "GP":1, "MS":0}).fillna(0)

X = data[features]
y = data[target]

# ---------------------------
# Train/test split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# ---------------------------
# Train Random Forest Classifier
# ---------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("✅ Model trained successfully!")

# ---------------------------
# Save trained model
# ---------------------------
joblib.dump(model, "student_risk_model.pkl")
print("💾 Model saved as 'student_risk_model.pkl'")

# ---------------------------
# Feature importance plot
# ---------------------------
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=True)

plt.figure(figsize=(10,8))
sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
print("📊 Feature importance plot saved as 'feature_importance.png'")

# Optional: show top features
print("Top features by importance:")
print(importance_df.sort_values(by='Importance', ascending=False).head(10))

