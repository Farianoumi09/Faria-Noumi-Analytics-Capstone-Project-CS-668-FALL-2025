import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load cleaned data
data = pd.read_csv("student_data_clean.csv")

# Ensure target exists
if "at_risk" not in data.columns:
    # Example creation of 'at_risk' if missing
    data['at_risk'] = ((data['G1'] < 10) | (data['G2'] < 10) | (data['studytime'] < 2) | (data['famsup']==0)).astype(int)

# Features for prediction
features = ['G1','G2','studytime','absences','famsup']
X = data[features]
y = data['at_risk']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

# Save model
joblib.dump(rf_model, "student_risk_model.pkl")
print("✅ Model saved as student_risk_model.pkl")
