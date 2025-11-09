import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv("student-mat.csv", sep=';')
df.columns = df.columns.str.strip()

# -------------------------------
# Encode categorical columns
# -------------------------------
cat_cols = df.select_dtypes(include='object').columns
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# -------------------------------
# Scale numeric columns
# -------------------------------
num_cols = ['studytime', 'failures', 'absences', 'G1', 'G2']
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# -------------------------------
# Define 'at_risk' label
# -------------------------------
df['low_early_grade'] = ((df['G1'] < 0) | (df['G2'] < 0)).astype(int)
df['low_studytime'] = (df['studytime'] < 0).astype(int)
df['no_famsup'] = (df['famsup'] == 0).astype(int)
df['high_absences'] = (df['absences'] > 0.5).astype(int)

df['at_risk'] = ((df['low_early_grade'] + df['low_studytime'] + df['no_famsup'] + df['high_absences']) >= 2).astype(int)

# -------------------------------
# Prepare X and y
# -------------------------------
X = df.drop(['G3','at_risk','low_early_grade','low_studytime','no_famsup','high_absences'], axis=1)
y = df['at_risk']

# -------------------------------
# Train-test split and model
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

# -------------------------------
# Evaluate
# -------------------------------
y_pred = rf_model.predict(X_test)
print(classification_report(y_test, y_pred))
print("AUC-ROC:", roc_auc_score(y_test, y_pred))

# -------------------------------
# Save model and cleaned data
# -------------------------------
joblib.dump(rf_model, "student_risk_model.pkl")

df.to_csv("student_data_clean.csv", index=False)

print("✅ Model and cleaned data saved successfully.")
