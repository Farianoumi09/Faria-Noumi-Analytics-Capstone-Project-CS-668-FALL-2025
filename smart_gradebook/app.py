# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import RandomForestClassifier

# ----------------------
# Load Data & Model
# ----------------------
data = pd.read_csv("student_data_clean.csv")
rf_model = joblib.load("student_risk_model.pkl")

# Encode categorical features for model
categorical_features = ['school','sex','address','famsize','Pstatus','Mjob','Fjob',
                        'reason','guardian','schoolsup','famsup','paid','activities','nursery',
                        'higher','internet','romantic']
data_encoded = pd.get_dummies(data, columns=categorical_features, drop_first=True)

# Features and target
features = ['G1','G2','studytime','absences','famsup']  # simple numeric features
target = 'at_risk'  # ensure this exists in your data

# ----------------------
# Streamlit Page
# ----------------------
st.set_page_config(page_title="Smart Gradebook", page_icon="🎓", layout="wide")
st.title("🎓 Smart Gradebook")
st.markdown("**Welcome back, Faria Noumi!**")

# ----------------------
# Single Student Prediction
# ----------------------
st.header("Single Student Prediction")
col1, col2 = st.columns(2)
with col1:
    G1 = st.slider("G1 (First Period Grade)", 0, 20, 10)
    G2 = st.slider("G2 (Second Period Grade)", 0, 20, 10)
    studytime = st.slider("Study Time (hours/week)", 1, 10, 5)
with col2:
    absences = st.slider("Absences", 0, 50, 5)
    famsup = st.selectbox("Family Support", ["Yes", "No"])
famsup_value = 1 if famsup=="Yes" else 0

input_data = pd.DataFrame({
    "G1":[G1],
    "G2":[G2],
    "studytime":[studytime],
    "absences":[absences],
    "famsup":[famsup_value]
})

pred = rf_model.predict(input_data)[0]
prob = rf_model.predict_proba(input_data)[0][1]

st.subheader("📊 Prediction Result")
if pred==1:
    st.error(f"At Risk (Probability: {prob:.2f})")
else:
    st.success(f"Low Risk (Probability: {prob:.2f})")

# ----------------------
# SHAP Explanation (Single Student)
# ----------------------
st.subheader("🔍 SHAP Feature Contributions (Single Student)")
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(input_data)

# Descriptive text for SHAP instead of plotting
if isinstance(shap_values, list):
    shap_values_to_use = shap_values[1]
else:
    shap_values_to_use = shap_values

for f, val in zip(input_data.columns, shap_values_to_use[0]):
    st.write(f"- {f}: impact = {val:.2f}")

# ----------------------
# LIME Explanation (Single Student)
# ----------------------
st.subheader("💡 LIME Explanation (Top Features)")
lime_explainer = LimeTabularExplainer(
    data[features].values,
    feature_names=features,
    class_names=['Low Risk','At Risk'],
    discretize_continuous=True
)
exp = lime_explainer.explain_instance(input_data.values[0], rf_model.predict_proba, num_features=5)
for feature, impact in exp.as_list():
    st.write(f"- {feature}: {impact:.2f}")

# ----------------------
# Batch Predictions
# ----------------------
st.header("📁 Upload Class CSV for Batch Predictions")
uploaded_file = st.file_uploader("Upload CSV", type="csv")
if uploaded_file:
    class_data = pd.read_csv(uploaded_file)
    missing_cols = [col for col in features if col not in class_data.columns]
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
    else:
        X_class = class_data[features]
        preds = rf_model.predict(X_class)
        class_data['Predicted Risk'] = preds
        st.write(class_data[['G1','G2','studytime','absences','famsup','Predicted Risk']])
        st.bar_chart(class_data['Predicted Risk'].value_counts())

# ----------------------
# EDA: Histograms & Heatmap
# ----------------------
st.header("📊 Data Exploration (EDA)")
st.subheader("Feature Distributions")
fig, ax = plt.subplots(1,3,figsize=(15,4))
sns.histplot(data['G1'], kde=True, ax=ax[0])
ax[0].set_title("G1 Distribution")
sns.histplot(data['G2'], kde=True, ax=ax[1])
ax[1].set_title("G2 Distribution")
sns.histplot(data['absences'], kde=True, ax=ax[2])
ax[2].set_title("Absences Distribution")
st.pyplot(fig)

st.subheader("Correlation Heatmap")
num_data = data[['G1','G2','studytime','absences','G3']]  # numeric only
fig, ax = plt.subplots(figsize=(7,5))
sns.heatmap(num_data.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
st.pyplot(fig)
