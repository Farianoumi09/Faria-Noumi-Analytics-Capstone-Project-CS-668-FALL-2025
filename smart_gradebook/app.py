# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from lime.lime_tabular import LimeTabularExplainer
import shap

st.set_page_config(page_title="Smart Gradebook", layout="wide")
st.title("📚 Smart Gradebook")
st.subheader("Welcome back Faria Noumi")

# ---------------------------
# Load model and data
# ---------------------------
@st.cache_data
def load_model():
    return joblib.load("student_risk_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("student_data_clean.csv")

model = load_model()
data = load_data()

st.sidebar.header("Dataset Info")
st.sidebar.write(f"Shape: {data.shape}")
st.sidebar.write("Columns:", list(data.columns))

# ---------------------------
# EDA: G1/G2/G3 histograms
# ---------------------------
st.subheader("📊 Student Grades Distribution")
fig, axes = plt.subplots(1, 3, figsize=(18,5))
for i, col in enumerate(['G1', 'G2', 'G3']):
    sns.histplot(data[col], bins=10, kde=True, ax=axes[i], color='skyblue')
    axes[i].set_title(f"{col} Distribution")
plt.tight_layout()
st.pyplot(fig)

# ---------------------------
# Correlation heatmap
# ---------------------------
st.subheader("📈 Feature Correlations")
numeric_data = data.select_dtypes(include=[np.number])
fig, ax = plt.subplots(figsize=(12,10))
sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
st.pyplot(fig)

# ---------------------------
# Single student prediction
# ---------------------------
st.subheader("👤 Single Student Prediction")
st.write("Enter student info to predict risk:")

categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
input_dict = {}
for col in data.columns:
    if col in ['G3','risk']:  # skip target columns
        continue
    if col in categorical_cols:
        val = st.selectbox(f"{col}", options=data[col].unique())
        input_dict[col] = 1 if val in ["Yes","GP"] else 0
    else:
        val = st.number_input(f"{col}", value=int(data[col].median()))
        input_dict[col] = val

input_data = pd.DataFrame([input_dict])

if st.button("Predict Risk"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    risk_text = "At Risk" if prediction == 1 else "Low Risk"
    st.write(f"**Prediction:** {risk_text}")
    st.write(f"**Probability:** Low Risk = {probability[0]:.2f}, At Risk = {probability[1]:.2f}")

    # ---------------------------
    # SHAP explanation (text)
    # ---------------------------
    st.subheader("🔍 SHAP Feature Contributions (Text)")
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_data)
        if isinstance(shap_values, list):
            shap_vals = shap_values[1][0]
        else:
            shap_vals = shap_values[0]
        for f, v in zip(input_data.columns, shap_vals):
            st.write(f"- {f}: impact = {float(v):.3f}")
    except Exception as e:
        st.write("SHAP explanation not available:", e)

    # ---------------------------
    # LIME explanation (text)
    # ---------------------------
    st.subheader("💡 LIME Feature Description (Text)")
    try:
        numeric_features = input_data.select_dtypes(include=[np.number]).columns.tolist()
        lime_explainer = LimeTabularExplainer(
            input_data[numeric_features].values,
            feature_names=numeric_features,
            class_names=["Low Risk","At Risk"],
            discretize_continuous=True
        )
        exp = lime_explainer.explain_instance(
            input_data[numeric_features].iloc[0].values,
            model.predict_proba,
            num_features=len(numeric_features)
        )
        for feature_desc, weight in exp.as_list():
            st.write(f"- {feature_desc} (weight: {weight:.3f})")
    except Exception as e:
        st.write("LIME explanation not available:", e)

# ---------------------------
# Batch CSV prediction
# ---------------------------
st.subheader("📁 Upload CSV for Batch Predictions")
uploaded_file = st.file_uploader("Choose CSV file", type="csv")
if uploaded_file:
    batch_data = pd.read_csv(uploaded_file)
    st.write("Uploaded CSV shape:", batch_data.shape)

    # Convert categorical features in batch
    for col in batch_data.select_dtypes(include=['object']).columns:
        batch_data[col] = batch_data[col].map({"Yes":1, "No":0, "GP":1, "MS":0}).fillna(0)

    batch_pred = model.predict(batch_data)
    batch_data['Predicted_Risk'] = np.where(batch_pred==1, "At Risk", "Low Risk")
    st.write(batch_data.head())

    st.download_button(
        label="Download Predictions CSV",
        data=batch_data.to_csv(index=False).encode('utf-8'),
        file_name="batch_predictions.csv",
        mime="text/csv"
    )
