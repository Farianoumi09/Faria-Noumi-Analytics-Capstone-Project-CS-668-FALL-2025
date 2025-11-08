import streamlit as st
import pandas as pd
import joblib
import shap

# Load model and data
@st.cache_resource
def load_model():
    return joblib.load("student_risk_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("student_data_clean.csv")

model = load_model()
data = load_data()

st.set_page_config(page_title="Smart Gradebook", page_icon="🎓", layout="centered")
st.title("🎓 Smart Gradebook: Predicting Student Success")
st.write("Predict which students are at risk and see why using Explainable AI (SHAP).")

# Individual student input
st.header("Enter Individual Student Info")
G1 = st.slider("G1 (First Grade)", 0, 20, 10)
G2 = st.slider("G2 (Second Grade)", 0, 20, 10)
studytime = st.slider("Study Time (hours/week)", 1, 10, 5)
absences = st.slider("Absences", 0, 50, 5)
famsup = st.selectbox("Family Support", ["Yes", "No"])
famsup_value = 1 if famsup == "Yes" else 0

input_data = pd.DataFrame({
    "G1": [G1],
    "G2": [G2],
    "studytime": [studytime],
    "absences": [absences],
    "famsup": [famsup_value]
})

# Individual prediction
st.subheader("📊 Individual Prediction")
prediction = model.predict(input_data)[0]
prob = model.predict_proba(input_data)[0][1]

if prediction == 1:
    st.error(f"At Risk (Probability: {prob:.2f})")
else:
    st.success(f"Low Risk (Probability: {prob:.2f})")

# SHAP feature explanation
st.subheader("🔍 Feature Contributions (SHAP)")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(input_data)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot(shap.summary_plot(shap_values, input_data, plot_type="bar"))

# Batch prediction via CSV upload
st.subheader("📁 Upload Class CSV for Batch Predictions")
uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    class_data = pd.read_csv(uploaded_file)
    if 'famsup' in class_data.columns:
        class_data['famsup'] = class_data['famsup'].apply(
            lambda x: 1 if str(x).lower() in ['yes','1','true'] else 0
        )
    preds = model.predict(class_data)
    probs = model.predict_proba(class_data)[:,1]
    class_data['Risk Prediction'] = ["At Risk" if p==1 else "Low Risk" for p in preds]
    class_data['Probability'] = probs.round(2)
    st.dataframe(class_data)
    csv = class_data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions CSV", csv, "class_predictions.csv", "text/csv")
