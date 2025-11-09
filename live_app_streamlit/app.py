import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import shap
from lime.lime_tabular import LimeTabularExplainer

# --- Page Config ---
st.set_page_config(page_title="Smart AI Gradebook", layout="wide")
st.title("🎓 Smart AI Gradebook Dashboard")
st.write("By: Faria Noumi")

# --- Simple Login ---
USER_CREDENTIALS = {"teacher": "password123", "admin": "admin123"}

if "login" not in st.session_state:
    st.session_state.login = False

if not st.session_state.login:
    st.subheader("Login to access the dashboard")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_btn = st.button("Login")
    if login_btn:
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state.login = True
            st.success(f"Logged in as {username}")
        else:
            st.error("Invalid username or password")
    st.stop()  # Stop execution until login

else:
    logout_btn = st.button("Logout")
    if logout_btn:
        st.session_state.login = False
        st.experimental_rerun()

# --- Main App ---
uploaded_file = st.file_uploader("Upload your student CSV", type=["csv"])
if uploaded_file:
    # Load data
    df = pd.read_csv(uploaded_file, sep=';')
    df.columns = df.columns.str.strip()

    # Encode categorical columns
    cat_cols = df.select_dtypes(include='object').columns
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    # Numeric columns
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    # --- Interactive EDA Tabs ---
    tab_overview, tab_hist, tab_box, tab_corr, tab_pair = st.tabs([
        "Overview & Descriptive Stats", "Histograms", "Boxplots", "Correlation", "Pairplot"
    ])

    # Overview
    with tab_overview:
        st.subheader("Dataset Overview")
        st.write(f"Shape: {df.shape}")
        st.write("Missing Values:")
        st.dataframe(df.isnull().sum())
        st.write("Descriptive Statistics:")
        st.dataframe(df.describe())

        st.subheader("Initial Risk Verdict")
        if all(c in df.columns for c in ['G1','G2']):
            df['Initial_Verdict'] = np.where((df['G1']<0) | (df['G2']<0), 'Potential At-Risk','Low Risk')
            st.dataframe(df[['G1','G2','Initial_Verdict']].head(10))
        else:
            st.warning("G1 or G2 not found for verdict")

    # Histograms
    with tab_hist:
        selected_cols = st.multiselect("Select numeric columns for histogram", num_cols, default=num_cols)
        for col in selected_cols:
            with st.expander(f"Histogram: {col}"):
                fig, ax = plt.subplots()
                sns.histplot(df[col], bins=15, kde=True, color='skyblue', ax=ax)
                ax.set_title(f"Histogram: {col}")
                st.pyplot(fig)
                plt.close(fig)

    # Boxplots
    with tab_box:
        selected_cols = st.multiselect("Select numeric columns for boxplot", num_cols, default=num_cols)
        for col in selected_cols:
            with st.expander(f"Boxplot: {col}"):
                fig, ax = plt.subplots()
                sns.boxplot(x=df[col], color='lightgreen', ax=ax)
                ax.set_title(f"Boxplot: {col}")
                st.pyplot(fig)
                plt.close(fig)

    # Correlation Heatmap
    with tab_corr:
        st.subheader("Correlation Heatmap")
        with st.expander("Show correlation heatmap"):
            numeric_df = df.select_dtypes(include=np.number)
            fig, ax = plt.subplots(figsize=(10,6))
            sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap='viridis', linewidths=0.5, ax=ax)
            st.pyplot(fig)
            plt.close(fig)

    # Pairplot
    with tab_pair:
        st.subheader("Pairplot")
        grade_cols = ['G1','G2','G3','studytime','failures','absences']
        if all(c in df.columns for c in grade_cols):
            with st.expander("Show pairplot"):
                fig = sns.pairplot(df[grade_cols], diag_kind='kde')
                st.pyplot(fig)
        else:
            st.warning("Required columns for pairplot not found")

    # --- Define At-Risk Students ---
    df['low_early_grade'] = ((df['G1'] < 0) | (df['G2'] < 0)).astype(int)
    df['low_studytime'] = (df['studytime'] < 0).astype(int)
    df['no_famsup'] = (df['famsup'] == 0).astype(int)
    df['high_absences'] = (df['absences'] > 0.5).astype(int)
    df['at_risk'] = ((df['low_early_grade'] + df['low_studytime'] +
                      df['no_famsup'] + df['high_absences']) >= 2).astype(int)

    st.subheader("At-Risk vs Low-Risk Students")
    fig, ax = plt.subplots()
    sns.countplot(x='at_risk', data=df, palette='Reds', ax=ax)
    st.pyplot(fig)
    plt.close(fig)

    # --- ML Model ---
    y = df['at_risk']
    X = df.select_dtypes(include=np.number).drop(
        ['G3','at_risk','low_early_grade','low_studytime','no_famsup','high_absences'], axis=1, errors='ignore'
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    st.subheader("Random Forest Model Performance")
    st.text(classification_report(y_test, y_pred))
    st.metric("AUC-ROC", round(roc_auc_score(y_test, y_pred), 3))

    # --- SHAP (Descriptive Safe) ---
    st.subheader("SHAP Feature Importance (Descriptive)")
    try:
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_test)
        if isinstance(shap_values, list):
            shap_values_to_plot = shap_values[1]
        else:
            shap_values_to_plot = shap_values
        shap_mean = np.abs(shap_values_to_plot).mean(axis=0)
        shap_df = pd.DataFrame({'Feature': X_test.columns, 'Mean_ABS_SHAP': shap_mean})
        shap_df = shap_df.sort_values(by='Mean_ABS_SHAP', ascending=True)
        fig, ax = plt.subplots(figsize=(10,6))
        ax.barh(shap_df['Feature'], shap_df['Mean_ABS_SHAP'], color='skyblue')
        ax.set_title("Feature Importance based on SHAP values")
        ax.set_xlabel("Mean |SHAP value|")
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.error(f"SHAP failed: {e}")

    # --- LIME ---
    st.subheader("LIME Explanation")
    lime_explainer = LimeTabularExplainer(X_train.values, feature_names=X_train.columns,
                                          class_names=['Low Risk','At Risk'], discretize_continuous=True)
    idx = st.number_input("Select student index for LIME", 0, len(X_test)-1, 0)
    exp = lime_explainer.explain_instance(X_test.iloc[idx].values, rf.predict_proba, num_features=5)
    st.components.v1.html(exp.as_html(), height=600, scrolling=True)

    # --- Teacher Smart Gradebook ---
    st.subheader("Teacher Smart Gradebook")
    teacher_view = df[['G1','G2','G3','studytime','famsup','absences','at_risk']].copy()
    teacher_view['Risk_Level'] = teacher_view['at_risk'].apply(lambda x: 'High' if x==1 else 'Low')

    def intervention(row):
        tips = []
        if row['studytime'] < 0: tips.append("Increase study time")
        if row['famsup'] == 0: tips.append("Engage family support")
        if row['absences'] > 0.5: tips.append("Monitor attendance")
        if row['G1'] < 0 or row['G2'] < 0: tips.append("Provide tutoring")
        return ", ".join(tips) if tips else "None"

    teacher_view['Suggested_Intervention'] = teacher_view.apply(intervention, axis=1)

    def future_support(row):
        support = []
        if row['G1'] < 0 or row['G2'] < 0: support.append("Extra Tutoring")
        if row['studytime'] < 0: support.append("Study Plan / Time Management")
        if row['absences'] > 0.5: support.append("Attendance Monitoring")
        if row['famsup'] == 0: support.append("Engage Family Support")
        return ", ".join(support) if support else "None"

    teacher_view['Future_Support'] = teacher_view.apply(future_support, axis=1)
    st.dataframe(teacher_view.head(10))

    # Student Risk vs Grades
    st.subheader("Student Risk Comparison by Grades")
    fig, ax = plt.subplots(figsize=(8,5))
    sns.scatterplot(data=teacher_view, x='G3', y='G2', hue='Risk_Level', style='Risk_Level', s=100,
                    palette={'High':'red','Low':'green'}, ax=ax)
    ax.set_title("Student Risk vs Grades")
    ax.set_xlabel("Final Grade (G3)")
    ax.set_ylabel("Second Grade (G2)")
    st.pyplot(fig)
    plt.close(fig)

    # Download CSV
    csv = teacher_view.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Smart Gradebook CSV", csv, "smart_gradebook.csv", "text/csv")

else:
    st.info("Upload the student CSV to start analysis.")
