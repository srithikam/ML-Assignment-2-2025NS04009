import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

st.set_page_config(page_title="ML Classification Dashboard", layout="wide")

st.title("Machine Learning Classification Dashboard")
st.write("BITS Pilani WILP - Machine Learning Assignment 2")

# Feature a: CSV Data Upload Option
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload test_data.csv", type=["csv"])

# Feature b: Model Selection Dropdown
st.sidebar.header("2. Select Model")
model_option = st.sidebar.selectbox(
    "Choose a Classification Model:",
    [
        "Logistic Regression",
        "Decision Tree",
        "kNN",
        "Naive Bayes",
        "Support Vector Machine"
    ]
)

# Mapping friendly model names to saved model filenames
model_file_map = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Support Vector Machine": "support_vector_machine.pkl"
}

if uploaded_file is not None:
    # Read test dataset
    data = pd.read_csv(uploaded_file)
    data.columns = data.columns.str.strip()
    
    st.subheader("Uploaded Dataset Preview")
    st.dataframe(data.head())
    
    # Identify target column dynamically
    target_candidates = [col for col in data.columns if 'histological staging' in col.lower()]
    target_col = target_candidates[0] if target_candidates else data.columns[-1]
    
    if target_col in data.columns:
        X_test = data.drop(columns=[target_col])
        y_test = data[target_col]
        
        # Preprocess categorical features if any
        X_test = pd.get_dummies(X_test, drop_first=True)
        
        # Load pre-trained model binary
        model_path = os.path.join("model", model_file_map[model_option])
        
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Calculate evaluation metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            mcc = matthews_corrcoef(y_test, y_pred)
            
            try:
                y_prob = model.predict_proba(X_test)
                auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
            except Exception:
                auc = "N/A"
            
            # Feature c: Display Evaluation Metrics
            st.subheader(f"Evaluation Metrics: {model_option}")
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Accuracy", f"{acc:.4f}")
            col2.metric("AUC Score", f"{auc:.4f}" if isinstance(auc, float) else auc)
            col3.metric("Precision", f"{prec:.4f}")
            col4.metric("Recall", f"{rec:.4f}")
            col5.metric("F1 Score", f"{f1:.4f}")
            col6.metric("MCC", f"{mcc:.4f}")
            
            st.markdown("---")
            
            # Feature d: Confusion Matrix & Classification Report
            col_cm, col_rep = st.columns(2)
            
            with col_cm:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                plt.xlabel("Predicted Class")
                plt.ylabel("Actual Class")
                st.pyplot(fig)
                
            with col_rep:
                st.subheader("Classification Report")
                report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                st.dataframe(pd.DataFrame(report_dict).transpose())
                
        else:
            st.error(f"Saved model file `{model_path}` not found in the repository.")
    else:
        st.error(f"Target column `{target_col}` not found in uploaded dataset.")
else:
    st.info("Please upload `test_data.csv` using the sidebar to run model inference.")