import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
import os
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Netflix Title Classification: Movie or TV Show")
st.write("Predict whether a Netflix title is a **Movie (0)** or a **TV Show (1)** based on 12 engineered features.")

st.sidebar.header("Upload Test Data")
uploaded_file = st.sidebar.file_uploader("Upload test_data.csv", type=["csv"])

model_choice = st.sidebar.selectbox("Select Model", 
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "SVM"])

model_files = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "SVM": "model/svm.joblib"
}

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview")
    st.write(data.head())
    
    if 'target' in data.columns:
        X_test = data.drop(columns=['target'])
        y_test = data['target']
        
        try:
            scaler = joblib.load("model/scaler.joblib")
            model = joblib.load(model_files[model_choice])
            
            X_test_scaled = scaler.transform(X_test)
            y_pred = model.predict(X_test_scaled)
            
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test_scaled)[:, 1]
                auc = roc_auc_score(y_test, y_prob)
            else:
                auc = "N/A"
                
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            mcc = matthews_corrcoef(y_test, y_pred)
            
            st.write("### Evaluation Metrics")
            metrics_df = pd.DataFrame({
                "Metric": ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"],
                "Score": [acc, auc, prec, rec, f1, mcc]
            })
            st.table(metrics_df.style.format({"Score": "{:.4f}"}))
            
            st.write("### Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)
            
            st.write("### Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())
            
        except Exception as e:
            st.error(f"Error loading model or evaluating: {e}")
    else:
        st.error("Uploaded CSV must contain a 'target' column.")
else:
    st.info("Please upload the test_data.csv file from the repository to proceed.")
