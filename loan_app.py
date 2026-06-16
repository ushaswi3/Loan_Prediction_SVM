import streamlit as st
import numpy as np
import pickle

# Load models
svm_linear = pickle.load(open("svm_linear.pkl", "rb"))
svm_poly   = pickle.load(open("svm_polynomial.pkl", "rb"))
svm_rbf    = pickle.load(open("svm_rbf.pkl", "rb"))
scaler     = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Smart Loan Approval System")

st.title("🏦 Smart Loan Approval System")
st.write("This system uses Support Vector Machines to predict loan approval.")

# Sidebar
st.sidebar.header("Applicant Details")

income = st.sidebar.number_input("Applicant Income", min_value=0)
loan = st.sidebar.number_input("Loan Amount", min_value=0)

credit = st.sidebar.selectbox("Credit History", ["Yes", "No"])
credit_val = 1 if credit == "Yes" else 0

employment = st.sidebar.selectbox(
    "Employment Status", ["Employed", "Self Employed"]
)
employment_val = 1 if employment == "Self Employed" else 0

kernel = st.radio(
    "Select SVM Kernel",
    ["Linear SVM", "Polynomial SVM", "RBF SVM"]
)

if st.button("Check Loan Eligibility"):

    input_data = np.array([[income, loan, credit_val, employment_val]])
    input_scaled = scaler.transform(input_data)

    if kernel == "Linear SVM":
        model = svm_linear
    elif kernel == "Polynomial SVM":
        model = svm_poly
    else:
        model = svm_rbf

    prediction = model.predict(input_scaled)[0]
    confidence = model.predict_proba(input_scaled).max()

    st.subheader("Loan Decision")

    if prediction == 1:
        st.success("✅ Loan Approved")
        st.write(
            "Based on credit history and income pattern, "
            "the applicant is likely to repay the loan."
        )
    else:
        st.error("❌ Loan Rejected")
        st.write(
            "Based on financial risk indicators, "
            "the applicant is unlikely to repay the loan."
        )

    st.info(f"Kernel Used: {kernel}")
    st.info(f"Confidence Score: {confidence:.2f}")
