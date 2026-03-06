import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="CreditWise Loan System", page_icon="🏦")

# Load saved files
@st.cache_resource
def load_artifacts():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    return model, scaler, encoder

model, scaler, encoder = load_artifacts()

# App title
st.title("🏦 CreditWise Loan Approval Predictor")
st.markdown("Fill in the details below to predict loan approval.")

# Input form
col1, col2 = st.columns(2)

with col1:
    applicant_income   = st.number_input("Applicant Income (₹)", 1000, 100000, 10000)
    coapplicant_income = st.number_input("Co-applicant Income (₹)", 0, 50000, 2000)
    age                = st.slider("Age", 18, 70, 30)
    dependents         = st.selectbox("Dependents", [0, 1, 2, 3])
    credit_score       = st.slider("Credit Score", 300, 900, 650)
    existing_loans     = st.selectbox("Existing Loans", [0, 1, 2, 3, 4])

with col2:
    dti_ratio        = st.slider("DTI Ratio", 0.0, 1.0, 0.3, 0.01)
    savings          = st.number_input("Savings (₹)", 0, 100000, 10000)
    collateral_value = st.number_input("Collateral Value (₹)", 0, 200000, 25000)
    loan_amount      = st.number_input("Loan Amount (₹)", 1000, 100000, 20000)
    loan_term        = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60, 72, 84])
    education_level  = st.selectbox("Education", ["Graduate", "Not Graduate"])

employment_status = st.selectbox("Employment Status", ["Salaried", "Self-employed", "Unemployed"])
marital_status    = st.selectbox("Marital Status", ["Married", "Single"])
loan_purpose      = st.selectbox("Loan Purpose", ["Business", "Car", "Education", "Home", "Personal"])
property_area     = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])
gender            = st.selectbox("Gender", ["Female", "Male"])
employer_category = st.selectbox("Employer Category", ["Government", "MNC", "Private", "Unemployed"])

# Predict button
if st.button("🔍 Predict Loan Approval", use_container_width=True):

    # Feature engineering (matches your notebook 8)
    dti_sq      = dti_ratio ** 2
    credit_sq   = credit_score ** 2
    income_log  = np.log1p(applicant_income)
    edu_encoded = 1 if education_level == "Not Graduate" else 0

    # One-Hot Encoding (matches your notebook 4)
    cat_input = pd.DataFrame([[employment_status, marital_status, loan_purpose,
                                property_area, gender, employer_category]],
                              columns=["Employment_Status", "Marital_Status",
                                       "Loan_Purpose", "Property_Area",
                                       "Gender", "Employer_Category"])

    encoded_cats = encoder.transform(cat_input)
    encoded_df   = pd.DataFrame(encoded_cats,
                                 columns=encoder.get_feature_names_out(
                                     ["Employment_Status", "Marital_Status",
                                      "Loan_Purpose", "Property_Area",
                                      "Gender", "Employer_Category"]))

    # Numeric features
    base = pd.DataFrame([[coapplicant_income, age, dependents, existing_loans,
                           savings, collateral_value, loan_amount, loan_term,
                           edu_encoded, dti_sq, credit_sq, income_log]],
                         columns=["Coapplicant_Income", "Age", "Dependents",
                                  "Existing_Loans", "Savings", "Collateral_Value",
                                  "Loan_Amount", "Loan_Term", "Education_Level",
                                  "DTI_ratio_sq", "Credit_Score_sq", "Applicant_Income_log"])

    # Combine & scale
    final_input = pd.concat([base.reset_index(drop=True),
                          encoded_df.reset_index(drop=True)], axis=1)

# Fix column order to exactly match training data
correct_order = [
    "Coapplicant_Income", "Age", "Dependents", "Existing_Loans",
    "Savings", "Collateral_Value", "Loan_Amount", "Loan_Term",
    "Education_Level",
    "Employment_Status_Salaried", "Employment_Status_Self-employed",
    "Employment_Status_Unemployed", "Marital_Status_Single",
    "Loan_Purpose_Car", "Loan_Purpose_Education", "Loan_Purpose_Home",
    "Loan_Purpose_Personal", "Property_Area_Semiurban", "Property_Area_Urban",
    "Gender_Male", "Employer_Category_Government", "Employer_Category_MNC",
    "Employer_Category_Private", "Employer_Category_Unemployed",
    "DTI_ratio_sq", "Credit_Score_sq", "Applicant_Income_log"
]

final_input = final_input[correct_order]
scaled      = scaler.transform(final_input)

    # Predict
    prediction  = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0]

    st.divider()
    if prediction == 1:
        st.success(f"✅ Loan APPROVED — Confidence: {probability[1]*100:.1f}%")
    else:
        st.error(f"❌ Loan REJECTED — Confidence: {probability[0]*100:.1f}%")

    st.caption("Model: Logistic Regression | Accuracy: 89%")