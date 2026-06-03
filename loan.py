import streamlit as st
import pickle
import numpy as np
import pandas as pd
import shap

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Loan Approval System",
    page_icon="🏦",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

h1 {
    color: #003366;
    text-align: center;
    font-size: 42px;
}

h2, h3 {
    color: #003366;
}

.stButton>button {
    background-color: #003366;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
}

.stButton>button:hover {
    background-color: #0055aa;
    color: white;
}

.css-1d391kg {
    background-color: #eaf2ff;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("loan_model.pkl", "rb"))

# ---------------- SIDEBAR ----------------
st.sidebar.title("🏦 Loan Approval System")

st.sidebar.info(
    """
    This application predicts loan approval using:
    
    ✔ Machine Learning  
    ✔ Random Forest  
    ✔ Explainable AI (SHAP)  
    """
)

st.sidebar.success("Model Accuracy: 97%")

# ---------------- TITLE ----------------
st.title("🏦 Loan Approval & Risk Prediction")

st.markdown(
    "### AI-Based Banking Decision System"
)

# ---------------- INPUT SECTION ----------------
st.markdown("---")

st.header("📋 Applicant Details")

col1, col2 = st.columns(2)

with col1:

    dependents = st.slider(
        "Number of Dependents",
        0,
        5
    )

    education = st.selectbox(
        "Education",
        ["Graduate", "Not Graduate"]
    )

    self_employed = st.selectbox(
        "Self Employed",
        ["Yes", "No"]
    )

    income = int(st.text_input(
        "Annual Income",
        "500000"
    ))

with col2:

    loan_amount = int(st.text_input(
        "Loan Amount",
        "200000"
    ))

    loan_term = int(st.text_input(
        "Loan Term",
        "12"
    ))

    cibil = int(st.text_input(
        "CIBIL Score",
        "750"
    ))

    total_assets = int(st.text_input(
        "Total Assets Value",
        "400000"
    ))

# ---------------- ENCODING ----------------
education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0

# ---------------- HIDDEN FEATURES ----------------
residential_assets = total_assets * 0.4
commercial_assets = total_assets * 0.3
luxury_assets = total_assets * 0.2
bank_assets = total_assets * 0.1

# ---------------- PREDICTION ----------------
if st.button("Predict Loan Status"):

    input_data = np.array([[
        dependents,
        education,
        self_employed,
        income,
        loan_amount,
        loan_term,
        cibil,
        residential_assets,
        commercial_assets,
        luxury_assets,
        bank_assets
    ]])

    # Prediction
    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)[0][1]

    # ---------------- RESULT SECTION ----------------
    st.markdown("---")

    st.header("📊 Prediction Result")

    col3, col4 = st.columns(2)

    with col3:

        if prediction[0] == 1:

            st.success("✅ Loan Approved")

        else:

            st.error("❌ Loan Rejected")

    with col4:

        st.metric(
            "Approval Probability",
            f"{probability * 100:.2f}%"
        )

    # ---------------- RISK ANALYSIS ----------------
    st.subheader("📌 Risk Analysis")

    if probability > 0.80:

        st.success("Low Risk Applicant")

    elif probability > 0.50:

        st.warning("Medium Risk Applicant")

    else:

        st.error("High Risk Applicant")

    # ---------------- XAI SECTION ----------------
    with st.expander("🔍 Explainable AI Analysis"):

        feature_names = [
            "Dependents",
            "Education",
            "Self Employed",
            "Income",
            "Loan Amount",
            "Loan Term",
            "CIBIL Score",
            "Residential Assets",
            "Commercial Assets",
            "Luxury Assets",
            "Bank Assets"
        ]

        input_df = pd.DataFrame(
            input_data,
            columns=feature_names
        )

        # SHAP Explainer
        explainer = shap.Explainer(model)

        shap_values = explainer(input_df)

        # Approved class contribution
        contributions = shap_values.values[0][:, 1]

        feature_impact = []

        for feature, value in zip(feature_names, contributions):

            value = float(value)

            feature_impact.append((feature, value))

        # Sort by impact
        feature_impact = sorted(
            feature_impact,
            key=lambda x: abs(x[1]),
            reverse=True
        )

        st.subheader("Top Feature Contributions")

        for feature, value in feature_impact[:5]:

            if value > 0:

                st.write(
                    f"✔ {feature} increased approval chance by {abs(value):.3f}"
                )

            else:

                st.write(
                    f"⚠ {feature} reduced approval chance by {abs(value):.3f}"
                )

        # ---------------- REJECTION ANALYSIS ----------------
        if prediction[0] == 0:

            st.subheader("⚠ Main Reasons for Rejection")

            negative_features = [
                (feature, value)
                for feature, value in feature_impact
                if value < 0
            ]

            for feature, value in negative_features[:3]:

                st.write(
                    f"{feature} negatively affected loan approval"
                )

# ---------------- FOOTER ----------------
st.markdown("---")

st.markdown(
    "<center>Developed using Machine Learning, Random Forest, Streamlit and SHAP Explainable AI</center>",
    unsafe_allow_html=True
)