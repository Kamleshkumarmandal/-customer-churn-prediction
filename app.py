import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("churn_model.pkl")
model = joblib.load("churn_model.pkl")

# -----------------------------
# HEADER
# -----------------------------
st.title("📊 Customer Churn Prediction")
st.write(
    "Enter customer information below to predict the probability "
    "of customer churn."
)

st.divider()

# -----------------------------
# CUSTOMER INFORMATION
# -----------------------------
st.subheader("👤 Customer Information")

with st.form("churn_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1]
        )

        partner = st.selectbox(
            "Partner",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"]
        )

        tenure = st.number_input(
            "Tenure (months)",
            min_value=0,
            max_value=72,
            value=12
        )

    with col2:
        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"]
        )

        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        online_security = st.selectbox(
            "Online Security",
            ["No", "Yes", "No internet service"]
        )

        online_backup = st.selectbox(
            "Online Backup",
            ["No", "Yes", "No internet service"]
        )

    with col3:
        device_protection = st.selectbox(
            "Device Protection",
            ["No", "Yes", "No internet service"]
        )

        tech_support = st.selectbox(
            "Tech Support",
            ["No", "Yes", "No internet service"]
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            ["No", "Yes", "No internet service"]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["No", "Yes", "No internet service"]
        )

    st.subheader("💳 Account Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

    with col2:
        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

    with col3:
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    col1, col2 = st.columns(2)

    with col1:
        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=70.0
        )

    with col2:
        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=1000.0
        )

    st.write("")

    submit = st.form_submit_button(
        "🔮 Predict Churn",
        use_container_width=True
    )


# -----------------------------
# PREDICTION
# -----------------------------
if submit:

    # Create blank row with model features
    customer_data = pd.DataFrame(
        0,
        index=[0],
        columns=feature_columns
    )

    # Numeric features
    customer_data["SeniorCitizen"] = senior_citizen
    customer_data["tenure"] = tenure
    customer_data["MonthlyCharges"] = monthly_charges
    customer_data["TotalCharges"] = total_charges

    # Categorical features
    categorical_data = {
        "gender": gender,
        "Partner": partner,
        "Dependents": dependents,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method
    }

    # Convert categorical input to dummy variables
    for column, value in categorical_data.items():

        feature_name = f"{column}_{value}"

        if feature_name in customer_data.columns:
            customer_data.loc[0, feature_name] = 1

    # Prediction
    prediction = model.predict(customer_data)[0]

    churn_probability = model.predict_proba(
        customer_data
    )[0][1]

    # -----------------------------
    # RESULT
    # -----------------------------

    st.divider()

    st.subheader("📈 Prediction Result")

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:
        st.metric(
            "Churn Probability",
            f"{churn_probability * 100:.2f}%"
        )

    with result_col2:

        if churn_probability >= 0.70:
            risk = "High Risk"
        elif churn_probability >= 0.40:
            risk = "Medium Risk"
        else:
            risk = "Low Risk"

        st.metric(
            "Risk Level",
            risk
        )

    with result_col3:

        if prediction == 1:
            result = "Likely to Churn"
        else:
            result = "Likely to Stay"

        st.metric(
            "Prediction",
            result
        )

    # -----------------------------
    # RESULT MESSAGE
    # -----------------------------

    if prediction == 1:

        st.error(
            "⚠️ This customer is likely to CHURN."
        )

        st.write(
            "The company may consider taking customer retention "
            "actions for this customer."
        )

    else:

        st.success(
            "✅ This customer is likely to STAY."
        )

        st.write(
            "The customer currently shows a lower probability "
            "of churn."
        )