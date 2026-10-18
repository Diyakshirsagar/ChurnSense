
import joblib
import streamlit as st
import pandas as pd
import os

@st.cache_resource
def load_model():
    model_path = os.path.join(
        os.path.dirname(__file__),
        "churn_model.pkl"
    )
    return joblib.load(model_path)

model = load_model()

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="ChurnSense | Customer Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================= DARK THEME & ALIGNMENT =================
st.markdown("""
<style>
/* Main application background */
.stApp,
[data-testid="stAppViewContainer"] {
    background-color: #111827 !important;
    color: #F3F4F6 !important;
}

/* Header and toolbar */
[data-testid="stHeader"] {
    background-color: #111827 !important;
}

/* Page width and alignment */
.block-container {
    max-width: 1400px !important;
    width: 100% !important;
    padding: 2rem 2.5rem 3rem 2.5rem !important;
    margin: 0 auto !important;
}

/* Headings */
h1, h2, h3, h4 {
    color: #F9FAFB !important;
    line-height: 1.35 !important;
}

/* General text */
p, label, [data-testid="stMarkdownContainer"] {
    color: #D1D5DB;
}

[data-testid="stCaptionContainer"] {
    color: #9CA3AF !important;
}

/* Overview metric cards */
[data-testid="stMetric"] {
    background: #1F2937 !important;
    border: 1px solid #374151;
    border-radius: 12px;
    padding: 18px;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricValue"],
[data-testid="stMetricDelta"] {
    color: #F9FAFB !important;
}

/* Input fields */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background-color: #1F2937 !important;
    border-color: #4B5563 !important;
}

div[data-baseweb="select"] *,
div[data-baseweb="input"] input {
    color: #F9FAFB !important;
}

/* Teal buttons */
div.stButton > button,
div.stFormSubmitButton > button {
    background-color: #0F766E !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px;
    min-height: 44px;
    font-weight: 600;
}

div.stButton > button:hover,
div.stFormSubmitButton > button:hover {
    background-color: #115E59 !important;
}

/* Cards and containers */
[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #1F2937;
    border: 1px solid #374151;
    border-radius: 12px;
}

/* Dividers */
hr {
    border-color: #374151 !important;
}

/* Responsive padding */
@media (max-width: 768px) {
    .block-container {
        padding: 1.25rem 1rem 2rem 1rem !important;
    }
}
</style>
""", unsafe_allow_html=True)


# ================= HEADER =================
st.markdown("##### 🟢 CUSTOMER INTELLIGENCE")
st.title("📊 ChurnSense")


st.markdown(
    "Understand customer behavior, explore churn patterns, "
    "and make informed retention decisions."
)

st.divider()



# ================= CUSTOMER ASSESSMENT =================
st.subheader("Customer Assessment")
st.write("Enter customer details to prepare an assessment.")

form_col, result_col = st.columns(
    [1.15, 0.85],
    gap="large"
)


# ================= CUSTOMER FORM =================
with form_col:
    with st.container(border=True):
        st.markdown("### 01 · Customer Profile")

        with st.form("customer_form"):
            tenure = st.number_input(
                "Tenure (months)",
                min_value=0,
                max_value=72,
                value=12
            )

            contract = st.selectbox(
                "Contract Type",
                ["Month-to-month", "One year", "Two year"]
            )

            internet = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"]
            )

            st.markdown("### 02 · Customer Services")

            tech_support = st.selectbox(
                "Tech Support",
                ["Yes", "No", "No internet service"]
            )

            online_security = st.selectbox(
                "Online Security",
                ["Yes", "No", "No internet service"]
            )

            paperless = st.selectbox(
                "Paperless Billing",
                ["Yes", "No"]
            )

            st.markdown("### 03 · Billing Details")

            monthly_charges = st.number_input(
                "Monthly Charges ($)",
                min_value=0.0,
                max_value=200.0,
                value=70.0,
                step=1.0
            )

            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )

            submitted = st.form_submit_button(
                "Prepare Assessment →",
                use_container_width=True
            )


# ================= PREDICTION RESULT =================
# ================= PREDICTION RESULT =================
with result_col:
    with st.container(border=True):
        st.markdown("### Prediction Result")

        if model is None:
            st.error("The trained model could not be loaded.")
            st.caption(model_error or "Check that churn_model.pkl exists.")

        elif submitted:
            # Use the same feature names as the training dataset.
            customer = {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "No",
                "Dependents": "No",
                "tenure": tenure,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": internet,
                "OnlineSecurity": online_security,
                "OnlineBackup": "No internet service" if internet == "No" else "No",
                "DeviceProtection": "No internet service" if internet == "No" else "No",
                "TechSupport": tech_support,
                "StreamingTV": "No internet service" if internet == "No" else "No",
                "StreamingMovies": "No internet service" if internet == "No" else "No",
                "Contract": contract,
                "PaperlessBilling": paperless,
                "PaymentMethod": payment,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": tenure * monthly_charges,
            }

            # Include this only if the trained model expects it.
            if tenure <= 12:
                tenure_group = "New"
            elif tenure <= 36:
                tenure_group = "Developing"
            else:
                tenure_group = "Long-term"

            customer["TenureGroup"] = tenure_group

            try:
                input_df = pd.DataFrame([customer])

                # Match the exact features used to train the model.
                expected_features = getattr(
                    model, "feature_names_in_", None
                )

                if expected_features is not None:
                    input_df = input_df[list(expected_features)]

                prediction = int(model.predict(input_df)[0])

                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(input_df)[0]
                    classes = list(model.classes_)

                    if 1 in classes:
                        churn_probability = float(
                            probabilities[classes.index(1)]
                        )
                    else:
                        churn_probability = float(prediction)
                else:
                    churn_probability = float(prediction)

                st.markdown("#### Churn Probability")
                st.progress(churn_probability)
                st.metric(
                    "Estimated Probability",
                    f"{churn_probability:.1%}"
                )

                st.divider()
                st.markdown("#### Predicted Outcome")

                if prediction == 1:
                    st.error("⚠️ Customer likely to churn")
                else:
                    st.success("✅ Customer likely to stay")

                st.markdown("#### Risk Category")

                if churn_probability >= 0.70:
                    st.error("High Risk")
                elif churn_probability >= 0.40:
                    st.warning("Medium Risk")
                else:
                    st.success("Low Risk")

                st.caption(
                    "Prediction generated by your trained ML model. "
                    "Risk categories are illustrative thresholds."
                )

            except Exception as e:
                st.error("Prediction failed. Check the input features.")
                st.code(str(e))

        else:
            st.info("Enter customer details and prepare an assessment.")
            st.markdown("#### Churn Probability")
            st.progress(0)
            st.metric("Estimated Probability", "—")
            st.divider()
            st.markdown("#### Predicted Outcome")
            st.write("Awaiting customer assessment.")
            st.markdown("#### Risk Category")
            st.write("Not available yet.")

