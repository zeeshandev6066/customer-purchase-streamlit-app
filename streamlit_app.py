
import streamlit as st
import joblib

# Load Model
model=joblib.load("customer_purchase_model.pkl")


st.set_page_config(
    page_title="AI Purchase Predictor",
    page_icon="🧠",
    layout="centered"
)

st.title("🛒 Customer Purchase Prediction")
st.markdown("Predict whether a customer will purchase based on visit count.")
st.divider()


# User Input

visits=st.number_input(
    "👤 Number of Customer Visit",
    min_value=0,
    step=1
)

if st.button("🔍 Predict"):
    prediction=model.predict([[visits]])
    probability=model.predict_proba([[visits]])
    confidence=probability[0][1]*100

    st.divider()

    if prediction[0] == 1:
        st.success("✅ Customer is **LIKELY TO PURCHASE**")
    else:
        st.error("❌ Customer is **NOT LIKELY TO PURCHASE**")


    st.metric(
        label="📊 Confidence Level",
        value=f"{confidence:.2f}%"
    )

    st.divider()
    st.caption("Built with Python • Machine Learning • Streamlit")

# Run the App
# streamlit run day2.py
