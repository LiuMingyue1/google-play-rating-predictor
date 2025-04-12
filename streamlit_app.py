import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load model and required metadata
model = joblib.load("best_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")
feature_means = joblib.load("feature_means.pkl")

# Streamlit page configuration
st.set_page_config(page_title="App Rating Classifier", layout="centered")
st.title("📱 Google Play App Rating Classifier")
st.markdown("Fill in the app details below. We will predict whether your app is likely to be **highly rated (Rating ≥ 4.5)**.")

# User input: main features
reviews = st.number_input("Number of Reviews", min_value=0, value=100)
size = st.number_input("App Size (MB)", min_value=0.0, value=10.0)
installs = st.number_input("Number of Installs", min_value=0, value=10000)
price = st.number_input("App Price (USD)", min_value=0.0, value=0.0)

# Derived features
engagement = (installs * reviews) / (size + 1)
price_per_review = price / (reviews + 1)

# Partial input dictionary
user_input = {
    "Reviews": reviews,
    "Size": size,
    "Installs": installs,
    "Price": price,
    "Engagement Score": engagement,
    "Price per Review": price_per_review
}

# Fill in missing features using training-set means
for col in feature_columns:
    if col not in user_input:
        user_input[col] = feature_means.get(col, 0.0)

# Create input DataFrame in correct order
input_df = pd.DataFrame([user_input])[feature_columns]

# Prediction logic
if st.button("Predict Rating Category"):
    prob = model.predict_proba(input_df)[0][1]  # Probability of class 1 (high rating)
    threshold = 0.4  # Hidden classification threshold
    pred = 1 if prob > threshold else 0

    label = "🌟 Highly Rated App (Rating ≥ 4.5)" if pred == 1 else "📉 Average Rated App (Rating < 4.5)"

    st.write("---")
    st.markdown(f"### 🧠 Prediction Result: {label}")
    st.info(f"📊 Predicted Probability of Being Highly Rated: **{prob:.2%}**")
    st.caption("Model internally uses a threshold of 0.4 for classification.")
