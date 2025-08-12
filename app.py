import streamlit as st
import joblib

# Load saved model & vectorizer
model = joblib.load("model/spam_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

st.title("📩 Spam Detection App")

message = st.text_area("Enter a message:")

if st.button("Predict"):
    data = vectorizer.transform([message])
    prediction = model.predict(data)[0]
    if prediction == 1:
        st.error("🚨 Spam Message Detected!")
    else:
        st.success("✅ Not Spam")
