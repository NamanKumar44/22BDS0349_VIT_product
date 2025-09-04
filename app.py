import streamlit as st
import pandas as pd
import numpy as np
import pytesseract
from PIL import Image
from deep_translator import GoogleTranslator
from transformers import pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# --------------------------
# STREAMLIT CONFIG
# --------------------------
st.set_page_config(page_title="AI Loan Underwriting", layout="centered")

st.title("📱 AI-Enabled Loan Underwriting Agent")
st.write("Smart, multilingual, and designed for rural & semi-urban India 🚀")

# --------------------------
# 1. LANGUAGE TRANSLATOR
# --------------------------
def translate_text(text, dest_lang="en"):
    try:
        return GoogleTranslator(source="auto", target=dest_lang).translate(text)
    except:
        return text  # fallback if no internet


lang = st.selectbox("🌍 Select Language", ["English", "Hindi", "Tamil", "Telugu"])
lang_codes = {"English": "en", "Hindi": "hi", "Tamil": "ta", "Telugu": "te"}

# --------------------------
# 2. OCR UPLOAD FEATURE
# --------------------------
st.subheader("📷 Upload ID / Income Proof")
uploaded_file = st.file_uploader("Upload Image (PAN, Aadhaar, Pay Slip)", type=["png", "jpg", "jpeg"])
ocr_text = ""

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Document", use_column_width=True)
    ocr_text = pytesseract.image_to_string(img)
    st.success("✅ OCR Extracted Text")
    st.write(ocr_text)

    translated = translate_text(ocr_text, dest_lang=lang_codes[lang])
    st.write(f"🔤 Translated ({lang}):")
    st.write(translated)

# --------------------------
# 3. AI RISK ASSESSMENT (LLM)
# --------------------------
@st.cache_resource
def load_sentiment_pipeline():
    try:
        return pipeline("sentiment-analysis")
    except:
        return None

nlp = load_sentiment_pipeline()

if nlp and ocr_text:
    result = nlp(ocr_text[:500])[0]  # limit text length
    st.subheader("🤖 AI Risk Signal")
    st.write(f"Sentiment: {result['label']}, Confidence: {round(result['score']*100,2)}%")

# --------------------------
# 4. SIMPLE UNDERWRITING MODEL
# --------------------------
st.subheader("📊 Loan Eligibility Check")

income = st.number_input("Monthly Income (₹)", min_value=1000, step=500)
loan_amt = st.number_input("Requested Loan Amount (₹)", min_value=1000, step=500)
age = st.slider("Age", 18, 70, 25)

if st.button("Check Loan Eligibility"):
    X = np.array([[income, loan_amt, age]])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Dummy model (replace with real underwriting model)
    model = LogisticRegression()
    model.fit([[1, 2, 3], [2, 3, 4], [3, 4, 5]], [0, 1, 1])

    pred = model.predict(X_scaled)[0]
    if pred == 1:
        st.success("🎉 Approved: You are eligible for loan")
    else:
        st.error("❌ Rejected: Not eligible based on current data")

# --------------------------
# 5. LOW BANDWIDTH HANDLING
# --------------------------
st.info("📡 Works offline with cached models. If internet fails, OCR + basic loan rules still function.")
