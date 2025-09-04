import streamlit as st
import pytesseract
from PIL import Image
from deep_translator import GoogleTranslator

# ------------------------------
# Translation Helper
# ------------------------------
def translate_text(text, lang):
    if lang == "English":
        return text
    try:
        return GoogleTranslator(source="auto", target=lang.lower()).translate(text)
    except:
        return text

# ------------------------------
# UI Starts Here
# ------------------------------
st.set_page_config(page_title="AI Loan Agent", page_icon="📱", layout="centered")

# Language Options
languages = ["English", "Hindi", "Tamil", "Telugu"]
selected_lang = st.selectbox("🌐 Select Language", languages)

# Title
st.title(translate_text("📱 AI-Enabled Loan Underwriting Agent", selected_lang))
st.caption(translate_text("Smart, multilingual, and designed for rural & semi-urban India 🚀", selected_lang))

# File Upload
st.subheader(translate_text("📸 Upload ID / Income Proof", selected_lang))
uploaded_file = st.file_uploader(
    translate_text("Upload Image (PAN, Aadhaar, Pay Slip)", selected_lang),
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption=translate_text("Uploaded Document", selected_lang), use_column_width=True)
    extracted_text = pytesseract.image_to_string(image)
    st.text_area(translate_text("📑 Extracted Text", selected_lang), extracted_text)

# Loan Eligibility
st.subheader(translate_text("📊 Loan Eligibility Check", selected_lang))
income = st.number_input(translate_text("Monthly Income (₹)", selected_lang), min_value=0)
loan_amount = st.number_input(translate_text("Requested Loan Amount (₹)", selected_lang), min_value=0)

if st.button(translate_text("Check Eligibility", selected_lang)):
    if income > 0:
        if loan_amount <= income * 10:
            st.success(translate_text("✅ Eligible for Loan", selected_lang))
        else:
            st.error(translate_text("❌ Loan Request Too High", selected_lang))
    else:
        st.warning(translate_text("⚠️ Please enter valid income", selected_lang))
