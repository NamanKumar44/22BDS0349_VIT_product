import streamlit as st
import pytesseract
from PIL import Image
from transformers import pipeline
from googletrans import Translator

# ----------------------------
# 1. Load AI Models
# ----------------------------
qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
translator = Translator()

# ----------------------------
# 2. Loan Underwriting Logic
# ----------------------------
def underwriting_engine(age, income, loan_amount, credit_score=650):
    score = 0
    if 21 <= age <= 60: score += 20
    if income >= 15000: score += 40
    if loan_amount <= income * 10: score += 30
    if credit_score >= 650: score += 10

    if score >= 80:
        return "✅ Loan Approved", score
    elif score >= 60:
        return "⚠️ Loan Pending Review", score
    else:
        return "❌ Loan Rejected", score

# ----------------------------
# 3. OCR Function
# ----------------------------
def extract_text_from_image(img_file):
    image = Image.open(img_file)
    text = pytesseract.image_to_string(image)
    return text

# ----------------------------
# 4. Streamlit UI
# ----------------------------
st.set_page_config(page_title="AI Loan Underwriting Agent", page_icon="💰")

st.title("💰 AI Loan Underwriting Agent")
st.write("Designed for Rural & Semi-Urban India")

# Language selector
lang = st.radio("🌐 Choose Language / भाषा चुनें", ["English", "हिन्दी"])

# Text translator helper
def t(text):
    return translator.translate(text, dest="hi").text if lang == "हिन्दी" else text

st.header(t("📋 Loan Application Form"))
age = st.number_input(t("Enter Age"), min_value=18, max_value=80, value=25)
income = st.number_input(t("Enter Monthly Income (₹)"), min_value=0, value=20000, step=1000)
loan_amount = st.number_input(t("Enter Loan Amount (₹)"), min_value=1000, value=50000, step=1000)
credit_score = st.slider(t("Credit Score"), 300, 900, 650)

if st.button(t("Check Loan Eligibility")):
    decision, score = underwriting_engine(age, income, loan_amount, credit_score)
    st.subheader(f"{t('Decision')}: {decision}")
    st.write(f"{t('Eligibility Score')}: {score}")

st.header(t("🪪 Upload Document (Aadhaar/PAN)"))
uploaded_file = st.file_uploader(t("Upload an ID document"), type=["jpg", "png", "jpeg"])
if uploaded_file:
    text = extract_text_from_image(uploaded_file)
    st.text_area(t("Extracted Text"), text, height=200)

st.header(t("🤖 AI Assistant"))
context = st.text_area(t("Provide applicant details (income, occupation, etc.)"))
question = st.text_input(t("Ask a question (e.g., What is the applicant's income?)"))
if st.button(t("Ask AI")):
    if context and question:
        # If Hindi selected, translate both to English for model
        if lang == "हिन्दी":
            context_en = translator.translate(context, src="hi", dest="en").text
            question_en = translator.translate(question, src="hi", dest="en").text
            answer = qa_model(question=question_en, context=context_en)
            # Translate answer back to Hindi
            answer_trans = translator.translate(answer["answer"], dest="hi").text
            st.write("**उत्तर:**", answer_trans)
        else:
            answer = qa_model(question=question, context=context)
            st.write("**Answer:**", answer["answer"])
    else:
        st.warning(t("Please enter both context and question."))
