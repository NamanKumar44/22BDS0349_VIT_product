import io
import re
import uuid
import base64
from datetime import datetime

import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import pytesseract
import cv2
from deep_translator import GoogleTranslator
from fpdf import FPDF

# ------------------------------
# App Config
# ------------------------------
st.set_page_config(page_title="AI Loan Agent", page_icon="📱", layout="centered")

APP_VERSION = "2.0"
LANGUAGES = ["English", "Hindi", "Tamil", "Telugu", "Bengali", "Marathi", "Kannada"]

# ------------------------------
# i18n Helper
# ------------------------------
def t(text: str, lang: str) -> str:
    if lang == "English" or not text:
        return text
    try:
        return GoogleTranslator(source="auto", target=lang.lower()).translate(text)
    except Exception:
        return text

# ------------------------------
# OCR & Doc Validation
# ------------------------------
AADHAAR_REGEX = r"\b(\d{4}\s?\d{4}\s?\d{4})\b"
PAN_REGEX = r"\b([A-Z]{5}\d{4}[A-Z])\b"
PAYSLIP_HINTS = ["pay slip", "salary", "net pay", "gross pay", "employee id", "earnings"]

def run_ocr(image: Image.Image) -> str:
    pil = ImageOps.exif_transpose(image.convert("L"))
    return pytesseract.image_to_string(pil, lang="eng")

def detect_doc_type(ocr_text: str):
    text = ocr_text.lower()
    if re.search(AADHAAR_REGEX, ocr_text.replace(" ", "")) or "aadhaar" in text:
        return "aadhaar"
    if re.search(PAN_REGEX, ocr_text) or "income tax" in text:
        return "pan"
    if any(k in text for k in PAYSLIP_HINTS):
        return "payslip"
    return "unknown"

# ------------------------------
# Quality Coach
# ------------------------------
def quality_metrics(image: Image.Image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, _, v = cv2.split(hsv)
    glare_ratio = float(np.mean(v > 240))

    edges = cv2.Canny(gray, 80, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = max([cv2.contourArea(c) for c in contours], default=0.0)
    coverage = max_area / float(img.shape[0] * img.shape[1])

    score = min(1.0, (blur_var/150 + (1-glare_ratio*3) + coverage*2) / 3)
    tips = []
    if blur_var < 120: tips.append("Hold steady to reduce blur")
    if glare_ratio > 0.2: tips.append("Tilt to reduce glare")
    if coverage < 0.35: tips.append("Capture full document")

    return score, tips

# ------------------------------
# Face Detection
# ------------------------------
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def face_present(pil_img: Image.Image):
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.2, 6)
    return len(faces) > 0

# ------------------------------
# Risk & Decision
# ------------------------------
def risk_score(income, loan, doc_ok, quality, face_ok):
    if income <= 0: return 0, "reject"
    util = loan / max(1, income)
    score = 0.4*(1.5 - util/8) + 0.3*(1 if doc_ok else 0) + 0.2*quality + 0.1*(1 if face_ok else 0)
    score = np.clip(score, 0, 1)
    if score > 0.7: return score, "approve"
    elif score > 0.5: return score, "conditional"
    else: return score, "reject"

def explain(decision, lang):
    if decision == "approve":
        return t("✅ Approved — Offer ready!", lang)
    elif decision == "conditional":
        return t("⚠️ Almost there — Please upload better ID/selfie or income proof", lang)
    else:
        return t("❌ Not approved — Try again with valid ID & lower loan amount", lang)

# ------------------------------
# PDF Export
# ------------------------------
def generate_pdf(data: dict) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, "Loan Application Summary", ln=True, align="C")
    for k,v in data.items():
        pdf.cell(0, 10, f"{k}: {v}", ln=True)

    # Return as bytes instead of saving to file
    return pdf.output(dest="S").encode("latin-1")



# ------------------------------
# UI
# ------------------------------
lang = st.selectbox("🌐 Select Language", LANGUAGES, index=0)
st.title(t("📱 AI-Enabled Loan Underwriting Agent", lang))
st.caption(t("Smart, multilingual, designed for rural India 🚀", lang))

with st.expander(t("Consent & Privacy", lang), expanded=True):
    st.write(t("We process your documents only for eligibility. Sensitive fields are masked.", lang))
    consent = st.checkbox(t("I agree & consent", lang))

st.subheader(t("📸 Upload ID / Income Proof", lang))
id_file = st.file_uploader(t("Upload PAN/Aadhaar", lang), type=["png","jpg","jpeg"])
pay_file = st.file_uploader(t("Upload Pay Slip (optional)", lang), type=["png","jpg","jpeg"])
selfie = st.camera_input(t("Take a Selfie", lang))

st.subheader(t("📊 Loan Details", lang))
income = st.number_input(t("Monthly Income (₹)", lang), min_value=0, step=1000)
loan = st.number_input(t("Requested Loan (₹)", lang), min_value=0, step=1000)

if st.button(t("Run Underwriting", lang), disabled=not consent):
    if not consent:
        st.error(t("Please provide consent", lang))
        st.stop()

    doc_ok, quality, tips = False, 0, []
    if id_file:
        id_img = Image.open(id_file)
        st.image(id_img, caption=t("Uploaded ID", lang))
        ocr_text = run_ocr(id_img)
        st.text_area("Extracted Text", ocr_text, height=100)
        doc_type = detect_doc_type(ocr_text)
        quality, tips = quality_metrics(id_img)
        st.write(t("Doc Type:", lang), doc_type, " | ", t("Quality:", lang), f"{int(quality*100)}%")
        if tips: st.info(" | ".join(tip for tip in tips))
        doc_ok = doc_type in ["pan","aadhaar"] and quality > 0.5
    else:
        st.warning(t("Upload a valid government ID", lang))

    face_ok = False
    if selfie:
        face_ok = face_present(Image.open(selfie))
        st.write(t("Selfie Detected:", lang), face_ok)

    score, decision = risk_score(income, loan, doc_ok, quality, face_ok)
    st.progress(score)
    st.subheader(explain(decision, lang))

    summary = {
        "Decision": decision.upper(),
        "Risk Score": f"{int(score*100)}",
        "Monthly Income": income,
        "Requested Loan": loan,
        "Doc Valid": doc_ok,
        "Face Detected": face_ok,
        "Time": datetime.utcnow().isoformat()
    }
    pdf = generate_pdf(summary)
    st.download_button("📥 Download Summary", pdf, "loan_summary.pdf")
