import io
import re
import uuid
import base64
from datetime import datetime

import numpy as np
import pandas as pd
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

APP_VERSION = "1.4.0"
LANGUAGES = ["English", "Hindi", "Tamil", "Telugu", "Bengali", "Marathi", "Kannada"]
DEFAULT_LOCALE = "English"

# ------------------------------
# i18n Helper
# ------------------------------
def t(text: str, lang: str) -> str:
    if lang == "English" or not text:
        return text
    try:
        return GoogleTranslator(source="auto", target=lang.lower()).translate(text)
    except Exception:
        return text  # graceful fallback

# ------------------------------
# Utilities
# ------------------------------
@st.cache_data(show_spinner=False)
def pil_to_cv2(image: Image.Image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def mask_aadhaar(s: str) -> str:
    # Keep last 4 visible; mask others
    digits = re.sub(r"\D", "", s)
    if len(digits) >= 12:
        return f"XXXX XXXX {digits[-4:]}"
    return s

def mask_pan(s: str) -> str:
    m = re.search(r"[A-Z]{5}\d{4}[A-Z]", s.upper())
    if not m:
        return s
    pan = m.group(0)
    return pan[:2] + "XXX" + pan[5:9].replace(pan[5:9], "XXXX") + pan[-1]

def bytes_download_link(file_bytes: bytes, filename: str, label: str):
    b64 = base64.b64encode(file_bytes).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{label}</a>'
    st.markdown(href, unsafe_allow_html=True)

# ------------------------------
# OCR & Document Intelligence
# ------------------------------
AADHAAR_REGEX = r"\b(\d{4}\s?\d{4}\s?\d{4})\b"
PAN_REGEX = r"\b([A-Z]{5}\d{4}[A-Z])\b"
PAYSLIP_HINTS = ["pay slip", "salary", "net pay", "gross pay", "employee id", "earnings", "deductions"]

VALID_ID_KEYWORDS = [
    "government of india", "income tax", "permanent account number", "aadhaar", "unique identification authority"
]

def run_ocr(image: Image.Image) -> str:
    # Convert to gray, improve contrast
    pil = ImageOps.exif_transpose(image.convert("L"))
    text = pytesseract.image_to_string(pil, lang="eng")
    return text

def detect_doc_type(ocr_text: str):
    text = ocr_text.lower()
    is_aadhaar = re.search(AADHAAR_REGEX, ocr_text.replace(" ", "")) or "aadhaar" in text
    is_pan = re.search(PAN_REGEX, ocr_text) or "income tax" in text or "permanent account number" in text
    is_payslip = any(k in text for k in PAYSLIP_HINTS)

    # Heuristic boost for Govt IDs
    has_valid_keywords = any(k in text for k in VALID_ID_KEYWORDS)

    doc_type = "unknown"
    score = 0.0
    if is_aadhaar:
        doc_type, score = "aadhaar", 0.7
    if is_pan and score < 0.8:
        doc_type, score = "pan", 0.8
    if is_payslip and score < 0.6:
        doc_type, score = "payslip", 0.6
    if has_valid_keywords and doc_type in ["pan", "aadhaar"]:
        score += 0.1
    score = min(score, 0.95)

    return doc_type, score

def extract_identifiers(ocr_text: str):
    aadhaar = None
    pan = None
    aadhaar_m = re.search(AADHAAR_REGEX, ocr_text)
    if aadhaar_m:
        aadhaar = aadhaar_m.group(1)
    pan_m = re.search(PAN_REGEX, ocr_text)
    if pan_m:
        pan = pan_m.group(1)
    return aadhaar, pan

# ------------------------------
# Document Quality Coach
# ------------------------------
def quality_metrics(image: Image.Image):
    img = pil_to_cv2(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur: variance of Laplacian
    blur_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Glare: very bright pixel ratio
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, _, v = cv2.split(hsv)
    glare_ratio = float(np.mean(v > 240))

    # Edge/Crop: largest contour area vs image area
    edges = cv2.Canny(gray, 80, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = max([cv2.contourArea(c) for c in contours], default=0.0)
    img_area = float(img.shape[0] * img.shape[1])
    coverage = max_area / img_area if img_area else 0.0

    # Normalize to 0..1 quality score (rough heuristic)
    blur_score = min(1.0, blur_var / 150.0)  # >150 is sharp
    glare_score = 1.0 - min(1.0, glare_ratio * 4.0)  # <0.25 glare is ok
    coverage_score = min(1.0, coverage * 2.2)  # >0.45 coverage is ok
    overall = np.clip(0.25 * blur_score + 0.35 * glare_score + 0.40 * coverage_score, 0, 1)

    tips = []
    if blur_var < 120:
        tips.append("Hold steady or move closer to reduce blur.")
    if glare_ratio > 0.2:
        tips.append("Avoid direct light; tilt card to reduce glare.")
    if coverage < 0.35:
        tips.append("Fill the frame; capture all four corners.")

    return dict(
        blur_var=float(blur_var),
        glare_ratio=float(glare_ratio),
        coverage=float(coverage),
        quality=float(overall),
        tips=tips
    )

# ------------------------------
# Face Presence & Similarity (Basic)
# ------------------------------
@st.cache_resource(show_spinner=False)
def _load_face_detector():
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def face_present(pil_img: Image.Image):
    det = _load_face_detector()
    img = pil_to_cv2(pil_img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = det.detectMultiScale(gray, 1.2, 6)
    return len(faces) > 0

def face_similarity(id_img: Image.Image, selfie_img: Image.Image):
    # Very lightweight similarity using color histograms of face region; demo-grade only
    det = _load_face_detector()

    def crop_face(pil_img):
        img = pil_to_cv2(pil_img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = det.detectMultiScale(gray, 1.2, 6)
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
        crop = img[y:y+h, x:x+w]
        return crop

    a = crop_face(id_img)
    b = crop_face(selfie_img)
    if a is None or b is None:
        return 0.0

    def hist(img):
        return cv2.normalize(cv2.calcHist([img],[0,1,2],None,[8,8,8],[0,256,0,256,0,256]).flatten(), None).flatten()

    h1, h2 = hist(a), hist(b)
    # Bhattacharyya similarity (invert distance)
    dist = cv2.compareHist(h1.astype("float32"), h2.astype("float32"), cv2.HISTCMP_BHATTACHARYYA)
    sim = float(max(0.0, 1.0 - dist))  # 0..1
    return sim

# ------------------------------
# Risk & Eligibility
# ------------------------------
def risk_and_eligibility(income, loan_amt, doc_ok, q, face_ok, face_sim):
    # Base affordability
    if income <= 0:
        return 0.0, "reject", {"reason":"No income declared"}

    util = loan_amt / max(1, income)
    util_score = np.clip(1.5 - (util/8.0), 0, 1)  # best around util<=8x
    doc_score = 0.3 + 0.7 * (1.0 if doc_ok else 0.0)
    qual_score = q
    face_score = 0.2 + 0.8 * (1.0 if (face_ok and face_sim >= 0.43) else 0.0)

    # Weighted risk score
    score = np.clip(0.35*util_score + 0.25*doc_score + 0.25*qual_score + 0.15*face_score, 0, 1)

    if score >= 0.72:
        decision = "approve"
    elif score >= 0.52:
        decision = "conditional"
    else:
        decision = "reject"
    return float(score), decision, {}

def decision_explainer(decision, lang, context):
    reasons = []
    if context.get("no_doc"):
        reasons.append("Valid government ID not detected.")
    if context.get("bad_quality"):
        reasons.append("Document image is blurry / cropped / has glare.")
    if context.get("face_missing"):
        reasons.append("Selfie or face not detected.")
    if context.get("face_mismatch"):
        reasons.append("Selfie does not sufficiently match the ID photo.")
    if context.get("high_util"):
        reasons.append("Requested amount is high relative to declared income.")

    if decision == "approve":
        msg = "Approved 🎉 — Offer generated based on your information."
        next_steps = ["Review the offer terms.", "Accept and proceed to e-mandate."]
    elif decision == "conditional":
        msg = "Almost there — We need one more step."
        next_steps = [
            "Upload a clearer image of the ID (or a different ID).",
            "Add a pay slip or bank statement.",
            "Retake selfie in good light, face centered."
        ]
    else:
        msg = "Not approved at this time."
        next_steps = [
            "Reduce requested amount and try again.",
            "Share additional income proof.",
            "Ensure your ID and selfie are clear and well-lit."
        ]
    return t(msg, lang), [t(r, lang) for r in reasons], [t(s, lang) for s in next_steps]

# ------------------------------
# Offline Queue (Simulated)
# ------------------------------
def init_queue():
    if "queue" not in st.session_state:
        st.session_state.queue = []

def enqueue(op_type, payload):
    op = {"id": str(uuid.uuid4()), "type": op_type, "payload": payload, "status": "pending"}
    st.session_state.queue.append(op)

def process_queue():
    # Simulated: mark all pending as success (idempotent)
    for op in st.session_state.queue:
        if op["status"] == "pending":
            op["status"] = "done"

# ------------------------------
# PDF Summary
# ------------------------------
def generate_pdf_summary(data: dict) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Loan Application Summary", ln=True)
    pdf.set_font("Arial", "", 12)

    for k, v in data.items():
        pdf.multi_cell(0, 8, f"{k}: {v}")

    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()

# ------------------------------
# UI
# ------------------------------
st.markdown(f"<small>v{APP_VERSION}</small>", unsafe_allow_html=True)

lang = st.selectbox("🌐 Select Language", LANGUAGES, index=LANGUAGES.index(DEFAULT_LOCALE))
st.title(t("📱 AI-Enabled Loan Underwriting Agent", lang))
st.caption(t("Smart, multilingual, and designed for rural & semi-urban India 🚀", lang))

# Consent
with st.expander(t("Consent & Privacy", lang), expanded=True):
    st.write(t("We will process your documents to assess eligibility. We store only what is necessary and mask sensitive fields.", lang))
    consent = st.checkbox(t("I understand and consent to data processing for loan evaluation.", lang), value=False)

st.divider()

# ID / Income Proof
st.subheader(t("📸 Upload ID / Income Proof", lang))
id_file = st.file_uploader(t("Upload Image (PAN/Aadhaar)", lang), type=["png","jpg","jpeg"], key="id_file")
payslip_file = st.file_uploader(t("Upload Income Proof (Pay Slip / Bank Statement image)", lang), type=["png","jpg","jpeg"], key="pay_file")
selfie_img = st.camera_input(t("Take a Selfie (for face presence & match)", lang))

st.divider()

# Income & Loan
st.subheader(t("📊 Loan Eligibility Check", lang))
col1, col2 = st.columns(2)
with col1:
    income = st.number_input(t("Monthly Income (₹)", lang), min_value=0, step=1000)
with col2:
    loan_amount = st.number_input(t("Requested Loan Amount (₹)", lang), min_value=0, step=1000)

init_queue()

if st.button(t("Run Underwriting", lang), use_container_width=True, disabled=not consent):
    if not consent:
        st.warning(t("Please provide consent to proceed.", lang))
        st.stop()

    doc_ok = False
    context = {}

    # --- ID processing ---
    ocr_text = ""
    id_quality = {"quality": 0.0, "tips": []}
    id_img_pil = None
    if id_file is not None:
        id_img_pil = Image.open(id_file).convert("RGB")
        st.image(id_img_pil, caption=t("Uploaded ID", lang), use_column_width=True)

        id_quality = quality_metrics(id_img_pil)
        st.progress(id_quality["quality"])
        if id_quality["tips"]:
            st.info(t("Doc Coach Tips:", lang) + " " + " | ".join([t(x, lang) for x in id_quality["tips"]]))

        ocr_text = run_ocr(id_img_pil)
        st.text_area(t("📑 Extracted Text (ID)", lang), ocr_text, height=140)

        doc_type, doc_score = detect_doc_type(ocr_text)
        aadhaar, pan = extract_identifiers(ocr_text)

        # Mask for display
        masked_pan = mask_pan(pan) if pan else None
        masked_aadhaar = mask_aadhaar(aadhaar) if aadhaar else None

        cols = st.columns(3)
        cols[0].metric(t("Detected Doc", lang), doc_type.upper() if doc_type!="unknown" else t("Unknown", lang))
        cols[1].metric(t("Doc Confidence", lang), f"{int(doc_score*100)}%")
        cols[2].metric(t("Quality Score", lang), f"{int(id_quality['quality']*100)}%")

        if masked_pan:
            st.success(t("PAN:", lang) + f" {masked_pan}")
        if masked_aadhaar:
            st.success(t("Aadhaar:", lang) + f" {masked_aadhaar}")

        doc_ok = doc_type in ["pan","aadhaar"] and id_quality["quality"] >= 0.48
        if not doc_ok:
            if doc_type not in ["pan","aadhaar"]:
                context["no_doc"] = True
            if id_quality["quality"] < 0.48:
                context["bad_quality"] = True
    else:
        st.error(t("Please upload a government ID image.", lang))
        context["no_doc"] = True

    # --- Income proof optional heuristic ---
    if payslip_file is not None:
        pay_img = Image.open(payslip_file).convert("RGB")
        pay_text = run_ocr(pay_img)
        if any(k in pay_text.lower() for k in PAYSLIP_HINTS):
            st.success(t("Income proof detected.", lang))
            doc_ok = True  # boosts doc_ok if present

    # --- Face checks ---
    face_ok, face_sim = False, 0.0
    if selfie_img is not None:
        selfie_pil = Image.open(selfie_img).convert("RGB")
        st.image(selfie_pil, caption=t("Selfie", lang), use_column_width=False)
        face_ok = face_present(selfie_pil)
        if not face_ok:
            st.error(t("No face detected in selfie. Retake with your face centered.", lang))
            context["face_missing"] = True

        if face_ok and id_img_pil is not None:
            face_sim = face_similarity(id_img_pil, selfie_pil)
            st.metric(t("Face Similarity", lang), f"{int(face_sim*100)}%")
            if face_sim < 0.43:
                context["face_mismatch"] = True
    else:
        context["face_missing"] = True

    # --- Risk & Decision ---
    if loan_amount > max(1, income) * 10:
        context["high_util"] = True

    score, decision, _ = risk_and_eligibility(
        income=income,
        loan_amt=loan_amount,
        doc_ok=doc_ok,
        q=id_quality["quality"] if id_file else 0.0,
        face_ok=face_ok,
        face_sim=face_sim
    )

    msg, reasons, next_steps = decision_explainer(decision, lang, context)

    st.subheader(t("🧮 Risk Score", lang))
    st.progress(score)
    st.caption(f"{int(score*100)} / 100")

    if decision == "approve":
        st.success(msg)
    elif decision == "conditional":
        st.warning(msg)
    else:
        st.error(msg)

    if reasons:
        st.write("**" + t("Why:", lang) + "**")
        for r in reasons:
            st.markdown(f"- {r}")

    st.write("**" + t("Next Steps:", lang) + "**")
    for s_ in next_steps:
        st.markdown(f"- {s_}")

    # --- Queue (simulated resumables) ---
    enqueue("upload_id", {"ts": datetime.utcnow().isoformat()})
    if payslip_file:
        enqueue("upload_income", {"ts": datetime.utcnow().isoformat()})
    if selfie_img:
        enqueue("upload_selfie", {"ts": datetime.utcnow().isoformat()})

    with st.expander(t("Sync & Offline Queue", lang), expanded=True):
        cols = st.columns([1,1,1])
        with cols[0]:
            if st.button(t("Retry All", lang)):
                process_queue()
        with cols[1]:
            st.write(t("Items in queue:", lang), len(st.session_state.queue))
        with cols[2]:
            done = sum(1 for x in st.session_state.queue if x["status"]=="done")
            st.write(t("Completed:", lang), done)

    # --- PDF Summary ---
    summary = {
        "Application ID": str(uuid.uuid4())[:8].upper(),
        "Decision": decision.upper(),
        "Risk Score": f"{int(score*100)} / 100",
        "Monthly Income (₹)": income,
        "Requested Loan (₹)": loan_amount,
        "Doc Type": ("N/A" if id_file is None else ("PAN/Aadhaar" if doc_ok else "Unverified")),
        "Face Similarity": f"{int(face_sim*100)}%",
        "Timestamp (UTC)": datetime.utcnow().isoformat(timespec="seconds")
    }
    pdf_bytes = generate_pdf_summary(summary)
    st.download_button(
        label=t("Download Application Summary (PDF)", lang),
        data=pdf_bytes,
        file_name="application_summary.pdf",
        mime="application/pdf",
        use_container_width=True
    )

st.markdown("---")
st.caption(t("Security: Sensitive fields are masked in the UI. Data is processed only for eligibility evaluation.", lang))
