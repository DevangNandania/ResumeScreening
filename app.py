import streamlit as st
import joblib
from preprocess import clean_text
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2

# Load model
model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

st.set_page_config(
    page_title="AI Resume Screening",
    layout="centered"
)

st.title("AI Resume Screening System")

st.markdown("Upload a resume (PDF format)")

# PDF Upload
uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

if st.button("Analyze Resume"):

    if uploaded_file is not None:

        resume_text = extract_text_from_pdf(uploaded_file)

        cleaned = clean_text(resume_text)
        vector = vectorizer.transform([cleaned])

        prediction = model.predict(vector)[0]
        prob = model.predict_proba(vector).max()

        st.success(f"Predicted Category: {prediction}")
    else:
        st.warning("Please upload a PDF resume.")