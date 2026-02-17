import streamlit as st
import pickle
import re
import pandas as pd
import pdfplumber

# Load saved model files
tfidf = pickle.load(open("model/tfidf.pkl", "rb"))
model = pickle.load(open("model/clf.pkl", "rb"))
encoder = pickle.load(open("model/encoder.pkl", "rb"))

def clean_resume(text):
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'[^A-Za-z0-9 ]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower()

st.set_page_config(page_title="Resume Screening System", layout="wide")

st.title("AI Resume Screening System")

uploaded_file = st.file_uploader("Upload Resume", type=["pdf"])

if uploaded_file:

    with pdfplumber.open(uploaded_file) as pdf:
        text = ""
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t

    st.subheader("Extracted Resume Text (Preview)")
    st.write(text[:1000])

    if st.button("Analyze Resume"):

        cleaned = clean_resume(text)
        vector = tfidf.transform([cleaned])

        prediction = model.predict(vector)
        probabilities = model.predict_proba(vector)[0]

        category = encoder.inverse_transform(prediction)

        st.success(f"Predicted Category: {category[0]}")
        match_score = max(probabilities) * 100
        st.info(f"Resume Match Score: {match_score:.2f}%")


        # Confidence chart
        proba_df = pd.DataFrame({
            "Category": encoder.classes_,
            "Confidence": probabilities
        }).sort_values(by="Confidence", ascending=False)
        st.subheader("Prediction Confidence")
        st.bar_chart(proba_df.set_index("Category").head(5))
