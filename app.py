import streamlit as st
import pickle
import re
import pandas as pd
import pdfplumber

# ---------- Load models ----------
tfidf = pickle.load(open("model/tfidf.pkl", "rb"))
model = pickle.load(open("model/clf.pkl", "rb"))
encoder = pickle.load(open("model/encoder.pkl", "rb"))

# ---------- Clean text ----------
def clean_resume(text):
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'[^A-Za-z0-9 ]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower()

# ---------- Page config ----------
st.set_page_config(page_title="AI Resume Screening System", layout="wide")

st.title("AI Resume Screening System")
st.write("Upload resume (PDF) to automatically classify the candidate into the most relevant job role.")

# ---------- Upload ----------
uploaded_file = st.file_uploader("Upload Resume", type=["pdf"])

if uploaded_file:

    with pdfplumber.open(uploaded_file) as pdf:
        text = ""
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t

    st.subheader("Extracted Resume Text (Preview)")
    st.write(text[:1600])

    if st.button("Analyze Resume"):

        cleaned = clean_resume(text)
        vector = tfidf.transform([cleaned])

        prediction = model.predict(vector)
        probabilities = model.predict_proba(vector)[0]

        category = encoder.inverse_transform(prediction)[0]
        match_score = max(probabilities) * 100

        # ---------- Result section ----------
        col1, col2 = st.columns(2)

        with col1:
            st.success(f"Predicted Role: {category}")

        with col2:
            st.info(f"Match Score: {match_score:.2f}%")

        # ---------- Confidence chart ----------
        proba_df = pd.DataFrame({
            "Category": encoder.classes_,
            "Confidence": probabilities
        }).sort_values(by="Confidence", ascending=False)

        st.subheader("Top Prediction Confidence")
        st.bar_chart(
            proba_df.set_index("Category").head(5),
            use_container_width=True
        )
