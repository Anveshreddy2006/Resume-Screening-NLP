AI Resume Screening System (NLP + Machine Learning)

Recruiters in modern organizations receive thousands of resumes for every job opening. Manually reviewing these resumes is time-consuming, inefficient, and often leads to delayed hiring decisions.

This project provides an AI-powered Resume Screening System that automatically analyzes uploaded resumes and predicts the most relevant technical job role using Natural Language Processing (NLP) and Machine Learning.

The system enables recruiters to quickly filter large volumes of resumes and identify suitable candidates efficiently, significantly reducing manual screening effort.

🎯 Problem Statement

Traditional resume screening requires recruiters to manually read each resume, which creates several challenges:

Extremely high time consumption when dealing with thousands of applications

Difficulty in consistently matching candidate skills with job requirements

Increased hiring delays

Risk of missing qualified candidates due to manual oversight

This system solves the above problems by automating resume classification, helping recruiters shortlist candidates faster and more accurately.

🌐 Live Demo

Deployed Application:
https://resume-screening-nlp-1.onrender.com/

🚀 Key Features

Upload resume in PDF format

Automatic resume text extraction

NLP-based skill and content analysis

Automatic job role prediction

Prediction confidence visualization

Helps recruiters quickly shortlist relevant candidates

🧠 How It Works

User uploads a resume (PDF).

Resume text is extracted using PDF processing libraries.

Text is cleaned and processed using NLP preprocessing techniques.

TF-IDF converts resume text into numerical vectors.

A trained Machine Learning classification model predicts the most relevant job role.

Prediction confidence scores are displayed for recruiter decision support.

🛠 Tech Stack

Python

Streamlit

Scikit-learn

Natural Language Processing (TF-IDF)

▶ Run Locally
pip install -r requirements.txt
streamlit run app.py
☁ Deployment

The application is deployed using Render.
It can also be deployed using:

Render


📈 Future Improvements

Resume-Job Description matching system

Candidate ranking based on job requirement similarity

Multi-resume batch screening for recruiters

Advanced deep learning NLP models

👤 Author

Anvesh Reddy
