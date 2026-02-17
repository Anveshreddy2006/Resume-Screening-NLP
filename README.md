AI Resume Screening System (NLP + ML)

An intelligent AI-powered Resume Screening System that automatically analyzes uploaded resumes and predicts the most relevant technical job role using Machine Learning and Natural Language Processing.

This system helps recruiters quickly filter thousands of resumes and identify suitable candidates efficiently.

🌐 Live Demo

Deployed App:
https://resume-screening-nlp-1.onrender.com/


🚀 Features

Upload Resume in PDF format

Automatic Resume Text Extraction

NLP-based Resume Cleaning & Processing

Machine Learning based Job Role Prediction

Top Prediction Confidence Visualization

Fully deployed Streamlit Web Application

🧠 Tech Stack

Python

Machine Learning (Scikit-learn)

NLP (TF-IDF Vectorization)

Streamlit (Frontend + Deployment)

Pandas / NumPy

PDFPlumber (Resume parsing)

📂 Project Structure
resumeanalyzer/
│
├── app.py
├── train_model.py
├── requirements.txt
├── .gitignore
│
├── data/
│   └── candidate_job_role_dataset.csv
│
├── model/
│   ├── tfidf.pkl
│   ├── clf.pkl
│   └── encoder.pkl
│
└── utils/
    └── preprocess.py

⚙️ Installation

Clone the repository:

git clone https://github.com/Anveshreddy2006/Resume-Screening-NLP.git
cd Resume-Screening-NLP


Install dependencies:

pip install -r requirements.txt


Train the model (if required):

python train_model.py


Run the application:

streamlit run app.py

🌐 Deployment

The application can be deployed using:

Streamlit Cloud

Render

Railway

HuggingFace Spaces

📊 Example Output

Predicted Role: Web Developer

Confidence Chart displaying Top Matching Roles

🎯 Use Case

This project is designed for:

Automated Resume Screening

Recruiter Resume Filtering

Job Role Classification

HR Tech Automation Systems

📌 Future Improvements

Resume-Job Description Matching

Skill Gap Analysis

Multi-resume Batch Screening

Recruiter Dashboard

👤 Author

Anvesh Reddy
GitHub: https://github.com/Anveshreddy2006

⭐ If you like this project

Give it a ⭐ on GitHub — it helps a lot!

Next (important)
