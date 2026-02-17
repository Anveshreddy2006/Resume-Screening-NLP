import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Load dataset
df = pd.read_csv("data/candidate_job_role_dataset.csv")

print("Dataset loaded:", df.shape)

# Use only skills + job_role
df = df[['skills', 'job_role']]
df.dropna(inplace=True)

# 2. Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(df['job_role'])

# 3. Vectorize text
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['skills'])

# 4. Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 6. Evaluate
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

# 7. Save model
os.makedirs("model", exist_ok=True)

pickle.dump(tfidf, open("model/tfidf.pkl", "wb"))
pickle.dump(model, open("model/clf.pkl", "wb"))
pickle.dump(encoder, open("model/encoder.pkl", "wb"))

print("Model saved successfully")
