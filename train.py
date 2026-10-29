# ================================
# MENTAL HEALTH SENTIMENT CLASSIFICATION
# ================================

import pandas as pd
import numpy as np
import re
import nltk
import pickle

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Download stopwords (only first time)
nltk.download('stopwords')

# ================================
# 1️⃣ LOAD DATASET
# ================================

# Change file name if needed
df = pd.read_csv("Mental Health Dataset.csv")

print("Dataset Loaded Successfully")
print("Shape:", df.shape)
print(df.head())

# ================================
# 2️⃣ CHECK LABEL DISTRIBUTION
# ================================

print("\nLabel Counts:")
print(df['predicted'].value_counts())

# ================================
# 3️⃣ CLEAN TEXT FUNCTION
# ================================

stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

df['clean_text'] = df['posts'].apply(clean_text)

# ================================
# 4️⃣ FEATURE EXTRACTION (TF-IDF)
# ================================

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_text'])

y = df['predicted']

# ================================
# 5️⃣ TRAIN TEST SPLIT
# ================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================================
# 6️⃣ TRAIN MODEL
# ================================

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("\nModel Training Completed")

# ================================
# 7️⃣ TEST MODEL (AUTOMATIC)
# ================================

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# ================================
# 8️⃣ MANUAL TEST FUNCTION
# ================================

def predict_text(text):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)
    return prediction[0]

print("\nSample Predictions:")
print("Input: I feel very sad and tired")
print("Output:", predict_text("I feel very sad and tired"))

print("\nInput: Today is a beautiful day")
print("Output:", predict_text("Today is a beautiful day"))

# ================================
# 9️⃣ SAVE MODEL
# ================================

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\nModel Saved Successfully (model.pkl & vectorizer.pkl)")

# ================================
# 🔟 INTERACTIVE TESTING
# ================================

while True:
    user_input = input("\nEnter text (type 'exit' to stop): ")

    if user_input.lower() == "exit":
        print("Exiting...")
        break

    result = predict_text(user_input)
    print("Predicted Sentiment:", result)
