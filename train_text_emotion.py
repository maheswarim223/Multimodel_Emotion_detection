import pandas as pd
import nltk
import joblib
import string

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Download stopwords
nltk.download("stopwords")

# ==========================
# LOAD DATASET
# ==========================
df = pd.read_csv("Mental Health Dataset.csv")  # change filename if needed

print("Dataset shape:", df.shape)
print(df.head())

# ==========================
# SELECT FEATURES + LABEL
# ==========================
X = df["posts"]        # text
y = df["predicted"]    # emotion label

# ==========================
# TEXT CLEANING FUNCTION
# ==========================
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

X = X.apply(clean_text)

# ==========================
# TRAIN TEST SPLIT
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================
# TF-IDF FEATURE EXTRACTION
# ==========================
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ==========================
# TRAIN MODEL
# ==========================
model = LogisticRegression()
model.fit(X_train_vec, y_train)

print("\n✅ Training completed!")

# ==========================
# TEST MODEL
# ==========================
y_pred = model.predict(X_test_vec)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ==========================
# SAVE MODEL
# ==========================
joblib.dump(model, "emotion_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\n✅ Model saved!")
