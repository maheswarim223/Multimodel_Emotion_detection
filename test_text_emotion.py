import joblib

# load model + vectorizer
model = joblib.load("emotion_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

while True:
    text = input("\nEnter text (type 'exit' to stop): ")

    if text == "exit":
        break

    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)

    print("Predicted Emotion:", prediction[0])
