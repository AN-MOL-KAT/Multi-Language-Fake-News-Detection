import joblib

model = joblib.load("model/lr_model.pkl")
vectorizer = joblib.load("model/tfidf.pkl")

text = input("Enter news: ")

text_tfidf = vectorizer.transform([text])
pred = model.predict(text_tfidf)[0]

label = "REAL" if pred == 1 else "FAKE"
print("Prediction:", label)