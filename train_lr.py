import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load data
df = pd.read_csv("data/news.csv")

# Clean
df = df.dropna(subset=["text", "label"])
df["text"] = df["text"].astype(str)
df["label"] = df["label"].map({"FAKE": 0, "REAL": 1})

# Split
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Predict
y_pred = model.predict(X_test_tfidf)

# Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

os.makedirs("results", exist_ok=True)
plt.figure()
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=["FAKE","REAL"],
            yticklabels=["FAKE","REAL"])
plt.title("LR Confusion Matrix")
plt.savefig("results/lr_confusion_matrix.png")
plt.close()

# Save model
joblib.dump(model, "model/lr_model.pkl")
joblib.dump(vectorizer, "model/tfidf.pkl")