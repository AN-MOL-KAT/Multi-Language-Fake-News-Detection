from transformers import pipeline

classifier = pipeline("text-classification", model="model")

text = input("Enter news: ")
result = classifier(text)[0]

label_map = {
    "LABEL_0": "FAKE",
    "LABEL_1": "REAL"
}

print("Prediction:", label_map[result["label"]])
print("Confidence:", round(result["score"], 3))