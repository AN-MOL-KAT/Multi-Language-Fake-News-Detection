import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# -------- Load & Clean Dataset --------
df = pd.read_csv("data/news.csv")

df = df.dropna(subset=["text", "label"])
df["text"] = df["text"].astype(str)

# 🔥 SPEED: use smaller dataset (important)
df = df.sample(1500, random_state=42)

df["label"] = df["label"].map({"FAKE": 0, "REAL": 1})
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)

dataset = Dataset.from_pandas(df)

# -------- Tokenizer --------
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128   # 🔥 shorter input
    )

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.train_test_split(test_size=0.2)

# -------- Model (FASTER than BERT) --------
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)

# -------- Metrics --------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)

    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# -------- Training Config --------
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,  
    num_train_epochs=1,              
    logging_dir="./logs",
    save_strategy="no"             
)

# -------- Trainer --------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics
)

# -------- Train --------
trainer.train()

# Save model
model.save_pretrained("model")
tokenizer.save_pretrained("model")

# -------- Confusion Matrix --------
preds = trainer.predict(dataset["test"])
y_true = preds.label_ids
y_pred = np.argmax(preds.predictions, axis=1)

cm = confusion_matrix(y_true, y_pred)

os.makedirs("results", exist_ok=True)
file_path = "results/confusion_matrix.png"

plt.figure()
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=["FAKE", "REAL"],
            yticklabels=["FAKE", "REAL"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig(file_path)
plt.close()

print(f"Confusion matrix saved at: {file_path}")