# Fake News Detection (BERT + Logistic Regression)

Lightweight project demonstrating fake news detection using a DistilBERT classifier and a Logistic Regression baseline.

## Project structure

- `app.py`: Flask web app (uses the Hugging Face pipeline) with language detection and translation.
- `train.py`: Train a DistilBERT sequence classification model (uses `transformers` + `datasets`). Saves model to `model/` and confusion matrix to `results/`.
- `train_lr.py`: Train a TF-IDF + Logistic Regression baseline. Saves `model/lr_model.pkl` and `model/tfidf.pkl`.
- `predict.py`: Simple interactive prediction using the transformer model in `model/`.
- `predict_lr.py`: Simple interactive prediction using the saved LR model and TF-IDF vectorizer.
- `data/news.csv`: Source dataset (used by training scripts).
- `model/`: Model artifacts (saved transformers model, tokenizer, or `model.safetensors`).
- `results/`: Output artifacts (confusion matrices, plots).
- `templates/index.html`: Template used by the Flask app.

## Requirements

- Python 3.8+ recommended
- Recommended packages (install via pip):

```
pip install torch transformers datasets scikit-learn pandas matplotlib seaborn flask deep-translator langdetect joblib safetensors
```

- If you plan to train with a GPU, install a CUDA-enabled `torch` wheel from https://pytorch.org/.

## Setup (Windows)

1. Create and activate a virtual environment (PowerShell):

```powershell
python -m venv venv
.\\venv\\Scripts\\Activate.ps1
pip install -U pip
pip install -r requirements.txt   # if you have one, otherwise run the pip install above
```

Or (CMD):

```cmd
python -m venv venv
venv\\Scripts\\activate.bat
pip install -U pip
```

## How to run

- Train transformer model (DistilBERT):

```
python train.py
```

This will:
- load `data/news.csv` (small sample by default)
- train a DistilBERT classifier for 1 epoch (see `train.py` for hyperparams)
- save the model and tokenizer to `model/`
- save a confusion matrix image to `results/confusion_matrix.png`

- Train logistic regression baseline:

```
python train_lr.py
```

This will save `model/lr_model.pkl` and `model/tfidf.pkl` and a confusion matrix to `results/lr_confusion_matrix.png`.

- Predict using the transformer model (interactive):

```
python predict.py
```

Enter the news text when prompted. The script loads the model from `model/` and prints prediction + confidence.

- Predict using the logistic regression model (interactive):

```
python predict_lr.py
```

- Run the web app (Flask):

```
python app.py
```

Then open http://127.0.0.1:5000/ in your browser. The app will:
- detect input language with `langdetect`
- translate to English using `deep-translator` when necessary
- run the transformer pipeline on the (possibly translated) text and display label + confidence

## Model files

- The Flask app and `predict.py` expect a Hugging Face-style model folder at `model/` (this can contain `pytorch_model.bin`, `model.safetensors`, `config.json`, tokenizer files, etc.). If you already have `model/model.safetensors` and the tokenizer files in `model/`, the scripts should load them directly.

If you see errors about missing weights or tokenizer, either run `train.py` or place the appropriate model files inside the `model/` directory.

## Notes & Troubleshooting

- If training is slow or you run out of memory, reduce `max_length` and `per_device_train_batch_size` in `train.py`.
- For inference issues, confirm `transformers` and `torch` versions are compatible and that `model/` contains both weights and tokenizer.
- If the Flask app shows `Error` for prediction, check that `langdetect` and `deep-translator` are installed and that the model loads without exceptions.

## Next steps (suggested)

- Add a `requirements.txt` for reproducible installs.
- Add CLI args to `predict.py` / `train.py` for batch prediction and configurable hyperparameters.

---

If you'd like, I can add a `requirements.txt` file now and/or run a quick smoke test of `predict.py` in this environment. 
