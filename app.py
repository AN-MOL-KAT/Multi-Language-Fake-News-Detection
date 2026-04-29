# import streamlit as st
# from transformers import pipeline

# # Page config
# st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# # Load model
# classifier = pipeline("text-classification", model="model")

# # Custom styling
# st.markdown("""
# <style>
# .big-title {font-size:32px; font-weight:bold; text-align:center;}
# .result-box {padding:15px; border-radius:10px; text-align:center; font-size:20px;}
# </style>
# """, unsafe_allow_html=True)

# # Title
# st.markdown('<div class="big-title">📰 Fake News Detection System</div>', unsafe_allow_html=True)
# st.write("Detect whether a news article is **REAL or FAKE** using NLP + Deep Learning")

# # Input
# user_input = st.text_area("✍️ Enter News Text:", height=200)

# col1, col2 = st.columns(2)

# with col1:
#     predict_btn = st.button("🔍 Predict")

# with col2:
#     clear_btn = st.button("🧹 Clear")

# # Clear input
# if clear_btn:
#     st.experimental_rerun()

# # Prediction
# if predict_btn:
#     if user_input.strip() == "":
#         st.warning("⚠️ Please enter some text")
#     else:
#         with st.spinner("Analyzing..."):
#             result = classifier(user_input)[0]

#         label_map = {
#             "LABEL_0": "FAKE",
#             "LABEL_1": "REAL"
#         }

#         prediction = label_map[result["label"]]
#         confidence = round(result["score"] * 100, 2)

#         st.subheader("📊 Result")

#         if prediction == "FAKE":
#             st.markdown(f'<div class="result-box" style="background-color:#ffcccc;">🚨 FAKE NEWS<br>Confidence: {confidence}%</div>', unsafe_allow_html=True)
#         else:
#             st.markdown(f'<div class="result-box" style="background-color:#ccffcc;">✅ REAL NEWS<br>Confidence: {confidence}%</div>', unsafe_allow_html=True)

# # Footer
# st.markdown("---")
# st.caption("Built using NLP + DistilBERT")
from flask import Flask, render_template, request
from transformers import pipeline
from deep_translator import GoogleTranslator
from langdetect import detect

app = Flask(__name__)

# Load model once
classifier = pipeline("text-classification", model="model")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None
    detected_lang = None
    translated_text = None

    if request.method == "POST":
        text = request.form["news"]

        try:
            # Detect language
            detected_lang = detect(text)

            # Translate to English if not English
            if detected_lang != "en":
                translated_text = GoogleTranslator(source='auto', target='en').translate(text)
            else:
                translated_text = text

            # Predict
            result = classifier(translated_text)[0]

            label_map = {
                "LABEL_0": "FAKE",
                "LABEL_1": "REAL"
            }

            prediction = label_map[result["label"]]
            confidence = round(result["score"] * 100, 2)

        except:
            prediction = "Error"
            confidence = 0

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        lang=detected_lang,
        translated=translated_text
    )

if __name__ == "__main__":
    app.run(debug=True)