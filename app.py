from flask import Flask, render_template, request
import tensorflow as tf
import pickle
import re
import os
import requests
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Google Drive Direct Links
MODEL_URL = "https://drive.google.com/uc?id=1oW9aU_5tXses81z6_Yd4xX2GCIEBZ13S"
TOKENIZER_URL = "https://drive.google.com/uc?id=14bFQK4ewg9fRm3Xdzhg-BpqfxONbVA60"

MODEL_PATH = "sentiment_model.h5"
TOKENIZER_PATH = "tokenizer.pkl"

# Download files if they don't exist
def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename} from cloud...")
        r = requests.get(url)
        with open(filename, "wb") as f:
            f.write(r.content)

download_file(MODEL_URL, MODEL_PATH)
download_file(TOKENIZER_URL, TOKENIZER_PATH)

# Load tokenizer (lightweight)
with open(TOKENIZER_PATH, "rb") as handle:
    tokenizer = pickle.load(handle)

# Lazy-load model to save memory
_model = None
def get_model():
    global _model
    if _model is None:
        print("Loading TensorFlow model into memory...")
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model

# Text cleaning
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower().strip()

# Routes
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    user_review = request.form["review"]
    cleaned_review = clean_text(user_review)
    sequence = tokenizer.texts_to_sequences([cleaned_review])
    padded = pad_sequences(sequence, maxlen=100)

    model = get_model()
    prediction = model.predict(padded)[0][0]

    if prediction > 0.5:
        sentiment = "Positive 😊"
        confidence = f"{prediction * 100:.1f}%"
    else:
        sentiment = "Negative ☹️"
        confidence = f"{(1 - prediction) * 100:.1f}%"

    return render_template(
        "index.html",
        sentiment=sentiment,
        confidence=confidence,
        review=user_review
    )

if __name__ == "__main__":
    app.run()
