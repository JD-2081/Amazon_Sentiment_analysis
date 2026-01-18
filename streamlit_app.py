import streamlit as st
import tensorflow as tf
import pickle
import re
import os
import requests
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Amazon Sentiment Analysis",
    page_icon="📦",
    layout="centered"
)

# -------------------- BACKGROUND + CSS --------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.5), rgba(0,0,0,0.5)),
                    url("https://raw.githubusercontent.com/JD-2081/Amazon_Sentiment_analysis/main/static/images/img.png");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    .glass-box {
        background: rgba(255,255,255,0.15);
        backdrop-filter: blur(15px);
        padding: 35px;
        border-radius: 20px;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.5);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------- TITLE --------------------
st.markdown(
    "<div class='glass-box'><h1>📦 Amazon Review Sentiment Analysis</h1></div>",
    unsafe_allow_html=True
)

st.write("")

# -------------------- GOOGLE DRIVE FILES --------------------
MODEL_URL = "https://drive.google.com/uc?id=1oW9aU_5tXses81z6_Yd4xX2GCIEBZ13S"
TOKENIZER_URL = "https://drive.google.com/uc?id=14bFQK4ewg9fRm3Xdzhg-BpqfxONbVA60"

MODEL_PATH = "sentiment_model.h5"
TOKENIZER_PATH = "tokenizer.pkl"

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_resources():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading model..."):
            r = requests.get(MODEL_URL)
            open(MODEL_PATH, "wb").write(r.content)

    if not os.path.exists(TOKENIZER_PATH):
        with st.spinner("📥 Downloading tokenizer..."):
            r = requests.get(TOKENIZER_URL)
            open(TOKENIZER_PATH, "wb").write(r.content)

    model = tf.keras.models.load_model(MODEL_PATH)
    tokenizer = pickle.load(open(TOKENIZER_PATH, "rb"))
    return model, tokenizer

model, tokenizer = load_resources()

# -------------------- TEXT CLEANING --------------------
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower().strip()

# -------------------- UI FORM --------------------
st.markdown("<div class='glass-box'>", unsafe_allow_html=True)

review = st.text_area(
    "✍️ How was your product?",
    height=140,
    placeholder="Type your Amazon review here..."
)

col1, col2 = st.columns(2)

analyze = col1.button("✨ Analyze Now")
clear = col2.button("🗑️ Clear")

# -------------------- CLEAR --------------------
if clear:
    st.experimental_rerun()

# -------------------- PREDICTION --------------------
if analyze:
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        cleaned = clean_text(review)
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=200)

        with st.spinner("🔍 Analyzing sentiment..."):
            prediction = model.predict(padded)[0][0]

        if prediction > 0.5:
            st.success(f"😊 Positive Sentiment\n\nConfidence: **{prediction*100:.1f}%**")
        else:
            st.error(f"☹️ Negative Sentiment\n\nConfidence: **{(1-prediction)*100:.1f}%**")

st.markdown("</div>", unsafe_allow_html=True)
