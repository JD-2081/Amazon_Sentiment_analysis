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

# -------------------- ADVANCED CSS --------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.55), rgba(0,0,0,0.55)),
        url("https://raw.githubusercontent.com/JD-2081/Amazon_Sentiment_analysis/main/static/images/img.png");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    .glass-box {
        background: rgba(255,255,255,0.12);
        backdrop-filter: blur(16px);
        padding: 35px;
        border-radius: 22px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.55);
        border: 1px solid rgba(255,255,255,0.2);
        color: white;
        max-width: 450px;
        margin: auto;
    }

    h1 {
        text-align: center;
        color: white;
        font-weight: 600;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.4);
    }

    textarea {
        border-radius: 14px !important;
        font-size: 16px !important;
    }

    button[kind="primary"] {
        background: linear-gradient(45deg, #ff9900, #ffb347);
        border-radius: 30px;
        font-weight: bold;
        transition: 0.3s ease;
    }

    button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255,153,0,0.4);
    }

    .positive {
        color: #00ff88;
        font-size: 26px;
        font-weight: bold;
        text-align: center;
    }

    .negative {
        color: #ff4d4d;
        font-size: 26px;
        font-weight: bold;
        text-align: center;
    }

    .confidence {
        text-align: center;
        font-size: 18px;
        margin-top: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------- UI CONTAINER --------------------
st.markdown("<div class='glass-box'>", unsafe_allow_html=True)
st.markdown("<h1>📦 Review Analysis</h1>", unsafe_allow_html=True)

# -------------------- GOOGLE DRIVE LINKS --------------------
MODEL_URL = "https://drive.google.com/uc?id=1oW9aU_5tXses81z6_Yd4xX2GCIEBZ13S"
TOKENIZER_URL = "https://drive.google.com/uc?id=14bFQK4ewg9fRm3Xdzhg-BpqfxONbVA60"

MODEL_PATH = "sentiment_model.h5"
TOKENIZER_PATH = "tokenizer.pkl"

# -------------------- LOAD MODEL (MEMORY SAFE) --------------------
@st.cache_resource(show_spinner=False)
def load_resources():
    if not os.path.exists(MODEL_PATH):
        r = requests.get(MODEL_URL)
        open(MODEL_PATH, "wb").write(r.content)

    if not os.path.exists(TOKENIZER_PATH):
        r = requests.get(TOKENIZER_URL)
        open(TOKENIZER_PATH, "wb").write(r.content)

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    tokenizer = pickle.load(open(TOKENIZER_PATH, "rb"))
    return model, tokenizer

model, tokenizer = load_resources()

# -------------------- TEXT CLEANING --------------------
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower().strip()

# -------------------- SESSION STATE --------------------
if "review_text" not in st.session_state:
    st.session_state.review_text = ""

# -------------------- INPUT --------------------
review = st.text_area(
    "",
    placeholder="How was your product?",
    height=140,
    key="review_text"
)

# -------------------- BUTTONS --------------------
col1, col2 = st.columns(2)
analyze = col1.button("✨ Analyze Now", use_container_width=True)
clear = col2.button("🗑️ Clear", use_container_width=True)

if clear:
    st.session_state.review_text = ""

# -------------------- PREDICTION --------------------
if analyze:
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        cleaned = clean_text(review)
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=200)

        with st.spinner("🔍 Analyzing sentiment..."):
            prediction = float(model.predict(padded)[0][0])

        st.markdown("<hr>", unsafe_allow_html=True)

        confidence = prediction if prediction > 0.5 else (1 - prediction)
        st.progress(confidence)

        if prediction > 0.5:
            st.markdown(
                f"""
                <div class="positive">Positive 😊</div>
                <div class="confidence">Confidence: <b>{int(confidence*100)}%</b></div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="negative">Negative ☹️</div>
                <div class="confidence">Confidence: <b>{int(confidence*100)}%</b></div>
                """,
                unsafe_allow_html=True
            )

st.markdown("</div>", unsafe_allow_html=True)
