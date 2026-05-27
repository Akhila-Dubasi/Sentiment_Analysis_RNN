import streamlit as st
import numpy as np
import re
import string
import pickle
import matplotlib.pyplot as plt
import os
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =========================================================
# SAFE NLTK DOWNLOAD
# =========================================================

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Mental Health Sentiment Monitoring",
    page_icon="🧠",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #F4F7FC;
}

h1 {
    color: #1E3A5F;
    text-align: center;
    font-weight: bold;
}

h2, h3 {
    color: #24476B;
}

.stButton>button {
    background-color: #1E88E5;
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
    border: none;
}

.stButton>button:hover {
    background-color: #1565C0;
}

.stTextArea textarea {
    border-radius: 12px;
    border: 2px solid #1E88E5;
    font-size: 16px;
}

.result-box {
    padding: 20px;
    border-radius: 12px;
    background-color: #E3F2FD;
    color: #0D47A1;
    font-size: 22px;
    font-weight: bold;
}

.guide-box {
    padding: 18px;
    border-radius: 12px;
    background-color: #FFF3E0;
    color: #E65100;
    font-size: 18px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL & FILES
# =========================================================

MODEL_PATH = "mental_health_rnn_model.h5"

# Check model file
if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file 'mental_health_rnn_model.h5' not found.")
    st.stop()

# Check tokenizer
if not os.path.exists("tokenizer.pkl"):
    st.error("❌ tokenizer.pkl not found.")
    st.stop()

# Check label encoder
if not os.path.exists("label_encoder.pkl"):
    st.error("❌ label_encoder.pkl not found.")
    st.stop()

# Load model
model = load_model(MODEL_PATH)

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# =========================================================
# SETTINGS
# =========================================================

MAX_LEN = 100

stop_words = set(stopwords.words('english'))

# =========================================================
# TEXT PREPROCESSING
# =========================================================

def preprocess_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords
    tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    return " ".join(tokens)

# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_emotion(text):

    cleaned = preprocess_text(text)

    # Convert text to sequence
    sequence = tokenizer.texts_to_sequences([cleaned])

    # Padding
    padded = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding='post'
    )

    # Prediction
    prediction = model.predict(padded, verbose=0)

    predicted_index = np.argmax(prediction)

    predicted_label = label_encoder.inverse_transform(
        [predicted_index]
    )[0]

    confidence = np.max(prediction) * 100

    return predicted_label, confidence, prediction[0]

# =========================================================
# GUIDANCE MESSAGES
# =========================================================

guidance = {

    "Anxiety":
    "Take a deep breath. You are stronger than your worries.",

    "Depression":
    "You matter. Small positive steps every day can create change.",

    "Stress":
    "Pause for a moment and prioritize self-care.",

    "Normal":
    "Keep maintaining your positive mental wellness.",

    "Bipolar":
    "Balance and support are important. Stay connected with loved ones.",

    "Suicidal":
    "Please seek support from trusted people or professionals immediately.",

    "Personality disorder":
    "You are not alone. Consistent support can help improve emotional balance."
}

# =========================================================
# HEADER SECTION
# =========================================================

st.title("🧠 AI-Based Mental Health Sentiment Monitoring System")

st.markdown("""
### Emotion Detection using Simple Recurrent Neural Networks
""")

# =========================================================
# ABOUT PROJECT SECTION
# =========================================================

st.header("📘 About the Project")

st.write("""
This project uses Artificial Intelligence and Natural Language Processing (NLP)
to detect emotional sentiments from user text inputs.

### Importance of Emotional AI
Emotional AI helps systems understand human emotions through language patterns,
allowing early identification of mental health concerns.

### NLP Applications
- Mental health monitoring
- Chatbots
- Emotion detection
- Customer feedback analysis
- Social media sentiment analysis

### Role of RNN in Sequence Learning
Recurrent Neural Networks (RNNs) process text sequentially and remember
previous words using hidden states, making them useful for sentiment analysis.
""")

# =========================================================
# USER INPUT SECTION
# =========================================================

st.header("✍ User Text Input Area")

st.markdown("""
#### Sample Sentences

- I feel emotionally exhausted and lonely.
- Life feels beautiful and peaceful today.
- I am stressed about my future.
- Nobody understands my pain.
""")

user_input = st.text_area(
    "Enter your thoughts or feelings here...",
    height=200
)

# =========================================================
# ANALYZE BUTTON
# =========================================================

if st.button("🔍 Analyze Emotion"):

    if user_input.strip() == "":
        st.warning("⚠ Please enter some text.")

    else:

        # Prediction
        emotion, confidence, probs = predict_emotion(user_input)

        # =================================================
        # PREDICTION OUTPUT
        # =================================================

        st.header("📊 Prediction Output")

        st.markdown(f"""
        <div class="result-box">
        Emotion Detected: {emotion}<br><br>
        Confidence Score: {confidence:.2f}%
        </div>
        """, unsafe_allow_html=True)

        # Emotional status
        if confidence > 85:
            status = "High Confidence Detection"
        elif confidence > 60:
            status = "Moderate Confidence Detection"
        else:
            status = "Low Confidence Detection"

        st.success(f"Emotional Status: {status}")

        # =================================================
        # VISUALIZATION SECTION
        # =================================================

        st.header("📈 Sentiment Confidence Graph")

        labels = label_encoder.classes_

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.bar(labels, probs)

        ax.set_xlabel("Emotion Categories")
        ax.set_ylabel("Probability")
        ax.set_title("Emotion Prediction Probabilities")

        plt.xticks(rotation=45)

        st.pyplot(fig)

        # =================================================
        # GUIDANCE SECTION
        # =================================================

        st.header("💡 Emotional Guidance")

        message = guidance.get(
            emotion,
            "Stay positive and take care of your mental wellness."
        )

        st.markdown(f"""
        <div class="guide-box">
        {message}
        </div>
        """, unsafe_allow_html=True)
