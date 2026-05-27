import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import re
import string
import os

from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ==================================================
# SAFE NLTK DOWNLOAD
# ==================================================

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Mental Health Sentiment Monitoring",
    page_icon="🧠",
    layout="wide"
)

# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown("""
<style>

.main {
    background-color: #F5F7FA;
}

h1 {
    color: #0D47A1;
    text-align: center;
    font-weight: bold;
}

h2, h3 {
    color: #1565C0;
}

.stButton>button {
    background-color: #1976D2;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
    border: none;
}

.stButton>button:hover {
    background-color: #0D47A1;
    color: white;
}

.stTextArea textarea {
    border-radius: 12px;
    border: 2px solid #1976D2;
    font-size: 16px;
}

.metric-container {
    background-color: #E3F2FD;
    padding: 15px;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# LOAD MODEL & FILES
# ==================================================

MODEL_PATH = "rnn_model.h5"

if not os.path.exists(MODEL_PATH):
    st.error("❌ rnn_model.h5 file not found")
    st.stop()

if not os.path.exists("tokenizer.pkl"):
    st.error("❌ tokenizer.pkl file not found")
    st.stop()

if not os.path.exists("label_encoder.pkl"):
    st.error("❌ label_encoder.pkl file not found")
    st.stop()

# Load model
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# ==================================================
# SETTINGS
# ==================================================

MAX_LEN = 100

stop_words = set(stopwords.words('english'))

# ==================================================
# TEXT PREPROCESSING
# ==================================================

def preprocess_text(text):

    # Lowercase
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

    # Convert to sequence
    sequence = tokenizer.texts_to_sequences(
        [" ".join(tokens)]
    )

    # Padding
    padded = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding='post',
        truncating='post'
    )

    return padded

# ==================================================
# GUIDANCE MESSAGES
# ==================================================

guidance = {

    "Anxiety":
    "Take a short break and talk with someone you trust.",

    "Depression":
    "Try a small positive activity such as a walk or journaling.",

    "Stress":
    "Practice deep breathing and organize one task at a time.",

    "Suicidal":
    "Please seek immediate support from a mental health professional or trusted person.",

    "Normal":
    "Keep maintaining healthy habits and positive routines.",

    "Bipolar":
    "Maintain regular sleep patterns and follow professional guidance.",

    "Personality Disorder":
    "Focus on self-awareness and professional emotional support."
}

# ==================================================
# HEADER SECTION
# ==================================================

st.title("🧠 AI-Based Mental Health Sentiment Monitoring System")

st.subheader(
    "Emotion Detection using Simple Recurrent Neural Networks"
)

# ==================================================
# ABOUT PROJECT SECTION
# ==================================================

st.markdown("---")

st.header("📘 About the Project")

st.info("""
### 🌟 Importance of Emotional AI

• Helps machines understand human emotions  
• Supports mental wellness monitoring  
• Enables early emotional assessment  

### 🤖 NLP Applications

• Sentiment Analysis  
• Emotion Detection  
• Chatbots & Virtual Assistants  
• Healthcare Applications  
• Social Media Analysis  

### 🔁 Role of RNN in Sequence Learning

• Processes sequential text data  
• Remembers previous words using hidden states  
• Learns emotional patterns in sentences  
""")

# ==================================================
# USER INPUT SECTION
# ==================================================

st.markdown("---")

st.header("✍️ User Text Input Area")

st.success("""
### 💡 Sample Sentences

• I feel nervous and worried about my future.  
• I am very happy today.  
• Nothing seems meaningful anymore.  
• I feel stressed because of my workload.  
""")

user_text = st.text_area(
    "📝 Enter Your Thoughts or Feelings",
    height=200,
    placeholder="Enter your thoughts or feelings here..."
)

# ==================================================
# BUTTON SECTION
# ==================================================

predict_btn = st.button("🔍 Analyze Emotion")

# ==================================================
# PREDICTION SECTION
# ==================================================

if predict_btn:

    if user_text.strip() == "":

        st.warning("⚠️ Please enter some text.")

    else:

        # Preprocess
        processed_text = preprocess_text(user_text)

        # Prediction
        prediction = model.predict(
            processed_text,
            verbose=0
        )

        predicted_index = np.argmax(prediction)

        confidence = float(np.max(prediction))

        predicted_emotion = label_encoder.inverse_transform(
            [predicted_index]
        )[0]

        # ==================================================
        # PREDICTION OUTPUT
        # ==================================================

        st.markdown("---")

        st.header("📊 Prediction Result")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Emotion Detected",
            predicted_emotion
        )

        col2.metric(
            "Confidence Score",
            f"{confidence*100:.2f}%"
        )

        col3.metric(
            "Status",
            "Analyzed"
        )

        # ==================================================
        # VISUALIZATION SECTION
        # ==================================================

        st.markdown("---")

        st.header("📈 Visualization Area")

        emotions = label_encoder.classes_

        probabilities = prediction[0]

        chart_df = pd.DataFrame({
            "Emotion": emotions,
            "Probability": probabilities
        })

        st.subheader("📌 Emotion Probability Chart")

        st.bar_chart(
            chart_df.set_index("Emotion")
        )

        st.subheader("📉 Sentiment Confidence Graph")

        fig, ax = plt.subplots(figsize=(8, 4))

        ax.plot(
            emotions,
            probabilities,
            marker="o"
        )

        ax.set_title("Sentiment Confidence Graph")

        ax.set_ylabel("Probability")

        ax.set_xlabel("Emotion")

        st.pyplot(fig)

        # ==================================================
        # GUIDANCE SECTION
        # ==================================================

        st.markdown("---")

        st.header("💡 Emotional Guidance")

        message = guidance.get(
            predicted_emotion,
            "Stay mindful and take care of your emotional well-being."
        )

        st.success(message)

        # ==================================================
        # POSITIVE ACTIVITIES
        # ==================================================

        st.subheader("🌿 Positive Activity Suggestions")

        st.markdown("""
• ✔ Take a short walk  
• ✔ Practice mindfulness  
• ✔ Listen to calming music  
• ✔ Talk with a trusted friend  
• ✔ Drink enough water  
• ✔ Maintain healthy sleep  
""")

        # ==================================================
        # WELLNESS TIPS
        # ==================================================

        st.subheader("🩺 Wellness Tips")

        st.markdown("""
• Maintain healthy sleep habits  
• Exercise regularly  
• Avoid excessive stress exposure  
• Stay connected with loved ones  
• Seek professional help if needed  
• Practice positive thinking  
""")
