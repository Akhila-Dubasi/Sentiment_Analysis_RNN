import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Mental Health AI Monitor",
    page_icon="🧠",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #0f172a;
    color: white;
}

h1, h2, h3, h4 {
    color: #38bdf8;
}

.stTextArea textarea {
    background-color: #1e293b;
    color: white;
    border-radius: 12px;
    border: 2px solid #38bdf8;
    font-size: 16px;
}

.stButton>button {
    background: linear-gradient(90deg, #06b6d4, #3b82f6);
    color: white;
    border-radius: 10px;
    border: none;
    height: 3em;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
}

.stButton>button:hover {
    background: linear-gradient(90deg, #3b82f6, #06b6d4);
    color: white;
}

.block-container {
    padding-top: 2rem;
}

.css-1d391kg {
    background-color: #111827;
}

.info-box {
    background-color: #1e293b;
    padding: 18px;
    border-radius: 12px;
    border-left: 5px solid #38bdf8;
    margin-bottom: 20px;
}

.result-box {
    background-color: #1e293b;
    padding: 25px;
    border-radius: 15px;
    border: 2px solid #38bdf8;
}

.tip-box {
    background-color: #172554;
    padding: 18px;
    border-radius: 12px;
    margin-top: 15px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL AND PREPROCESSING OBJECTS
# --------------------------------------------------

model = tf.keras.models.load_model("rnn_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

MAX_LEN = 100

# --------------------------------------------------
# TEXT PREPROCESSING FUNCTION
# --------------------------------------------------

def preprocess_text(text):
    sequence = tokenizer.texts_to_sequences([text])

    padded = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding='post',
        truncating='post'
    )

    return padded

# --------------------------------------------------
# GUIDANCE MESSAGES
# --------------------------------------------------

guidance = {
    "Anxiety":
        "💙 Take a short break and talk with someone you trust.",

    "Depression":
        "🌿 Try a small positive activity such as a walk or journaling.",

    "Stress":
        "🧘 Practice deep breathing and organize one task at a time.",

    "Suicidal":
        "🚨 Please seek immediate support from a mental health professional or trusted person.",

    "Normal":
        "😊 Keep maintaining healthy habits and positive routines.",

    "Bipolar":
        "🌙 Maintain regular sleep patterns and follow professional guidance.",

    "Personality Disorder":
        "🤝 Focus on self-awareness and professional emotional support."
}

# ==================================================
# HEADER SECTION
# ==================================================

st.markdown("""
<div style='text-align:center;'>

<h1>🧠 AI-Based Mental Health Sentiment Monitoring</h1>

<h4 style='color:#cbd5e1;'>
Emotion Detection using Simple Recurrent Neural Networks (RNN)
</h4>

</div>
""", unsafe_allow_html=True)

# ==================================================
# ABOUT SECTION
# ==================================================

st.markdown("---")

st.markdown("""
<div class='info-box'>

<h2>📘 About This Project</h2>

<ul>
<li>🤖 <b>Emotional AI</b> helps machines understand human emotions from text.</li>

<li>💬 <b>NLP Applications</b> include chatbots, healthcare systems, recommendation systems, and sentiment analysis.</li>

<li>🔁 <b>RNN Models</b> process sequential text by remembering previous words in a sentence.</li>

<li>🩺 This system helps in <b>mental wellness monitoring</b> and emotional assessment.</li>

</ul>

</div>
""", unsafe_allow_html=True)

# ==================================================
# INPUT SECTION
# ==================================================

st.markdown("---")

st.markdown("""
<h2>✍ Enter Your Thoughts or Feelings</h2>
""", unsafe_allow_html=True)

st.info("""
📝 Sample Inputs:

• I feel nervous and worried about my future.  
• I am very happy today.  
• Nothing seems meaningful anymore.  
• I feel stressed because of my workload.  
""")

user_text = st.text_area(
    "Your Text",
    height=180,
    placeholder="Type your feelings or thoughts here..."
)

# ==================================================
# PREDICTION BUTTON
# ==================================================

predict_btn = st.button("🔍 Analyze Emotion")

# ==================================================
# PREDICTION SECTION
# ==================================================

if predict_btn:

    if user_text.strip() == "":
        st.warning("⚠ Please enter some text.")
    else:

        processed_text = preprocess_text(user_text)

        prediction = model.predict(processed_text, verbose=0)

        predicted_index = np.argmax(prediction)

        confidence = float(np.max(prediction))

        predicted_emotion = label_encoder.inverse_transform(
            [predicted_index]
        )[0]

        # --------------------------------------------------
        # RESULT SECTION
        # --------------------------------------------------

        st.markdown("---")

        st.markdown("""
        <h2>📊 Prediction Results</h2>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "🧠 Emotion",
            predicted_emotion
        )

        col2.metric(
            "📈 Confidence",
            f"{confidence*100:.2f}%"
        )

        col3.metric(
            "💡 Status",
            predicted_emotion
        )

        # --------------------------------------------------
        # VISUALIZATION
        # --------------------------------------------------

        st.markdown("---")

        st.markdown("""
        <h2>📉 Emotion Probability Visualization</h2>
        """, unsafe_allow_html=True)

        emotions = label_encoder.classes_

        probabilities = prediction[0]

        chart_df = pd.DataFrame({
            "Emotion": emotions,
            "Probability": probabilities
        })

        st.bar_chart(
            chart_df.set_index("Emotion")
        )

        fig, ax = plt.subplots(figsize=(8, 4))

        ax.plot(
            emotions,
            probabilities,
            marker="o",
            linewidth=3
        )

        ax.set_title("Sentiment Confidence Graph")

        ax.set_ylabel("Probability")

        ax.set_xlabel("Emotion")

        st.pyplot(fig)

        # --------------------------------------------------
        # GUIDANCE SECTION
        # --------------------------------------------------

        st.markdown("---")

        st.markdown("""
        <h2>💙 Emotional Guidance</h2>
        """, unsafe_allow_html=True)

        message = guidance.get(
            predicted_emotion,
            "🌟 Stay mindful and take care of your emotional well-being."
        )

        st.success(message)

        # --------------------------------------------------
        # POSITIVE ACTIVITIES
        # --------------------------------------------------

        st.markdown("""
        <div class='tip-box'>

        <h3>🌈 Positive Activity Suggestions</h3>

        <ul>
        <li>🚶 Take a short walk</li>
        <li>🧘 Practice mindfulness</li>
        <li>🎵 Listen to calming music</li>
        <li>📞 Talk with a trusted friend</li>
        <li>📖 Read something inspiring</li>
        </ul>

        </div>
        """, unsafe_allow_html=True)

        # --------------------------------------------------
        # WELLNESS TIPS
        # --------------------------------------------------

        st.markdown("""
        <div class='tip-box'>

        <h3>✨ Wellness Tips</h3>

        <ul>
        <li>💤 Maintain healthy sleep habits</li>
        <li>💧 Stay hydrated</li>
        <li>🏃 Exercise regularly</li>
        <li>📅 Organize your daily tasks</li>
        <li>🌿 Take breaks from stress</li>
        <li>👨‍⚕ Seek professional help if needed</li>
        </ul>

        </div>
        """, unsafe_allow_html=True)

# ==================================================
# FOOTER
# ==================================================

st.markdown("---")

st.markdown("""
<div style='text-align:center; color:#94a3b8;'>

Made with ❤️ using Streamlit, TensorFlow & NLP

</div>
""", unsafe_allow_html=True)
