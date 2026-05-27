# ==================================================
# HEADER SECTION
# ==================================================

st.title("🧠 AI-Based Mental Health Sentiment Monitoring System")

st.markdown("""
### Emotion Detection using Simple Recurrent Neural Networks
""")

st.markdown("---")

# ==================================================
# ABOUT PROJECT
# ==================================================

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

        processed_text = preprocess_text(user_text)

        prediction = model.predict(processed_text, verbose=0)

        predicted_index = np.argmax(prediction)

        confidence = float(np.max(prediction))

        predicted_emotion = label_encoder.inverse_transform(
            [predicted_index]
        )[0]

        # ------------------------------------------------
        # PREDICTION OUTPUT
        # ------------------------------------------------

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

        # ------------------------------------------------
        # VISUALIZATION SECTION
        # ------------------------------------------------

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

        # ------------------------------------------------
        # GUIDANCE SECTION
        # ------------------------------------------------

        st.markdown("---")

        st.header("💡 Emotional Guidance")

        message = guidance.get(
            predicted_emotion,
            "Stay mindful and take care of your emotional well-being."
        )

        st.success(message)

        # ------------------------------------------------
        # POSITIVE ACTIVITIES
        # ------------------------------------------------

        st.subheader("🌿 Positive Activity Suggestions")

        st.markdown("""
• ✔ Take a short walk  
• ✔ Practice mindfulness  
• ✔ Listen to calming music  
• ✔ Talk with a trusted friend  
• ✔ Drink enough water  
• ✔ Maintain healthy sleep  
""")

        # ------------------------------------------------
        # WELLNESS TIPS
        # ------------------------------------------------

        st.subheader("🩺 Wellness Tips")

        st.markdown("""
• Maintain healthy sleep habits  
• Exercise regularly  
• Avoid excessive stress exposure  
• Stay connected with loved ones  
• Seek professional help if needed  
• Practice positive thinking  
""")
